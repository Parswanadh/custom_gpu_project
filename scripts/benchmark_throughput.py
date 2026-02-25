#!/usr/bin/env python3
"""
BitbyBit GPU — Comprehensive Throughput Benchmark
===================================================
Models ALL hardware optimizations and compares their impact:

  1. Deep Pipelining:       5x throughput (1 op/cycle vs 5 cycles/op)
  2. Memory BW (4-wide):    4x weight fetch bandwidth
  3. INT4 Parallel (4x):    4 multiplies per cycle in INT4 mode
  4. Weight Pruning:        Magnitude-based at 30%/50%/70%
  5. 2:4 Structured Sparsity: Guaranteed 50% weight zeros
  6. Combined optimizations

Usage:
    python scripts/benchmark_throughput.py
    python scripts/benchmark_throughput.py --verbose
"""

import os, sys, time, argparse
import numpy as np

# ============================================================================
# Constants (OPT-125M architecture)
# ============================================================================
DIM = 768
FFN_DIM = 3072
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS
NUM_LAYERS = 12
VOCAB_SIZE = 50272

# Clock and hardware specs
CLOCK_MHZ = 100
PIPELINE_DEPTH = 5
MEMORY_WIDTH = 4    # 4 values per memory read cycle
INT4_PARALLEL = 4   # 4 multiplies per cycle in INT4 mode

# ============================================================================
# Sparsity Techniques
# ============================================================================

def magnitude_prune(weights, sparsity_pct):
    """Zero out the smallest weights by magnitude."""
    pruned = {}
    for k, v in weights.items():
        if v.ndim < 2:
            pruned[k] = v.copy()
            continue
        flat = np.abs(v.flatten())
        threshold = np.percentile(flat[flat > 0], sparsity_pct) if np.any(flat > 0) else 0
        mask = np.abs(v) >= threshold
        pruned[k] = v * mask
    return pruned


def structured_2_4_prune(weights):
    """Enforce 2:4 structured sparsity: for every 4 consecutive values, zero the 2 smallest."""
    pruned = {}
    for k, v in weights.items():
        if v.ndim < 2:
            pruned[k] = v.copy()
            continue
        w = v.copy()
        flat = w.reshape(-1)
        # Process in groups of 4
        n = len(flat)
        padded_n = ((n + 3) // 4) * 4
        padded = np.zeros(padded_n, dtype=flat.dtype)
        padded[:n] = flat
        
        for g in range(0, padded_n, 4):
            group = padded[g:g+4]
            # Find 2 smallest by magnitude, zero them
            indices = np.argsort(np.abs(group))
            group[indices[0]] = 0
            group[indices[1]] = 0
            padded[g:g+4] = group
        
        pruned[k] = padded[:n].reshape(v.shape)
    return pruned


def quantize_int4(weights):
    """Quantize weights to INT4 (4-bit signed: -8 to +7)."""
    quantized = {}
    for k, v in weights.items():
        if v.ndim < 2:
            quantized[k] = v.copy()
            continue
        # Scale to INT4 range
        scale = np.max(np.abs(v)) / 7.0 if np.max(np.abs(v)) > 0 else 1.0
        q = np.round(v / scale).clip(-8, 7).astype(np.int8)
        # Store both quantized values and scale for dequantization
        quantized[k] = (q * scale).astype(np.float32)
    return quantized


def quantize_q88(x):
    """Quantize to Q8.8 fixed-point."""
    return np.round(x * 256) / 256


# ============================================================================
# Hardware Performance Model
# ============================================================================

class HardwareModel:
    """Models hardware performance characteristics."""
    
    def __init__(self, pipeline=False, wide_mem=False, int4=False):
        self.pipeline = pipeline
        self.wide_mem = wide_mem
        self.int4 = int4
    
    def cycles_per_matmul(self, M, K, N, zero_ratio):
        """Estimate cycles for an MxK @ KxN matmul.
        
        Models three hardware subsystems:
        - Compute: actual multiplications needed (reduced by zero-skip + INT4 + pipeline)
        - Memory: weight fetches needed (reduced by sparse storage + wide port)
        - Effective cycles = max(compute, memory) since they overlap
        """
        total_mults = M * K * N
        skipped = int(total_mults * zero_ratio)
        actual_mults = total_mults - skipped
        
        # === COMPUTE CYCLES ===
        compute_cycles = actual_mults
        
        # INT4 parallel: 4 multiplies per cycle
        if self.int4:
            compute_cycles = compute_cycles // INT4_PARALLEL
        
        # Pipeline: without pipeline, 5 cycles per op
        if not self.pipeline:
            compute_cycles *= PIPELINE_DEPTH
        
        # === MEMORY CYCLES ===
        # Sparse storage: zero weights are stored in compressed format
        # Only non-zero weights need to be fetched from memory
        # Weight elements to fetch = K * N * (1 - weight_zero_fraction)
        # We approximate weight_zero_fraction from overall zero_ratio
        # (conservative: only half of zero_ratio comes from weights)
        weight_zero_frac = min(zero_ratio * 0.8, 0.95)  # Cap at 95%
        weights_to_fetch = int(K * N * (1 - weight_zero_frac))
        
        if self.wide_mem:
            mem_cycles = max(weights_to_fetch // MEMORY_WIDTH, 1)
        else:
            mem_cycles = weights_to_fetch
        
        # Without pipeline, memory also takes 5x (FSM overhead per fetch)
        if not self.pipeline:
            mem_cycles *= PIPELINE_DEPTH
        
        # === EFFECTIVE = max(compute, memory) ===
        effective_cycles = max(compute_cycles, mem_cycles)
        
        return total_mults, skipped, effective_cycles


# ============================================================================
# Benchmark Engine
# ============================================================================

class BenchmarkEngine:
    """Runs inference with different optimization configurations."""
    
    def __init__(self, weights, hw_model):
        self.w = weights
        self.hw = hw_model
        self.stats = {
            'total_mults': 0,
            'skipped_mults': 0,
            'effective_cycles': 0,
            'relu_zeros': 0,
            'relu_total': 0,
        }
    
    def _count_zeros(self, a, b):
        """Count zero element ratio in matmul operands."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_zeros = np.sum(a_flat == 0) / max(len(a_flat), 1)
        b_zeros = np.sum(b_flat == 0) / max(len(b_flat), 1)
        # Probability either operand is zero
        combined = 1 - (1 - a_zeros) * (1 - b_zeros)
        return float(combined)
    
    def _mm(self, a, b, name=""):
        """Matrix multiply with hardware performance tracking."""
        zero_ratio = self._count_zeros(a, b)
        
        if a.ndim == 1:
            M, K = 1, a.shape[0]
        else:
            M, K = a.shape
        
        if b.ndim == 1:
            N = 1
        else:
            N = b.shape[1]
        
        total, skipped, cycles = self.hw.cycles_per_matmul(M, K, N, zero_ratio)
        self.stats['total_mults'] += total
        self.stats['skipped_mults'] += skipped
        self.stats['effective_cycles'] += cycles
        
        return a @ b
    
    def run_one_token(self, token_id, position):
        """Run one token through the full OPT-125M forward pass."""
        # Embedding
        x = self.w['model.decoder.embed_tokens.weight'][token_id] + \
            self.w['model.decoder.embed_positions.weight'][position + 2]
        
        for li in range(NUM_LAYERS):
            p = f'model.decoder.layers.{li}'
            
            # --- Self-Attention ---
            # LayerNorm
            ln_w = self.w[f'{p}.self_attn_layer_norm.weight']
            ln_b = self.w[f'{p}.self_attn_layer_norm.bias']
            h = (x - x.mean()) / (x.std() + 1e-5) * ln_w + ln_b
            
            # Q, K, V projections (3 matmuls)
            q = self._mm(h, self.w[f'{p}.self_attn.q_proj.weight'].T, "q_proj") + self.w[f'{p}.self_attn.q_proj.bias']
            k = self._mm(h, self.w[f'{p}.self_attn.k_proj.weight'].T, "k_proj") + self.w[f'{p}.self_attn.k_proj.bias']
            v = self._mm(h, self.w[f'{p}.self_attn.v_proj.weight'].T, "v_proj") + self.w[f'{p}.self_attn.v_proj.bias']
            
            # Simplified attention (single token, no KV cache for benchmark)
            q_heads = q.reshape(NUM_HEADS, HEAD_DIM)
            k_heads = k.reshape(NUM_HEADS, HEAD_DIM)
            v_heads = v.reshape(NUM_HEADS, HEAD_DIM)
            
            attn = np.sum(q_heads * k_heads, axis=1) / np.sqrt(HEAD_DIM)
            attn_weights = np.exp(attn - attn.max()) / np.sum(np.exp(attn - attn.max()))
            attn_out = np.sum(attn_weights[:, None] * v_heads, axis=0).flatten()
            
            # Pad back to full dim if needed
            if attn_out.shape[0] < DIM:
                attn_out = np.tile(attn_out, DIM // attn_out.shape[0] + 1)[:DIM]
            
            # Output projection (1 matmul)
            attn_out = self._mm(attn_out, self.w[f'{p}.self_attn.out_proj.weight'].T, "out_proj") + \
                       self.w[f'{p}.self_attn.out_proj.bias']
            
            x = x + attn_out
            
            # --- FFN ---
            ln_w2 = self.w[f'{p}.final_layer_norm.weight']
            ln_b2 = self.w[f'{p}.final_layer_norm.bias']
            h2 = (x - x.mean()) / (x.std() + 1e-5) * ln_w2 + ln_b2
            
            # fc1 → ReLU → fc2 (2 matmuls)
            fc1_out = self._mm(h2, self.w[f'{p}.fc1.weight'].T, "fc1") + self.w[f'{p}.fc1.bias']
            
            # ReLU (native OPT activation)
            relu_out = np.maximum(0, fc1_out)
            relu_zeros = int(np.sum(relu_out == 0))
            self.stats['relu_zeros'] += relu_zeros
            self.stats['relu_total'] += relu_out.size
            
            fc2_out = self._mm(relu_out, self.w[f'{p}.fc2.weight'].T, "fc2") + self.w[f'{p}.fc2.bias']
            
            x = x + fc2_out
        
        return x


# ============================================================================
# Main Benchmark
# ============================================================================

def run_benchmark(weights_orig, config_name, hw_model, sparsity_fn=None, verbose=False):
    """Run a single benchmark configuration."""
    # Apply sparsity technique if specified
    if sparsity_fn:
        weights = sparsity_fn(weights_orig)
    else:
        weights = {k: v.copy() for k, v in weights_orig.items()}
    
    # Quantize to Q8.8
    weights = {k: quantize_q88(v) for k, v in weights.items()}
    
    # Count weight sparsity
    total_weights = 0
    zero_weights = 0
    for k, v in weights.items():
        if v.ndim >= 2:
            total_weights += v.size
            zero_weights += np.sum(v == 0)
    weight_sparsity = zero_weights / max(total_weights, 1)
    
    engine = BenchmarkEngine(weights, hw_model)
    
    # Run 3 tokens through the model
    tokens = [2, 464, 499]  # </s>, 'The', 'future'
    for i, tid in enumerate(tokens):
        engine.run_one_token(tid, i)
    
    s = engine.stats
    skip_pct = s['skipped_mults'] / max(s['total_mults'], 1) * 100
    relu_pct = s['relu_zeros'] / max(s['relu_total'], 1) * 100
    
    # Throughput calculation
    baseline_cycles = s['total_mults'] * PIPELINE_DEPTH  # No optimizations
    effective_cycles = s['effective_cycles']
    speedup = baseline_cycles / max(effective_cycles, 1)
    latency_ms = effective_cycles / (CLOCK_MHZ * 1e3)
    
    result = {
        'config': config_name,
        'weight_sparsity': weight_sparsity * 100,
        'relu_sparsity': relu_pct,
        'zero_skip': skip_pct,
        'total_mults': s['total_mults'],
        'skipped': s['skipped_mults'],
        'effective_cycles': effective_cycles,
        'baseline_cycles': baseline_cycles,
        'speedup': speedup,
        'latency_ms': latency_ms,
    }
    
    if verbose:
        print(f"\n{'='*65}")
        print(f"  {config_name}")
        print(f"{'='*65}")
        print(f"  Weight sparsity:   {result['weight_sparsity']:6.1f}%")
        print(f"  ReLU sparsity:     {result['relu_sparsity']:6.1f}%")
        print(f"  Zero-skip rate:    {result['zero_skip']:6.1f}%")
        print(f"  Total mults:       {result['total_mults']:>14,}")
        print(f"  Skipped mults:     {result['skipped']:>14,}")
        print(f"  Effective cycles:  {result['effective_cycles']:>14,}")
        print(f"  Baseline cycles:   {result['baseline_cycles']:>14,}")
        print(f"  Speedup:           {result['speedup']:>10.1f}x")
        print(f"  Est. latency:      {result['latency_ms']:>10.2f} ms")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="BitbyBit GPU Throughput Benchmark")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-config results")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    npz_path = os.path.join(root, "weights", "opt125m", "opt125m_weights.npz")
    
    if not os.path.exists(npz_path):
        print(f"ERROR: Weights not found at {npz_path}")
        print("Run: python scripts/chat_opt.py  (it will auto-download)")
        sys.exit(1)
    
    print("Loading OPT-125M weights...")
    raw = np.load(npz_path)
    weights = {k: raw[k].astype(np.float32) for k in raw.files}
    print(f"Loaded {len(weights)} tensors\n")
    
    # ========================================================================
    # Define all benchmark configurations
    # ========================================================================
    
    configs = []
    
    # --- Baseline ---
    configs.append(("① Baseline (Q8.8, no optimizations)",
                     HardwareModel(pipeline=False, wide_mem=False, int4=False),
                     None))
    
    # --- Individual optimizations ---
    configs.append(("② + Deep Pipeline (5-stage, 1 op/cycle)",
                     HardwareModel(pipeline=True, wide_mem=False, int4=False),
                     None))
    
    configs.append(("③ + Memory BW (4-wide read port)",
                     HardwareModel(pipeline=False, wide_mem=True, int4=False),
                     None))
    
    configs.append(("④ + INT4 Parallel (4x multiply/cycle)",
                     HardwareModel(pipeline=False, wide_mem=False, int4=True),
                     None))
    
    # --- Sparsity techniques (no HW optimizations, to isolate effect) ---
    configs.append(("⑤ + Weight Pruning 30%",
                     HardwareModel(pipeline=False, wide_mem=False, int4=False),
                     lambda w: magnitude_prune(w, 30)))
    
    configs.append(("⑥ + Weight Pruning 50%",
                     HardwareModel(pipeline=False, wide_mem=False, int4=False),
                     lambda w: magnitude_prune(w, 50)))
    
    configs.append(("⑦ + Weight Pruning 70%",
                     HardwareModel(pipeline=False, wide_mem=False, int4=False),
                     lambda w: magnitude_prune(w, 70)))
    
    configs.append(("⑧ + 2:4 Structured Sparsity",
                     HardwareModel(pipeline=False, wide_mem=False, int4=False),
                     structured_2_4_prune))
    
    # --- Combined optimizations ---
    configs.append(("⑨ Pipeline + MemBW + INT4",
                     HardwareModel(pipeline=True, wide_mem=True, int4=True),
                     None))
    
    configs.append(("⑩ ALL + Weight Pruning 50%",
                     HardwareModel(pipeline=True, wide_mem=True, int4=True),
                     lambda w: magnitude_prune(w, 50)))
    
    configs.append(("⑪ ALL + 2:4 Structured Sparsity",
                     HardwareModel(pipeline=True, wide_mem=True, int4=True),
                     structured_2_4_prune))
    
    configs.append(("⑫ ALL + Weight Pruning 70%",
                     HardwareModel(pipeline=True, wide_mem=True, int4=True),
                     lambda w: magnitude_prune(w, 70)))
    
    # ========================================================================
    # Run all benchmarks
    # ========================================================================
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          BitbyBit GPU — Throughput Optimization Benchmark       ║")
    print("║          OPT-125M · 12 layers · 768-dim · Q8.8                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    results = []
    for name, hw, sparsity_fn in configs:
        t0 = time.time()
        r = run_benchmark(weights, name, hw, sparsity_fn, verbose=args.verbose)
        r['wall_time'] = time.time() - t0
        results.append(r)
        print(f"  ✓ {name} ... {r['speedup']:.1f}x ({r['wall_time']:.1f}s)")
    
    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "═" * 100)
    print(f"{'Configuration':<45} {'Wt Sparse':>9} {'ReLU':>6} {'Skip':>6} {'Speedup':>8} {'Latency':>10}")
    print("─" * 100)
    
    baseline_cycles = results[0]['baseline_cycles']
    for r in results:
        print(f"  {r['config']:<43} {r['weight_sparsity']:>7.1f}% {r['relu_sparsity']:>5.1f}% "
              f"{r['zero_skip']:>5.1f}% {r['speedup']:>7.1f}x {r['latency_ms']:>8.2f}ms")
    
    print("═" * 100)
    
    # ========================================================================
    # Sparsity Comparison: Weight Pruning vs 2:4
    # ========================================================================
    pruning_results = [r for r in results if 'Pruning' in r['config'] and 'ALL' not in r['config']]
    s24_result = [r for r in results if '2:4' in r['config'] and 'ALL' not in r['config']]
    
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║           Sparsity Comparison: Weight Pruning vs 2:4            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    all_sparse = pruning_results + s24_result
    print(f"  {'Technique':<30} {'Wt Zeros':>10} {'Overall Skip':>13} {'Speedup':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*13} {'─'*8}")
    for r in all_sparse:
        short_name = r['config'].split('+')[1].strip() if '+' in r['config'] else r['config']
        print(f"  {short_name:<30} {r['weight_sparsity']:>8.1f}% {r['zero_skip']:>11.1f}% {r['speedup']:>7.1f}x")
    
    # Determine winner
    if all_sparse:
        best = max(all_sparse, key=lambda r: r['speedup'])
        short = best['config'].split('+')[1].strip() if '+' in best['config'] else best['config']
        print(f"\n  ★ Best sparsity technique: {short}")
        print(f"    → {best['weight_sparsity']:.1f}% weight zeros, {best['zero_skip']:.1f}% overall skip, {best['speedup']:.1f}x speedup")
    
    # ========================================================================
    # Final combined results
    # ========================================================================
    combined = [r for r in results if 'ALL' in r['config']]
    if combined:
        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║              Combined Optimization Results                      ║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()
        
        best_combined = max(combined, key=lambda r: r['speedup'])
        for r in combined:
            marker = " ★" if r == best_combined else "  "
            short = r['config'].replace('ALL + ', '').strip()
            print(f"  {marker} Pipeline+MemBW+INT4 + {short}")
            print(f"       Skip: {r['zero_skip']:.1f}% | Speedup: {r['speedup']:.1f}x | Latency: {r['latency_ms']:.2f}ms")
        
        print(f"\n  ★★ BEST OVERALL: {best_combined['config']}")
        print(f"     Speedup: {best_combined['speedup']:.1f}x over baseline")
        print(f"     Zero-skip: {best_combined['zero_skip']:.1f}%")
        print(f"     Latency: {best_combined['latency_ms']:.2f}ms @ {CLOCK_MHZ}MHz")


if __name__ == "__main__":
    main()
