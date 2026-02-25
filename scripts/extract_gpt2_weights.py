#!/usr/bin/env python3
"""
extract_gpt2_weights.py — Extract real GPT-2 weights and quantize to Q8.8

Downloads GPT-2 weights from HuggingFace, extracts the first EMBED_DIM
dimensions, quantizes to Q8.8, and saves as hex files + numpy arrays
for both the Cocotb testbench and the Verilog $readmemh flow.

Usage:
    python scripts/extract_gpt2_weights.py
    python scripts/extract_gpt2_weights.py --embed-dim 8
"""

import numpy as np
import os
import json
import argparse
import struct
from pathlib import Path

# ============================================================================
# Q8.8 Fixed-Point Conversion
# ============================================================================

def float_to_q88(val):
    """Convert float to Q8.8 signed 16-bit."""
    q = int(round(val * 256))
    return max(-32768, min(32767, q))

def q88_to_float(q):
    """Convert Q8.8 to float."""
    if q >= 32768:
        q -= 65536
    return q / 256.0

def q88_hex(val):
    """Convert Q8.8 to 4-digit hex string."""
    return f"{val & 0xFFFF:04x}"

# ============================================================================
# Weight Downloading (numpy-only, no PyTorch needed!)
# ============================================================================

def download_gpt2_weights(cache_dir):
    """
    Download GPT-2 small (124M) weights from HuggingFace using safetensors.
    Falls back to generating synthetic weights if no internet.
    """
    weights_file = os.path.join(cache_dir, "gpt2_weights.npz")
    
    if os.path.exists(weights_file):
        print(f"  Loading cached weights from {weights_file}")
        return dict(np.load(weights_file, allow_pickle=True))
    
    print("  Downloading GPT-2 weights from HuggingFace (safetensors)...")
    
    # Strategy 1: Direct safetensors download (no PyTorch needed!)
    try:
        import urllib.request
        
        url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
        safetensors_path = os.path.join(cache_dir, "model.safetensors")
        
        if not os.path.exists(safetensors_path):
            print(f"  Downloading ~500MB from HuggingFace...")
            print(f"  URL: {url}")
            urllib.request.urlretrieve(url, safetensors_path,
                                       reporthook=_download_progress)
            print()
        else:
            print(f"  Using cached safetensors: {safetensors_path}")
        
        # Parse safetensors format (simple binary format)
        weights = parse_safetensors(safetensors_path)
        np.savez(weights_file, **weights)
        print(f"  Saved {len(weights)} weight tensors to cache")
        return weights
        
    except Exception as e:
        print(f"  Direct download failed: {e}")
    
    # Strategy 2: Use transformers + torch (if available)
    try:
        from transformers import GPT2Model
        import torch
        
        print("  Trying transformers library...")
        model = GPT2Model.from_pretrained("gpt2")
        state_dict = model.state_dict()
        
        weights = {}
        for key, tensor in state_dict.items():
            weights[key] = tensor.cpu().numpy()
        
        np.savez(weights_file, **weights)
        print(f"  Saved weights to {weights_file}")
        return weights
        
    except ImportError:
        pass
    
    # Strategy 3: Generate synthetic weights
    print("  No internet/dependencies available.")
    print("  Generating synthetic GPT-2-like weights instead...")
    return generate_synthetic_weights()

def _download_progress(count, block_size, total_size):
    """Progress callback for urllib download."""
    percent = min(100, int(count * block_size * 100 / total_size))
    mb_done = count * block_size / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\r  Progress: {percent:3d}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

def parse_safetensors(filepath):
    """Parse a safetensors file into numpy arrays."""
    weights = {}
    
    with open(filepath, 'rb') as f:
        # Header length (8 bytes, little-endian uint64)
        header_len = struct.unpack('<Q', f.read(8))[0]
        
        # Header JSON
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        
        # Data starts after header
        data_start = 8 + header_len
        
        dtype_map = {
            'F32': (np.float32, 4),
            'F16': (np.float16, 2),
            'BF16': (np.float16, 2),  # We'll treat BF16 as FP16 approx
        }
        
        for name, info in header.items():
            if name == '__metadata__':
                continue
            
            dtype_str = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']
            
            if dtype_str not in dtype_map:
                continue
            
            np_dtype, _ = dtype_map[dtype_str]
            
            f.seek(data_start + offsets[0])
            num_bytes = offsets[1] - offsets[0]
            raw_data = f.read(num_bytes)
            
            arr = np.frombuffer(raw_data, dtype=np_dtype).reshape(shape)
            weights[name] = arr.astype(np.float32)
    
    return weights

def generate_synthetic_weights():
    """Generate synthetic GPT-2-like weights (when download unavailable)."""
    print("  Generating synthetic GPT-2-like weights (768-dim, 12 layers)...")
    
    np.random.seed(42)  # Reproducible
    ED = 768  # GPT-2 small embedding dimension
    FFN = 3072  # 4 * ED
    NL = 12  # 12 layers
    VS = 50257  # GPT-2 vocabulary size
    MSL = 1024  # Max sequence length
    
    weights = {}
    
    # Embeddings
    weights['wte.weight'] = np.random.randn(VS, ED).astype(np.float32) * 0.02
    weights['wpe.weight'] = np.random.randn(MSL, ED).astype(np.float32) * 0.02
    
    for layer in range(NL):
        prefix = f'h.{layer}'
        
        # LayerNorm 1
        weights[f'{prefix}.ln_1.weight'] = np.ones(ED, dtype=np.float32)
        weights[f'{prefix}.ln_1.bias'] = np.zeros(ED, dtype=np.float32)
        
        # Attention (QKV combined + output projection)
        weights[f'{prefix}.attn.c_attn.weight'] = np.random.randn(ED, 3*ED).astype(np.float32) * 0.02
        weights[f'{prefix}.attn.c_attn.bias'] = np.zeros(3*ED, dtype=np.float32)
        weights[f'{prefix}.attn.c_proj.weight'] = np.random.randn(ED, ED).astype(np.float32) * 0.02
        weights[f'{prefix}.attn.c_proj.bias'] = np.zeros(ED, dtype=np.float32)
        
        # LayerNorm 2
        weights[f'{prefix}.ln_2.weight'] = np.ones(ED, dtype=np.float32)
        weights[f'{prefix}.ln_2.bias'] = np.zeros(ED, dtype=np.float32)
        
        # FFN
        weights[f'{prefix}.mlp.c_fc.weight'] = np.random.randn(ED, FFN).astype(np.float32) * 0.02
        weights[f'{prefix}.mlp.c_fc.bias'] = np.zeros(FFN, dtype=np.float32)
        weights[f'{prefix}.mlp.c_proj.weight'] = np.random.randn(FFN, ED).astype(np.float32) * 0.02
        weights[f'{prefix}.mlp.c_proj.bias'] = np.zeros(ED, dtype=np.float32)
    
    # Final layer norm
    weights['ln_f.weight'] = np.ones(ED, dtype=np.float32)
    weights['ln_f.bias'] = np.zeros(ED, dtype=np.float32)
    
    return weights

# ============================================================================
# Weight Extraction & Quantization for BitbyBit GPU
# ============================================================================

def extract_for_bitbybit(weights, embed_dim=4, ffn_dim=8, vocab_size=16,
                         max_seq_len=8, num_layers=2, output_dir="weights"):
    """
    Extract a slice of real GPT-2 weights, fitted to the BitbyBit GPU parameters.
    Quantize to Q8.8 and save in formats usable by both Python and Verilog.
    """
    ED = embed_dim
    FD = ffn_dim
    VS = vocab_size
    MSL = max_seq_len
    NL = num_layers
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = {}
    
    print(f"\n  Extracting weights for BitbyBit GPU:")
    print(f"    EMBED_DIM={ED}, FFN_DIM={FD}, VOCAB_SIZE={VS}, LAYERS={NL}")
    print()
    
    # ---- Token Embeddings ----
    if 'wte.weight' in weights:
        full_emb = weights['wte.weight']  # [50257, 768]
        # Take first VS tokens, first ED dims
        token_emb = full_emb[:VS, :ED].astype(np.float64)
    else:
        token_emb = np.random.randn(VS, ED).astype(np.float64) * 0.02
    
    token_emb_q = np.vectorize(float_to_q88)(token_emb).astype(np.int32)
    extracted['token_emb'] = token_emb_q
    
    print(f"  Token Embeddings: [{VS}x{ED}] — range [{token_emb.min():.3f}, {token_emb.max():.3f}]")
    
    # ---- Position Embeddings ----
    if 'wpe.weight' in weights:
        full_pos = weights['wpe.weight']  # [1024, 768]
        pos_emb = full_pos[:MSL, :ED].astype(np.float64)
    else:
        pos_emb = np.random.randn(MSL, ED).astype(np.float64) * 0.02
    
    pos_emb_q = np.vectorize(float_to_q88)(pos_emb).astype(np.int32)
    extracted['pos_emb'] = pos_emb_q
    
    print(f"  Position Embeddings: [{MSL}x{ED}] — range [{pos_emb.min():.3f}, {pos_emb.max():.3f}]")
    
    # ---- Layer 0 weights (used as shared weights in our design) ----
    prefix = 'h.0'
    
    # LayerNorm 1
    ln1_gamma = extract_ln(weights, f'{prefix}.ln_1.weight', ED)
    ln1_beta = extract_ln(weights, f'{prefix}.ln_1.bias', ED)
    extracted['ln1_gamma'] = np.vectorize(float_to_q88)(ln1_gamma)
    extracted['ln1_beta'] = np.vectorize(float_to_q88)(ln1_beta)
    
    # Attention weights: GPT-2 stores Q,K,V combined in c_attn [768, 2304]
    if f'{prefix}.attn.c_attn.weight' in weights:
        c_attn = weights[f'{prefix}.attn.c_attn.weight']  # [768, 2304]
        wq = c_attn[:ED, :ED].astype(np.float64)
        wk = c_attn[:ED, ED:2*ED].astype(np.float64) if c_attn.shape[1] >= 2*ED else c_attn[:ED, :ED].astype(np.float64)
        wv = c_attn[:ED, 2*ED:3*ED].astype(np.float64) if c_attn.shape[1] >= 3*ED else c_attn[:ED, :ED].astype(np.float64)
    else:
        wq = np.eye(ED, dtype=np.float64)
        wk = np.eye(ED, dtype=np.float64)
        wv = np.eye(ED, dtype=np.float64)
    
    extracted['wq'] = np.vectorize(float_to_q88)(wq)
    extracted['wk'] = np.vectorize(float_to_q88)(wk)
    extracted['wv'] = np.vectorize(float_to_q88)(wv)
    
    print(f"  Wq: [{ED}x{ED}] — range [{wq.min():.3f}, {wq.max():.3f}]")
    print(f"  Wk: [{ED}x{ED}] — range [{wk.min():.3f}, {wk.max():.3f}]")
    print(f"  Wv: [{ED}x{ED}] — range [{wv.min():.3f}, {wv.max():.3f}]")
    
    # Output projection
    if f'{prefix}.attn.c_proj.weight' in weights:
        wo = weights[f'{prefix}.attn.c_proj.weight'][:ED, :ED].astype(np.float64)
    else:
        wo = np.eye(ED, dtype=np.float64)
    extracted['wo'] = np.vectorize(float_to_q88)(wo)
    
    # LayerNorm 2
    ln2_gamma = extract_ln(weights, f'{prefix}.ln_2.weight', ED)
    ln2_beta = extract_ln(weights, f'{prefix}.ln_2.bias', ED)
    extracted['ln2_gamma'] = np.vectorize(float_to_q88)(ln2_gamma)
    extracted['ln2_beta'] = np.vectorize(float_to_q88)(ln2_beta)
    
    # FFN weights
    if f'{prefix}.mlp.c_fc.weight' in weights:
        ffn_w1 = weights[f'{prefix}.mlp.c_fc.weight'][:ED, :FD].astype(np.float64)
        ffn_b1 = weights[f'{prefix}.mlp.c_fc.bias'][:FD].astype(np.float64)
    else:
        ffn_w1 = np.random.randn(ED, FD).astype(np.float64) * 0.02
        ffn_b1 = np.zeros(FD, dtype=np.float64)
    
    if f'{prefix}.mlp.c_proj.weight' in weights:
        ffn_w2 = weights[f'{prefix}.mlp.c_proj.weight'][:FD, :ED].astype(np.float64)
        ffn_b2 = weights[f'{prefix}.mlp.c_proj.bias'][:ED].astype(np.float64)
    else:
        ffn_w2 = np.random.randn(FD, ED).astype(np.float64) * 0.02
        ffn_b2 = np.zeros(ED, dtype=np.float64)
    
    extracted['ffn_w1'] = np.vectorize(float_to_q88)(ffn_w1)
    extracted['ffn_b1'] = np.vectorize(float_to_q88)(ffn_b1)
    extracted['ffn_w2'] = np.vectorize(float_to_q88)(ffn_w2)
    extracted['ffn_b2'] = np.vectorize(float_to_q88)(ffn_b2)
    
    print(f"  FFN W1: [{ED}x{FD}] — range [{ffn_w1.min():.3f}, {ffn_w1.max():.3f}]")
    print(f"  FFN W2: [{FD}x{ED}] — range [{ffn_w2.min():.3f}, {ffn_w2.max():.3f}]")
    
    # Final LayerNorm
    ln_f_gamma = extract_ln(weights, 'ln_f.weight', ED)
    ln_f_beta = extract_ln(weights, 'ln_f.bias', ED)
    extracted['ln_final_gamma'] = np.vectorize(float_to_q88)(ln_f_gamma)
    extracted['ln_final_beta'] = np.vectorize(float_to_q88)(ln_f_beta)
    
    # ---- Save everything ----
    # NPZ for Python cosimulation
    npz_file = os.path.join(output_dir, "gpt2_q88_weights.npz")
    np.savez(npz_file, **{k: v.astype(np.int32) for k, v in extracted.items()})
    print(f"\n  Saved Q8.8 weights to {npz_file}")
    
    # Hex files for Verilog $readmemh
    save_hex_files(extracted, output_dir)
    
    # Summary JSON
    summary = {
        'embed_dim': ED, 'ffn_dim': FD, 'vocab_size': VS,
        'max_seq_len': MSL, 'num_layers': NL,
        'source': 'gpt2-small (first dims extracted)',
        'quantization': 'Q8.8 (signed 16-bit fixed-point)',
        'weight_count': sum(v.size for v in extracted.values()),
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Total Q8.8 parameters: {summary['weight_count']}")
    
    return extracted

def extract_ln(weights, key, dim):
    """Extract layer norm parameters."""
    if key in weights:
        return weights[key][:dim].astype(np.float64)
    if 'weight' in key:
        return np.ones(dim, dtype=np.float64)
    return np.zeros(dim, dtype=np.float64)

def save_hex_files(extracted, output_dir):
    """Save individual weight arrays as hex files for Verilog $readmemh."""
    hex_dir = os.path.join(output_dir, "hex")
    os.makedirs(hex_dir, exist_ok=True)
    
    for name, arr in extracted.items():
        filepath = os.path.join(hex_dir, f"{name}.hex")
        flat = arr.flatten()
        with open(filepath, 'w') as f:
            for val in flat:
                f.write(q88_hex(int(val)) + "\n")

# ============================================================================
# Float32 Reference Inference (for comparison)
# ============================================================================

def run_float32_reference(weights, token_id, position, embed_dim=4, ffn_dim=8):
    """Run a simplified GPT-2 forward pass in float32 for comparison."""
    ED = embed_dim
    FD = ffn_dim
    
    # Embedding
    if 'wte.weight' in weights:
        tok_emb = weights['wte.weight'][token_id, :ED].astype(np.float64)
        pos_emb = weights['wpe.weight'][position, :ED].astype(np.float64)
    else:
        tok_emb = np.zeros(ED)
        pos_emb = np.zeros(ED)
    
    x = tok_emb + pos_emb
    
    prefix = 'h.0'
    
    # 2 passes through the same layer (matching Verilog NUM_LAYERS=2)
    for _ in range(2):
        residual = x.copy()
        
        # LayerNorm 1
        gamma = extract_ln(weights, f'{prefix}.ln_1.weight', ED)
        beta = extract_ln(weights, f'{prefix}.ln_1.bias', ED)
        x = layer_norm_f32(x, gamma, beta)
        
        # Attention (simplified single-token)
        if f'{prefix}.attn.c_attn.weight' in weights:
            c_attn = weights[f'{prefix}.attn.c_attn.weight']
            wq = c_attn[:ED, :ED].astype(np.float64)
            wk_size = min(2*ED, c_attn.shape[1])
            wk = c_attn[:ED, ED:wk_size].astype(np.float64) if wk_size > ED else wq
            wv_start = min(2*ED, c_attn.shape[1])
            wv_end = min(3*ED, c_attn.shape[1])
            wv = c_attn[:ED, wv_start:wv_end].astype(np.float64) if wv_end > wv_start else wq
            wo = weights[f'{prefix}.attn.c_proj.weight'][:ED, :ED].astype(np.float64)
        else:
            wq = wk = wv = wo = np.eye(ED)
        
        q = wq.T @ x
        v = wv.T @ x
        out = wo.T @ v  # Single token: output = Wo @ V
        x = residual + out
        
        residual = x.copy()
        
        # LayerNorm 2
        gamma2 = extract_ln(weights, f'{prefix}.ln_2.weight', ED)
        beta2 = extract_ln(weights, f'{prefix}.ln_2.bias', ED)
        x = layer_norm_f32(x, gamma2, beta2)
        
        # FFN
        if f'{prefix}.mlp.c_fc.weight' in weights:
            w1 = weights[f'{prefix}.mlp.c_fc.weight'][:ED, :FD].astype(np.float64)
            b1 = weights[f'{prefix}.mlp.c_fc.bias'][:FD].astype(np.float64)
            w2 = weights[f'{prefix}.mlp.c_proj.weight'][:FD, :ED].astype(np.float64)
            b2 = weights[f'{prefix}.mlp.c_proj.bias'][:ED].astype(np.float64)
        else:
            w1 = np.zeros((ED, FD))
            b1 = np.zeros(FD)
            w2 = np.zeros((FD, ED))
            b2 = np.zeros(ED)
        
        h = w1.T @ x + b1
        h = gelu_f32(h)
        ffn_out = w2.T @ h + b2
        x = residual + ffn_out
    
    # Final LayerNorm
    gamma_f = extract_ln(weights, 'ln_f.weight', ED)
    beta_f = extract_ln(weights, 'ln_f.bias', ED)
    x = layer_norm_f32(x, gamma_f, beta_f)
    
    return x

def layer_norm_f32(x, gamma, beta, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def gelu_f32(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GPT-2 weights for BitbyBit GPU")
    parser.add_argument("--embed-dim", type=int, default=4, help="Embedding dimension")
    parser.add_argument("--ffn-dim", type=int, default=8, help="FFN hidden dimension")
    parser.add_argument("--vocab-size", type=int, default=16, help="Vocabulary size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--output-dir", type=str, default="weights/gpt2_real", help="Output directory")
    args = parser.parse_args()
    
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print()
    print("=" * 60)
    print("  GPT-2 Weight Extractor for BitbyBit GPU")
    print("=" * 60)
    
    # Download/load weights
    raw_weights = download_gpt2_weights(cache_dir)
    
    # Extract and quantize
    output_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    extracted = extract_for_bitbybit(
        raw_weights,
        embed_dim=args.embed_dim,
        ffn_dim=args.ffn_dim,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        output_dir=output_dir,
    )
    
    # Run float32 reference for comparison
    print()
    print("=" * 60)
    print("  Float32 Reference Inference (for comparison)")
    print("=" * 60)
    
    for token_id in [0, 3, 5]:
        logits = run_float32_reference(raw_weights, token_id, 0,
                                        args.embed_dim, args.ffn_dim)
        pred = int(np.argmax(logits))
        print(f"  Token {token_id:2d} → Predicted dim: {pred} | logits: {logits}")
    
    print()
    print("  Weights ready! Use with:")
    print("    - Cocotb testbench:  make -C tb/cocotb")
    print("    - Verilog $readmemh: weights/gpt2_real/hex/*.hex")
    print()
