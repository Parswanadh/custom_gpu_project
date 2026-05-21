"""
Gemma 3 270M — NVIDIA RTX 4070 Benchmark + INT4 Weight Export
=============================================================
1. Downloads Gemma 3 270M from HuggingFace
2. Benchmarks inference on RTX 4070 (tok/sec, latency, memory)
3. Exports INT4 quantized weights as hex files for FPGA Verilog $readmemh
4. Generates per-layer golden vectors for RTL validation

Fair comparison: use output JSON with compare_bitbybit_vs_rtx.py --nvidia-json
after FPGA runs the same model. See docs/PERFORMANCE_VALIDATION.md.
"""

import os
import sys
import json
import time
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"  # Fallback if 270M not available
MODEL_ID_SMALL = "google/gemma-3-270m"  # Preferred (270M base)
EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "gemma3_fpga")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "gemma3_golden")
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim", "gemma3_nvidia_benchmark.json")

PROMPT = "The future of AI hardware is"
MAX_NEW_TOKENS = 50
NUM_WARMUP = 3
NUM_MEASURED = 10

def ensure_dirs():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

def float_to_q88(val):
    """Convert float to Q8.8 signed fixed-point (16-bit)."""
    q = int(round(val * 256))
    q = max(-32768, min(32767, q))
    return q & 0xFFFF

def quantize_to_int4(tensor_np):
    """Symmetric INT4 quantization: scale = max(abs(tensor)) / 7."""
    absmax = np.max(np.abs(tensor_np))
    if absmax == 0:
        scale = 1.0
    else:
        scale = absmax / 7.0
    quantized = np.clip(np.round(tensor_np / scale), -8, 7).astype(np.int8)
    return quantized, scale

def pack_int4_to_hex(int4_array_flat):
    """Pack pairs of INT4 values into bytes. Return list of hex strings."""
    hex_lines = []
    arr = int4_array_flat.flatten()
    # Pad to even length
    if len(arr) % 2 != 0:
        arr = np.append(arr, np.int8(0))
    for i in range(0, len(arr), 2):
        lo = int(arr[i]) & 0x0F
        hi = int(arr[i+1]) & 0x0F
        byte_val = (hi << 4) | lo
        hex_lines.append(f"{byte_val:02x}")
    return hex_lines

def export_tensor_hex(tensor_np, name, export_dir):
    """Export a tensor as INT4 packed hex file + scale file."""
    q, scale = quantize_to_int4(tensor_np)
    hex_lines = pack_int4_to_hex(q)
    hex_path = os.path.join(export_dir, f"{name}.hex")
    with open(hex_path, 'w') as f:
        f.write('\n'.join(hex_lines) + '\n')
    scale_path = os.path.join(export_dir, f"{name}_scale.txt")
    with open(scale_path, 'w') as f:
        f.write(f"{scale:.10e}\n")
    return hex_path, q.shape, scale

def export_tensor_q88_hex(tensor_np, name, export_dir):
    """Export a tensor as Q8.8 hex file (16-bit per value)."""
    flat = tensor_np.flatten()
    hex_lines = []
    for val in flat:
        q = float_to_q88(float(val))
        hex_lines.append(f"{q:04x}")
    hex_path = os.path.join(export_dir, f"{name}_q88.hex")
    with open(hex_path, 'w') as f:
        f.write('\n'.join(hex_lines) + '\n')
    return hex_path

def main():
    ensure_dirs()
    print("=" * 70)
    print("  Gemma 3 — NVIDIA Benchmark + FPGA Weight Export")
    print("=" * 70)

    # ---- Step 1: Import torch and transformers ----
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"FATAL: Missing dependency: {e}")
        print("Run: pip install torch transformers accelerate")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Step 2: Load model ----
    print(f"\nLoading model: {MODEL_ID_SMALL}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_SMALL)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_SMALL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model_id_used = MODEL_ID_SMALL
    except Exception as e:
        print(f"Could not load {MODEL_ID_SMALL}: {e}")
        print(f"Trying fallback: {MODEL_ID}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model_id_used = MODEL_ID
        except Exception as e2:
            print(f"FATAL: Could not load any model: {e2}")
            print("You may need to: huggingface-cli login")
            sys.exit(1)

    model.eval()
    config = model.config
    print(f"Model loaded: {model_id_used}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Heads: {config.num_attention_heads}")
    print(f"  KV Heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  Intermediate: {config.intermediate_size}")
    print(f"  Vocab: {config.vocab_size}")

    # ---- Step 3: Benchmark inference ----
    print(f"\n--- Benchmarking on {device.upper()} ---")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    print(f"Warmup ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    if device == "cuda":
        torch.cuda.synchronize()

    # Measured runs
    print(f"Measured ({NUM_MEASURED} runs)...")
    latencies = []
    for i in range(NUM_MEASURED):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gen_tokens = outputs.shape[1] - input_len
        latencies.append((t1 - t0, gen_tokens))

    # Calculate stats
    total_times = [l[0] for l in latencies]
    total_tokens = [l[1] for l in latencies]
    tok_per_sec = [t / l for l, t in zip(total_times, total_tokens)]
    avg_tps = np.mean(tok_per_sec)
    avg_latency = np.mean(total_times) / np.mean(total_tokens) * 1000  # ms/token

    # Memory usage
    if device == "cuda":
        mem_used = torch.cuda.max_memory_allocated() / 1e6  # MB
        mem_reserved = torch.cuda.max_memory_reserved() / 1e6
    else:
        mem_used = 0
        mem_reserved = 0

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n--- NVIDIA Benchmark Results ---")
    print(f"  Model: {model_id_used}")
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Generated: '{generated_text}'")
    print(f"  Avg tokens/sec: {avg_tps:.2f}")
    print(f"  Avg latency: {avg_latency:.2f} ms/token")
    print(f"  GPU memory used: {mem_used:.1f} MB")
    print(f"  TDP: 110W")
    print(f"  Efficiency: {avg_tps / 110:.2f} tok/sec/watt")

    results = {
        "model": model_id_used,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "prompt": PROMPT,
        "generated_text": generated_text,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_warmup": NUM_WARMUP,
        "num_measured": NUM_MEASURED,
        "avg_tokens_per_sec": float(avg_tps),
        "avg_latency_ms_per_token": float(avg_latency),
        "gpu_memory_used_mb": float(mem_used),
        "gpu_memory_reserved_mb": float(mem_reserved),
        "gpu_tdp_watts": 110,
        "efficiency_tok_sec_watt": float(avg_tps / 110),
        "config": {
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": getattr(config, 'num_key_value_heads', None),
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
        }
    }

    # ---- Step 4: Export weights ----
    print(f"\n--- Exporting INT4 Weights for FPGA ---")
    manifest = {"model": model_id_used, "quantization": "INT4_symmetric", "layers": {}}

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        tensor_np = param.detach().cpu().float().numpy()
        safe_name = name.replace(".", "_")
        hex_path, shape, scale = export_tensor_hex(tensor_np, safe_name, EXPORT_DIR)
        manifest["layers"][name] = {
            "hex_file": os.path.basename(hex_path),
            "original_shape": list(tensor_np.shape),
            "int4_scale": float(scale),
            "num_params": int(np.prod(tensor_np.shape))
        }
        print(f"  Exported: {name} {list(tensor_np.shape)} -> {os.path.basename(hex_path)} (scale={scale:.6f})")

    manifest_path = os.path.join(EXPORT_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    total_params = sum(v["num_params"] for v in manifest["layers"].values())
    total_int4_bytes = total_params // 2
    print(f"  Total params: {total_params:,}")
    print(f"  INT4 size: {total_int4_bytes / 1e6:.1f} MB")

    # ---- Step 5: Generate golden vectors ----
    print(f"\n--- Generating Golden Vectors ---")
    inputs_golden = tokenizer("Hello", return_tensors="pt").to(device)

    # Hook to capture intermediate activations
    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activations[layer_name] = output[0].detach().cpu().float().numpy()
            else:
                activations[layer_name] = output.detach().cpu().float().numpy()
        return hook_fn

    # Register hooks on each decoder layer
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(f"layer_{i}"))
        hooks.append(h)

    # Register hook on embedding
    h_emb = model.model.embed_tokens.register_forward_hook(make_hook("embedding"))
    hooks.append(h_emb)

    # Forward pass
    with torch.no_grad():
        logits = model(**inputs_golden).logits

    # Export golden vectors
    for name, act in activations.items():
        # Save first token position only (for validation)
        vec = act[0, 0, :] if len(act.shape) == 3 else act[0, :]
        export_tensor_q88_hex(vec, f"golden_{name}", GOLDEN_DIR)
        print(f"  Golden: {name} shape={act.shape}")

    # Export final logits (first position)
    logits_np = logits[0, 0, :].detach().cpu().float().numpy()
    export_tensor_q88_hex(logits_np[:1024], "golden_logits_top1024", GOLDEN_DIR)
    print(f"  Golden logits (top 1024 entries)")

    # Remove hooks
    for h in hooks:
        h.remove()

    # ---- Step 6: Save results ----
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_FILE}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model: {model_id_used}")
    print(f"  NVIDIA RTX 4070: {avg_tps:.1f} tok/sec, {avg_latency:.1f} ms/token")
    print(f"  Weights exported: {total_int4_bytes / 1e6:.1f} MB (INT4)")
    print(f"  Golden vectors: {len(activations)} layers")
    print(f"  FPGA 512MB budget: {total_int4_bytes / 512e6 * 100:.1f}% used (weights only)")
    print("=" * 70)

if __name__ == "__main__":
    main()
