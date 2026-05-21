#!/usr/bin/env python3
"""
Compare BitbyBit simulation-measured throughput vs a local GPU baseline.

BitbyBit throughput is DERIVED from cycles/token and clock — never hard-coded.
Optional MEDUSA effective throughput is reported separately (3 draft heads).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Allow import when run from repo root or custom_gpu_project/scripts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from perf_metrics import (  # noqa: E402
    ArchitecturalMultipliers,
    load_sim_metrics,
    throughput_from_cycles,
)

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class SimpleTransformerBlock(nn.Module):
    def __init__(self, embd_dim: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = nn.MultiheadAttention(embd_dim, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim, 4 * embd_dim),
            nn.GELU(),
            nn.Linear(4 * embd_dim, embd_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, n_layer=12, embd_dim=768, n_head=12, vocab_size=50257):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embd_dim)
        self.blocks = nn.Sequential(
            *[SimpleTransformerBlock(embd_dim, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embd_dim)
        self.head = nn.Linear(embd_dim, vocab_size)

    def forward(self, idx):
        x = self.tok_emb(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def benchmark_gpu_baseline(seq_len: int = 128, n_warmup: int = 10, n_measured: int = 50):
    """Software GPT-2-shaped forward on CUDA/CPU — NOT Gemma-3 unless noted."""
    if torch is None:
        raise RuntimeError("PyTorch required: pip install torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(n_layer=12, embd_dim=768, n_head=12).to(device)
    model.eval()
    input_ids = torch.randint(0, 50257, (1, seq_len)).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)

    latencies = []
    with torch.no_grad():
        for _ in range(n_measured):
            start = time.perf_counter()
            _ = model(input_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    avg_latency_s = float(np.mean(latencies))
    throughput = seq_len / avg_latency_s
    latency_ms = avg_latency_s * 1000

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    return {
        "latency_ms": latency_ms,
        "throughput_tps": throughput,
        "latency_ms_per_token": (avg_latency_s / seq_len) * 1000,
        "device": gpu_name,
        "seq_len": seq_len,
        "workload": "MiniGPT 12L/768d forward, batch=1",
    }


def load_nvidia_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_comparison(rtx_data: dict, sim, medusa: bool, rtx_tdp_w: float, bbb_power_w: float):
    imprint_tps = sim.imprint_throughput_tps
    base_tps = sim.base_throughput_tps
    medusa_tps = sim.medusa_effective_throughput_tps

    imprint_latency_ms = (sim.imprint_cycles_per_token / sim.clock_hz) * 1000
    base_latency_ms = (sim.base_cycles_per_token / sim.clock_hz) * 1000

    gpu_tps = rtx_data["throughput_tps"]
    # Honest sim-vs-GPU ratios (different workloads — label clearly)
    ratio_imprint_vs_gpu = imprint_tps / gpu_tps
    ratio_medusa_vs_gpu = medusa_tps / gpu_tps if medusa else None

    arch = ArchitecturalMultipliers()
    projected_tps = imprint_tps * arch.combined_compute_multiplier
    projected_vs_gpu = projected_tps / gpu_tps

    rtx_energy = rtx_tdp_w / gpu_tps
    bbb_energy_imprint = bbb_power_w / imprint_tps

    result = {
        "disclaimer": (
            "BitbyBit numbers are RTL sim @ 100 MHz (mini imprint model). "
            "GPU numbers are PyTorch MiniGPT unless --nvidia-json is set. "
            "Do not quote MEDUSA effective tok/s as sustained single-path throughput."
        ),
        "sim": {
            "clock_mhz": sim.clock_hz / 1e6,
            "source": sim.source,
            "base_cycles_per_token": sim.base_cycles_per_token,
            "imprint_cycles_per_token": sim.imprint_cycles_per_token,
            "base_throughput_tps": base_tps,
            "imprint_throughput_tps": imprint_tps,
            "imprint_speedup_vs_base_x": sim.imprint_speedup_vs_base,
            "imprint_latency_ms": imprint_latency_ms,
            "base_latency_ms": base_latency_ms,
            "derived_check_imprint": throughput_from_cycles(
                sim.clock_hz, sim.imprint_cycles_per_token
            ),
        },
        "gpu_baseline": rtx_data,
        "comparison": {
            "imprint_vs_gpu_throughput_x": ratio_imprint_vs_gpu,
            "medusa_effective_vs_gpu_x": ratio_medusa_vs_gpu,
            "imprint_vs_gpu_note": (
                f"{ratio_imprint_vs_gpu:.2f}x uses sim imprint / GPU measured — workloads differ"
            ),
        },
        "architectural_projection": {
            "int4_mac_density_x": arch.int4_mac_density,
            "zero_skip_fraction": arch.zero_skip_fraction,
            "combined_multiplier_x": arch.combined_compute_multiplier,
            "projected_imprint_tps": projected_tps,
            "projected_vs_gpu_x": projected_vs_gpu,
            "note": "Projection applies documented multipliers to sim imprint — not silicon-validated",
        },
        "energy_estimate": {
            "gpu_tdp_w": rtx_tdp_w,
            "bbb_est_power_w": bbb_power_w,
            "gpu_j_per_token": rtx_energy,
            "bbb_j_per_token_imprint": bbb_energy_imprint,
            "ratio_gpu_to_bbb": rtx_energy / bbb_energy_imprint,
            "note": "Power estimates are placeholders until FPGA power measurement",
        },
    }
    if medusa:
        result["medusa"] = {
            "draft_heads": sim.medusa_draft_heads,
            "effective_throughput_tps": medusa_tps,
            "formula": f"{sim.medusa_draft_heads} * imprint_tps",
        }
    return result


def print_report(comparison: dict):
    sim = comparison["sim"]
    gpu = comparison["gpu_baseline"]
    cmp_ = comparison["comparison"]
    arch = comparison["architectural_projection"]

    print("\n--- BitbyBit (sim @ 100 MHz, derived) ---")
    print(f"  Source: {sim['source']}")
    print(f"  Base:    {sim['base_cycles_per_token']} cy/token -> {sim['base_throughput_tps']:,.0f} tok/s")
    print(
        f"  Imprint: {sim['imprint_cycles_per_token']} cy/token -> "
        f"{sim['imprint_throughput_tps']:,.0f} tok/s"
    )
    print(f"  Imprint speedup vs base: {sim['imprint_speedup_vs_base_x']:.4f}x")
    print(f"  Derivation check: {sim['derived_check_imprint']:,.0f} tok/s")

    if "medusa" in comparison:
        m = comparison["medusa"]
        print(
            f"  MEDUSA effective ({m['draft_heads']} heads): "
            f"{m['effective_throughput_tps']:,.0f} tok/s (speculative, not sustained IPC)"
        )

    print("\n--- GPU baseline (measured) ---")
    print(f"  Device: {gpu['device']}")
    print(f"  Workload: {gpu['workload']}")
    print(f"  Throughput: {gpu['throughput_tps']:,.2f} tok/s")
    print(f"  Latency ({gpu['seq_len']} tokens): {gpu['latency_ms']:.2f} ms")

    print("\n--- Honest comparison ---")
    print(f"  Imprint sim / GPU: {cmp_['imprint_vs_gpu_throughput_x']:.2f}x")
    print(f"  {cmp_['imprint_vs_gpu_note']}")
    if cmp_.get("medusa_effective_vs_gpu_x") is not None:
        print(
            f"  MEDUSA effective / GPU: {cmp_['medusa_effective_vs_gpu_x']:.2f}x "
            f"(includes {comparison['medusa']['draft_heads']}x speculative factor)"
        )

    print("\n--- Architectural projection (not measured on FPGA) ---")
    print(f"  INT4 MAC density: {arch['int4_mac_density_x']:.1f}x")
    print(f"  Zero-skip blend: {arch['zero_skip_fraction']*100:.0f}% -> factor {1/(1-arch['zero_skip_fraction']):.2f}x")
    print(f"  Combined multiplier: {arch['combined_multiplier_x']:.2f}x")
    print(f"  Projected imprint tok/s: {arch['projected_imprint_tps']:,.0f}")
    print(f"  Projected vs GPU: {arch['projected_vs_gpu_x']:.2f}x")

    print(f"\n  {comparison['disclaimer']}")


def main():
    parser = argparse.ArgumentParser(description="BitbyBit sim vs GPU comparison (derived metrics)")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--medusa", action="store_true", help="Include MEDUSA effective throughput line")
    parser.add_argument("--rtx-tdp", type=float, default=110.0, help="GPU TDP watts (RTX 4070 default)")
    parser.add_argument("--bbb-power", type=float, default=10.0, help="Estimated FPGA watts (placeholder)")
    parser.add_argument(
        "--nvidia-json",
        default=os.path.join(_SCRIPT_DIR, "..", "sim", "gemma3_nvidia_benchmark.json"),
        help="Optional measured Gemma benchmark JSON (generation tok/s)",
    )
    parser.add_argument("--json-out", default="", help="Write comparison JSON to this path")
    args = parser.parse_args()

    sim = load_sim_metrics()

    nvidia = load_nvidia_json(args.nvidia_json)
    if nvidia and "avg_tokens_per_sec" in nvidia:
        gpu_data = {
            "latency_ms": nvidia.get("avg_latency_ms_per_token", 0) * args.seq_len / 1000,
            "throughput_tps": float(nvidia["avg_tokens_per_sec"]),
            "latency_ms_per_token": float(nvidia.get("avg_latency_ms_per_token", 0)),
            "device": nvidia.get("gpu_name", "NVIDIA GPU"),
            "seq_len": args.seq_len,
            "workload": f"Gemma generate ({nvidia.get('model', 'unknown')})",
        }
        print("--- Using NVIDIA JSON baseline ---")
    else:
        print("--- Benchmarking local GPU (MiniGPT) ---")
        gpu_data = benchmark_gpu_baseline(seq_len=args.seq_len)
        print(f"Device: {gpu_data['device']}")
        print(f"Throughput: {gpu_data['throughput_tps']:.2f} tok/s")

    comparison = generate_comparison(
        gpu_data, sim, medusa=args.medusa, rtx_tdp_w=args.rtx_tdp, bbb_power_w=args.bbb_power
    )
    print_report(comparison)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nWrote: {args.json_out}")


if __name__ == "__main__":
    main()
