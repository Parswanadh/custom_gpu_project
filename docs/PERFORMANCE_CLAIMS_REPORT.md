# Performance Claims Report: BitbyBit Custom GPU

**Status:** Simulation-measured baseline + honest projection framework  
**Clock:** 100 MHz (cycle-accurate Verilog, `full_model_inference_*_tb`)  
**Last reconciled:** 2026-05-22

---

## Executive Summary

BitbyBit shows **3.20× lower latency** on the measured mini full-model imprint path (358 → 112 cycles) at 100 MHz, yielding **892,857 tokens/s** sustained in simulation. A prior headline of **2.67M tok/s** was **not** inconsistent math at 130 cycles/token — it was **MEDUSA speculative effective throughput** (3 draft heads × imprint path). That number must never be quoted as single-path sustained throughput or compared to NVIDIA without labeling speculative accounting and workload mismatch.

**Do not claim ~2,465× vs a consumer GPU** without: matched model, matched batch/sequence, silicon or FPGA timing, and explicit separation of sim vs GPU measurement.

---

## 1. Measured in Simulation

Source: `sim/phase3_benchmark_proof_pack.json`, `sim/compare_summary_latest.json`, testbenches `full_model_inference_tb.v` / `full_model_inference_imprint_tb.v`.

| Metric | Base path | Imprint path | Notes |
|--------|-----------|--------------|-------|
| **Cycles / token** (full mini model) | 358 | 112 | One forward pass, 12 mini layers @ 100 MHz |
| **Throughput** | 279,329 tok/s | 892,857 tok/s | `100 MHz ÷ cycles` |
| **Imprint speedup** | — | **3.196×** | 358 / 112 |
| **Imprint latency** | 3.58 µs | 1.12 µs | Per token, sim only |
| **MEDUSA effective throughput** | 837,988 tok/s | **2,678,571 tok/s** | `3 ×` imprint; draft speculation, not IPC |
| **Zero-skip (software model)** | ~26% blended | — | OPT-125M FFN ReLU + Q8.8; see `scripts/benchmark_throughput.py` |
| **Zero-skip (RTL counters)** | TBD on FPGA | — | Use `perf_counters` MAC/skip tallies |

### Mathematical consistency (required)

```
throughput (tok/s) = clock_hz / cycles_per_token

@ 100 MHz:
  100,000,000 / 358 = 279,329 tok/s   (base)
  100,000,000 / 112 = 892,857 tok/s   (imprint)
  3 × 892,857       = 2,678,571 tok/s (MEDUSA effective only)
```

### Legacy README numbers (different metric — do not mix)

| Claim | Value | Correct @ 100 MHz |
|-------|-------|-------------------|
| Avg cycles/token (GPT-2 steady-state doc) | 130 | **769,230 tok/s** |
| Throughput @ 100 MHz (old README) | 2.67M tok/s | Requires **37.5 cy/token** OR is MEDUSA effective on imprint |
| Dynamic latency 341 cy / 128 tok | 21.9 cy/token avg | Different measurement window than single-token full-model TB |

---

## 2. Projected vs NVIDIA (Matched Workload Required)

### What we can compare today

| Side | Measurement | Typical workload |
|------|-------------|------------------|
| **BitbyBit** | RTL simulation @ 100 MHz | Mini imprint model, 1 token forward |
| **NVIDIA** | `scripts/gemma3_benchmark_nvidia.py` or `compare_bitbybit_vs_rtx.py` | Gemma-3 generate or PyTorch MiniGPT forward |

These are **not** the same workload. Any ratio is **indicative only** until Gemma-3 (or chosen model) runs on FPGA with the same tokenization, layers, and precision.

### Documented throughput multipliers (architecture)

Apply only to an **already measured** baseline (sim or silicon):

| Multiplier | Factor | Basis |
|------------|--------|--------|
| **INT4 MAC density** | **4×** | 4 parallel INT4 MACs/cycle in documented RTL mode |
| **Zero-skip** | **~1.35×** | ~26% MAC skip → 1/(1−0.26) from OPT-125M software benchmark |
| **Variable precision** | **1×** (TBD) | Mixed-precision fetch paths; validate before claiming |

**Combined analytical ceiling (vs imprint sim baseline):**

```
892,857 × 4 × 1.35 × 1 ≈ 4.82M tok/s  (projection, not measured)
```

Example: imprint sim **892k tok/s** vs GPU MiniGPT **~1,083 tok/s** → **~825×** — still **not** a fair “vs NVIDIA” claim until model and silicon match.

### Fair NVIDIA comparison checklist

1. Same model family (e.g. Gemma-3 270M INT4 weights on FPGA vs `gemma3_benchmark_nvidia.py` on RTX).
2. Same metric: **sustained decode tok/s** (not MEDUSA effective unless both sides use speculative decoding).
3. Same batch size and sequence length.
4. Report **clock, cycles/token, power (W), J/token** together.

---

## 3. FPGA Validation Checklist

Before external “faster than NVIDIA” statements:

- [ ] **Timing closure** at target Fmax on target part (report WNS/TNS).
- [ ] **Cycles/token** from on-chip `perf_counters` or ILA (not only testbench `$time`).
- [ ] **Throughput** recomputed: `Fmax × 10⁶ / cycles_per_token`.
- [ ] **Functional** match to golden vectors (`weights/gemma3_golden/`).
- [ ] **Zero-skip %** from hardware counters vs software estimate.
- [ ] **Power** via VCD/XPE or board shunt; compute J/token.
- [ ] **NVIDIA baseline** re-run same day, same driver, same `gemma3_nvidia_benchmark.json` schema.
- [ ] **Workload table** in report: sim vs FPGA vs GPU columns side-by-side.

See [PERFORMANCE_VALIDATION.md](PERFORMANCE_VALIDATION.md) for step-by-step proof flow.

---

## 4. Reproducibility

```powershell
cd D:\Projects\BitbyBit\custom_gpu_project
python .\scripts\compare_bitbybit_vs_rtx.py --medusa
python .\scripts\build_phase3_benchmark_proof_pack.py
python .\scripts\validate_benchmark_payload.py `
  --input .\sim\compare_summary_latest.json `
  --schema .\sim\benchmark_schema.json `
  --proof-pack .\sim\phase3_benchmark_proof_pack.json
```

---

## 5. Suggested External Wording

> “At 100 MHz cycle-accurate simulation, BitbyBit imprint inference completes one mini-model token in **112 cycles** (**892k tok/s**), **3.2× faster** than the **358-cycle** base path. MEDUSA adds **3× speculative effective throughput** to **~2.68M tok/s** — a draft-head metric, not sustained single-path IPC. FPGA and matched-GPU benchmarks are required before datacenter comparison claims.”

---

*Report aligned with `docs/Performance_Methodology.md` (repo root) and `scripts/perf_metrics.py`.*
