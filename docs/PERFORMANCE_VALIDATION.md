# Performance Validation on FPGA

How to turn simulation metrics into **defensible** performance claims against NVIDIA (or other) baselines.

---

## 1. Metric Definitions (use everywhere)

| Symbol | Definition |
|--------|------------|
| `F` | Achieved clock frequency (Hz) on FPGA |
| `C` | Measured **cycles per token** (steady-state decode) |
| `T` | Throughput = `F / C` tokens/s |
| `P` | Average power (W) while sustaining `T` |
| `E` | Energy per token = `P / T` (J/token) |

**Rule:** Never publish `T` without `F` and `C`. Scripts derive `T` via `scripts/perf_metrics.py`.

---

## 2. Evidence Ladder

| Level | Evidence | Claim strength |
|-------|----------|----------------|
| L0 | Analytical spreadsheet | Internal only |
| L1 | Cycle-accurate Verilog TB | “Sim @ 100 MHz” |
| L2 | Post-route timing @ target Fmax | “Timing closed @ X MHz” |
| L3 | FPGA ILA / perf counters | “Measured cycles/token” |
| L4 | End-to-end host + FPGA + GPU same day | “Matched workload comparison” |

Current public numbers are mostly **L1**. Marketing comparisons to NVIDIA require **L4**.

---

## 3. Simulation Baseline (L1)

**Artifacts**

- `sim/phase3_benchmark_proof_pack.json`
- `sim/compare_summary_latest.json`
- Logs: `sim/demo_full_model_*.log`

**Reproduce**

```powershell
cd D:\Projects\BitbyBit\custom_gpu_project
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1 `
  -WorkloadMode matrix -WarmupRuns 5 -MeasuredRuns 20
python .\scripts\build_phase3_benchmark_proof_pack.py
python .\scripts\validate_benchmark_payload.py `
  --input .\sim\compare_summary_latest.json `
  --schema .\sim\benchmark_schema.json `
  --proof-pack .\sim\phase3_benchmark_proof_pack.json
```

**Acceptance (example, mini full-model @ 100 MHz)**

- Base: 358 cycles/token → 279,329 tok/s
- Imprint: 112 cycles/token → 892,857 tok/s
- Imprint speedup: 3.196× (±0.1% across matrix workloads)

---

## 4. FPGA Bring-Up (L2–L3)

### 4.1 Bitstream and clocks

1. Build design for target board; record part, voltage, temperature.
2. Run static timing; record **achieved Fmax** (not only 100 MHz sim assumption).
3. Recompute: `T_fpga = Fmax / C`.

### 4.2 Cycles per token on silicon

1. Enable `perf_counters` (cycles, MACs issued, MACs skipped, stalls).
2. Warm up ≥ 2× pipeline depth; measure ≥ 1000 tokens.
3. `C = (counter_end - counter_start) / token_count`.

### 4.3 Functional correctness

1. Load INT4 weights from `weights/gemma3_fpga/` (`gemma3_benchmark_nvidia.py` export).
2. Compare layer activations to `weights/gemma3_golden/` within Q8.8 tolerance.
3. Fail closed if any layer mismatch.

### 4.4 Zero-skip validation

1. Read skip counter / total MAC counter per layer.
2. Report `skip_pct = skipped / (skipped + executed)`.
3. Compare to software estimate (~26% blended on OPT-125M); investigate large deltas.

### 4.5 Power

1. Measure board power under sustained inference (not idle bitstream).
2. `E = P_avg / T_fpga`.

---

## 5. NVIDIA / GPU Baseline (L4)

```powershell
python .\scripts\gemma3_benchmark_nvidia.py
```

Produces `sim/gemma3_nvidia_benchmark.json` with `avg_tokens_per_sec`, latency, memory.

**Compare fairly**

```powershell
python .\scripts\compare_bitbybit_vs_rtx.py `
  --nvidia-json .\sim\gemma3_nvidia_benchmark.json `
  --medusa
```

Read output labels: sim imprint vs GPU generation are **different workloads** until FPGA runs the same Gemma graph.

---

## 6. Architectural Multipliers (projection only)

After L3 establishes `T_fpga`, optional analytical ceiling:

```
T_projected = T_fpga × 4 (INT4 MAC density) × 1/(1 - zero_skip) × variable_precision_gain
```

Document each factor in the report table. **Do not** multiply sim MEDUSA effective throughput by these factors again (double counting).

---

## 7. Sign-Off Template

| Field | Sim (100 MHz) | FPGA | NVIDIA GPU |
|-------|---------------|------|------------|
| Model | Mini imprint | Gemma-3 270M | Same as FPGA |
| Cycles/token | 112 / 358 | measured | N/A |
| Clock (MHz) | 100 | measured | boost clock |
| Throughput (tok/s) | 892k / 279k | `F/C` | JSON benchmark |
| Power (W) | N/A | measured | TDP or meter |
| J/token | N/A | `P/T` | `P/T` |
| Speculative? | MEDUSA optional | same policy | same policy |

**Approved claim example:**  
“FPGA at 125 MHz sustained **X tok/s** on Gemma-3 270M INT4 decode, **Y×** RTX 4070 measured **Z tok/s** same prompt/date, **P W** vs **Q W**.”

---

## 8. Common Failures

| Mistake | Fix |
|---------|-----|
| Quote 2.67M tok/s as @ 100 MHz sustained | Label as MEDUSA 3× effective or give `C=37.5` |
| Mix 130 cy/token with 112 cy/token | Separate GPT-2 steady-state vs mini full-model TB |
| Compare sim to GPU without workload column | Add model + measurement table |
| Apply 4× MAC + MEDUSA 3× | Pick one speculative story |
| Use 200 MHz in scripts while TB is 100 MHz | Single `clock_hz` in `perf_metrics.py` |

---

## References

- [PERFORMANCE_CLAIMS_REPORT.md](PERFORMANCE_CLAIMS_REPORT.md)
- [../docs/Performance_Methodology.md](../../docs/Performance_Methodology.md) (repo root)
- `scripts/perf_metrics.py`, `scripts/compare_bitbybit_vs_rtx.py`
