# Science Expo Win Playbook (31-03-2026)

Status: evidence-backed and reproducible  
Project: BitbyBit Custom GPU (simulation-measured at 100 MHz)
As-of anchor update: 2026-04-01 (latest validated run_id: 20260401-142256)

---

## 1) Executive Snapshot

1. Full regression status: **55 modules, 323 PASS, 0 FAIL**.
2. Production benchmark anchor run: **run_id 20260401-142256**.
3. Base vs imprint latency: **358 cycles -> 112 cycles**.
4. Mean speedup: **3.1964x**.
5. Latency reduction: **68.7151%**.
6. Throughput uplift: **279,329 tok/s -> 892,857 tok/s** (**+219.6435%**).
7. MEDUSA effective throughput (exploratory metric, not primary judge claim): **837,988 tok/s -> 2,678,571 tok/s**.
8. Benchmark matrix size: **20 workloads, 5 warmup runs, 20 measured runs, 400 measured samples**.
9. Contract validation: **PASS** for payload + schema + proof-pack.
10. WS1 scale/parity status: deterministic sweep across dims (16/32/64) is now fully passing with fail-close gate green.
11. Debate-converged expo policy: no new speculative RTL feature claims before expo; parity closure + claim integrity + demo resilience are P0.

---

## 1.1) Anchor Policy (Claim Safety)

1. Judge-facing performance claims must cite full-matrix benchmark run_id `20260401-142256`.
2. Quick-mode orchestration runs are gating checkpoints and must be labeled non-claim evidence.
3. WS1 parity claims must cite enforced run_id `20260401-085547`.
4. Every quoted metric should include run_id, workload_count_effective, and measured_runs.

---

## 2) What Was Run In This Campaign

## A) Full regression (hardware correctness breadth)

Command:

```powershell
powershell -ExecutionPolicy Bypass -File "d:\Projects\BitbyBit\custom_gpu_project\scripts\run_all_tests.ps1" *> "d:\Projects\BitbyBit\custom_gpu_project\sim\full_regression_20260401.log"
```

Outcome:
1. 55 modules tested.
2. 323 total PASS.
3. 0 total FAIL.

Evidence artifact:
1. `sim/full_regression_20260401.log`

## B) Expanded production benchmark matrix (showcase run)

Command:

```powershell
powershell -ExecutionPolicy Bypass -Command "Set-Location 'd:\Projects\BitbyBit\custom_gpu_project'; powershell -ExecutionPolicy Bypass -File '.\scripts\run_production_demo.ps1' -WorkloadMode matrix -WarmupRuns 5 -MeasuredRuns 20 -WorkloadCount 20 -WorkloadSeed 20260331"
```

Outcome:
1. run_id: `20260401-142256`
2. workload_count_effective: 20
3. warmup_runs: 5
4. measured_runs: 20
5. sample_count: 400
6. mean speedup_x: 3.1964
7. base_cycles_mean: 358
8. imprint_cycles_mean: 112
9. workload_coverage_pct: 15.625

Evidence artifacts:
1. `sim/compare_summary_latest.json`
2. `sim/bench_runs/20260401-142256/compare_summary.json`
3. `sim/phase3_benchmark_proof_pack.json`
4. `sim/phase3_benchmark_proof_pack.csv`

## C) Proof-pack contract validation (fail-closed integrity)

Command:

```powershell
python "d:\Projects\BitbyBit\custom_gpu_project\scripts\validate_benchmark_payload.py" --input "d:\Projects\BitbyBit\custom_gpu_project\sim\compare_summary_latest.json" --schema "d:\Projects\BitbyBit\custom_gpu_project\sim\benchmark_schema.json" --proof-pack "d:\Projects\BitbyBit\custom_gpu_project\sim\phase3_benchmark_proof_pack.json"
```

Outcome:
1. `[OK] benchmark payload validation passed`
2. `[OK] run_id=20260401-142256`

## D) Comprehensive optimization benchmark (architecture-level study)

Command:

```powershell
python "d:\Projects\BitbyBit\custom_gpu_project\scripts\benchmark_throughput.py"
```

Key outcomes:
1. Baseline latency estimate: 9170.99 ms.
2. Pipeline + MemBW + INT4: 27.8x speedup.
3. ALL + pruning 50%: 41.6x speedup.
4. ALL + 2:4 structured sparsity: 41.7x speedup.
5. Best overall (ALL + pruning 70%): **54.4x speedup**, 234.21 ms latency, 79.0% zero-skip.

## E) WS1 scale/parity expanded matrix (scientific rigor track)

Command (expanded sweep):

```powershell
python "d:\Projects\BitbyBit\custom_gpu_project\scripts\run_ws1_scale_proof.py" --dims 16,32,64 --workload-count 24 --workload-seed 20260331 --token-space 16 --position-space 8 --seq-len 32
```

Command (fail-close enforcement):

```powershell
python "d:\Projects\BitbyBit\custom_gpu_project\scripts\run_ws1_scale_proof.py" --dims 16,32,64 --workload-count 24 --workload-seed 20260331 --token-space 16 --position-space 8 --seq-len 32 --enforce-gate
```

Outcome:
1. run_id (non-enforced): `20260331-114415`
2. run_id (enforced): `20260401-085547`
3. dim=16: avg_cycles/token=601.25, zero_skip=3.09%, token_pass_count=24/24
4. dim=32: avg_cycles/token=1097.25, zero_skip=2.46%, token_pass_count=24/24
5. dim=64: avg_cycles/token=2089.25, zero_skip=13.07%, token_pass_count=24/24
6. parity overall: true (dim16/dim32/dim64 all PASS)
7. enforce-gate exits zero as designed when parity is green

Evidence artifacts:
1. `sim/dim_sweep_report.json`
2. `sim/parity_report.json`

---

## 3) Judge-Facing Claim Card (Use This Verbatim)

Claim:
1. In a seeded 20-workload matrix at 100 MHz with 400 measured paired samples, BitbyBit achieves **3.1964x** full-model latency speedup (**358 -> 112 cycles**) and **892,857 tok/s** on the imprint path.
2. MEDUSA-effective throughput is tracked as an exploratory metric and is excluded from the primary judge claim envelope for this expo cycle.

Why this is credible:
1. Regression breadth is green (323/323).
2. Raw logs + compare summary + proof-pack + schema validation are all present.
3. Run_id and seed are disclosed.
4. Statistical fields are included in the artifact (mean, stdev, CI, outlier counts).
5. Pipeline is fail-closed when parity gate is enforced.
6. WS1 scale parity is validated across dim16/dim32/dim64 with 24/24 token pass counts per dim.

Caveat (must always be stated):
1. Metrics are simulation-measured at 100 MHz on RTL/testbench flow, not post-route silicon measurements.
2. Cross-dimension parity closure claim is tied to run_id `20260401-085547` and its exact seeded workload configuration.

---

## 4) Fair Competitor Framing (So Judges Trust The Comparison)

## A) Use benchmark governance language

1. Use MLPerf terms: Closed division for apples-to-apples, Open division for innovation framing.
2. Report both latency-style and throughput-style metrics.
3. If power is discussed, explicitly state whether it is measured at wall/system level or estimated.

## B) Compare classes, not cherry-picked single numbers

For external context in slides:
1. NVIDIA Jetson Orin family public claims include up to **275 TOPS** (AGX Orin), **157 TOPS** (Orin NX), **67 TOPS** (Orin Nano), with configurable power bands.
2. Frame BitbyBit as an **architecture-specialized deterministic inference pipeline** with explicit reproducibility artifacts, not as a generic all-workload TOPS engine.

## C) Normalize your claims

Always include:
1. cycles/token
2. tok/s
3. effective tok/s (if speculative/draft decoding is counted)
4. workload count + seed + sample count
5. model path label (baseline vs imprint)

Optional normalization formulas for slide appendix:

$$
\text{Latency Reduction \%} = \frac{\text{base cycles} - \text{imprint cycles}}{\text{base cycles}} \times 100
$$

$$
\text{Speedup} = \frac{\text{base cycles}}{\text{imprint cycles}}
$$

$$
\text{Throughput Uplift \%} = \frac{\text{imprint tok/s} - \text{base tok/s}}{\text{base tok/s}} \times 100
$$

---

## 5) How To Win (Concise Strategy)

1. Lead with one hard metric pair: **358 -> 112 cycles (3.1964x)** and immediately show the artifact path.
2. Show rigor before hype: 323/323 regression + schema-validated proof-pack + deterministic run_id.
3. Separate claims clearly: baseline vs imprint, measured vs estimated, simulation vs silicon.
4. Present ablation logic: pipeline/memory/sparsity contributions (27.8x -> 41.7x -> 54.4x in architecture model).
5. Preempt judge skepticism with caveats yourself (simulation scope and seeded-run reproducibility boundaries).
6. Emphasize engineering maturity: fail-closed gates, reproducible commands, machine-readable evidence.
7. End with roadmap credibility: what is already proven vs what remains to close parity and hardware PPA.
8. Do not headline speculative/decode-effective metrics unless explicitly marked exploratory.

---

## 6) 60-Second Judge Pitch

"We built a custom RTL GPU inference pipeline and validated it with fail-closed evidence flow. In our latest seeded matrix benchmark run (20 workloads, 400 measured paired samples, run_id 20260401-142256), we reduced full-model latency from 358 to 112 cycles at 100 MHz, which is a 3.1964x speedup and about 68.7% lower latency. Throughput rises from 279,329 to 892,857 tokens per second on the imprint path. We backed every claim with raw logs, compare summary, proof-pack, and schema validation. We also validated WS1 cross-dimension parity closure with fail-close enforcement (run_id 20260401-085547), where dim16/dim32/dim64 all pass 24/24 token checks." 

---

## 7) Repro Command Bundle (Copy/Paste)

```powershell
cd D:\Projects\BitbyBit\custom_gpu_project

# 1) Full regression artifact
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1 *> .\sim\full_regression_20260401.log

# 2) Expanded production benchmark
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1 -WorkloadMode matrix -WarmupRuns 5 -MeasuredRuns 20 -WorkloadCount 20 -WorkloadSeed 20260331

# 3) Contract validation
python .\scripts\validate_benchmark_payload.py --input .\sim\compare_summary_latest.json --schema .\sim\benchmark_schema.json --proof-pack .\sim\phase3_benchmark_proof_pack.json

# 4) Throughput architecture benchmark
python .\scripts\benchmark_throughput.py

# 5) WS1 sweep + fail-close gate check
python .\scripts\run_ws1_scale_proof.py --dims 16,32,64 --workload-count 24 --workload-seed 20260331 --token-space 16 --position-space 8 --seq-len 32
python .\scripts\run_ws1_scale_proof.py --dims 16,32,64 --workload-count 24 --workload-seed 20260331 --token-space 16 --position-space 8 --seq-len 32 --enforce-gate
```

---

## 8) External Reference Anchors

1. MLPerf Inference Edge: https://mlcommons.org/benchmarks/inference-edge/
2. MLPerf Inference Datacenter: https://mlcommons.org/benchmarks/inference-datacenter/
3. MLPerf rules repository: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
4. NVIDIA Jetson Orin page: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
5. ISEF international rules index: https://www.societyforscience.org/isef/international-rules/
6. ACM artifact review/badging (v1.0 page with link to current v1.1): https://www.acm.org/publications/policies/artifact-review-badging
7. SPEC CPU documentation index (reporting/rules orientation): https://www.spec.org/cpu2017/Docs/

---

## 9) Round 2 Debate-Converged Protocol (10-Day)

Consensus participants:
1. Chief Hardware Architect
2. Verification and Numerical Correctness Lead
3. Benchmark and Claims Governance Lead
4. Science Expo Competitive Strategy Coach

Converged non-negotiables:
1. No speculative RTL feature additions before expo unless needed for bug fixing.
2. Parity closure for dim16/dim32 remains P0; fail-close gate behavior must remain unchanged.
3. All public metrics must come from one validated artifact source.
4. Day-8 freeze: only critical bug fixes and rehearsal corrections after Day 8.

Day-range plan:
1. Days 1-2: isolate dim16/dim32 root cause with reproducible evidence.
2. Days 3-4: apply minimal fix and rerun WS1 (`--enforce-gate` included).
3. Days 5-6: sync website/slides/demo metrics to validated artifacts only.
4. Days 7-8: run claim audit and adversarial judge Q&A rehearsals.
5. Days 9-10: perform full demo rehearsals, go/no-go review, and freeze.

Approved one-liners:
1. "WS1 parity is validated at dim16/dim32/dim64 under run_id 20260401-085547."
2. "All published metrics are reproducible from artifacted runs."
3. "Fail-close gate behavior is enforced and currently green for WS1."

Banned one-liners:
1. "These results generalize to all workloads and scales."
2. "Speculative execution already delivers production gains."
3. "Simulation numbers equal post-route silicon numbers."

---

This playbook is designed to maximize judge trust: high-signal metrics, reproducibility, transparent caveats, and a clear path from benchmark claim to raw evidence.
