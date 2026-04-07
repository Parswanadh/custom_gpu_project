# SOTA Benchmark Showcase (31-03-2026)

Run ID: 20260331-102611
Method: simulation-measured, matrix workload sweep, fail-closed benchmark pipeline with schema/proof-pack contract validation
Clock: 100 MHz

---

## 1) Showcase Headline Metrics

1. Baseline full-model latency: 358 cycles (3.580 us)
2. Imprint full-model latency: 112 cycles (1.120 us)
3. Mean speedup: 3.1964x
4. Baseline throughput: 279,329 tokens/s
5. Imprint throughput: 892,857 tokens/s
6. Baseline MEDUSA effective throughput: 837,988 tokens/s
7. Imprint MEDUSA effective throughput: 2,678,571 tokens/s
8. Workload diversity: 20 unique token/position workloads
9. Measured sample count: 400 (20 measured runs x 20 workloads)
10. Coverage reported: 15.625% of token/position space

---

## 2) Reproducibility Command

```powershell
cd D:\Projects\BitbyBit\custom_gpu_project
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1 -WorkloadMode matrix -WarmupRuns 5 -MeasuredRuns 20 -WorkloadCount 20 -WorkloadSeed 20260331
python .\scripts\build_phase3_benchmark_proof_pack.py
python .\scripts\validate_benchmark_payload.py --input .\sim\compare_summary_latest.json --schema .\sim\benchmark_schema.json --proof-pack .\sim\phase3_benchmark_proof_pack.json
```

---

## 3) Artifact Bundle (Evidence)

1. sim/compare_summary_latest.json
2. sim/phase3_benchmark_proof_pack.json
3. sim/phase3_benchmark_proof_pack.csv
4. sim/benchmark_schema.json
5. sim/demo_full_model_base_w*_r*.log
6. sim/demo_full_model_imprint_w*_r*.log
7. sim/demo_gpu_system_top_v2.log

Contract validation result:
1. [OK] benchmark payload validation passed
2. [OK] run_id=20260331-102611

---

## 4) Notes for External Presentation

1. All metrics are simulation-measured at 100 MHz and should be presented with that caveat.
2. Baseline and imprint numbers must always be shown side-by-side to avoid ambiguity.
3. The benchmark run is matrix-mode and seeded for reproducibility.
4. Payload/proof-pack consistency is now validated fail-closed.

---

## 5) Suggested One-Liner for Demo Slides

"In a seeded 20-workload, 400-sample matrix benchmark at 100 MHz, BitbyBit achieves 3.1964x latency speedup (358 -> 112 cycles) and up to 2,678,571 effective tok/s with MEDUSA on the imprint path, with reproducible evidence artifacts."
