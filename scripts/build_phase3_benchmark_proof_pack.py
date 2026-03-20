#!/usr/bin/env python3
"""
Build a reproducible benchmark proof-pack from phase3 simulation logs.

Inputs (expected):
  sim/phase3_*_bench.log (or sim/measured_*_bench.log when present)
  Optional:
    sim/measured_full_model_imprint.log

Outputs:
  sim/phase3_benchmark_proof_pack.json
  sim/phase3_benchmark_proof_pack.csv
"""

import csv
import json
import math
import re
import statistics
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def find(pattern: str, text: str, cast=int, default=None):
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return default
    value = m.group(1)
    return cast(value) if cast else value


def find_pair(pattern: str, text: str, cast=int, default=(None, None)):
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return default
    return cast(m.group(1)), cast(m.group(2))


def pick_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def validate_compare_summary(compare_summary: dict) -> None:
    required_top = (
        "run_id",
        "workload_mode",
        "warmup_runs",
        "measured_runs",
        "workloads",
        "samples",
        "stats",
        "run_quality",
        "system_environment",
        "file_integrity",
    )
    missing_top = [k for k in required_top if k not in compare_summary]
    if missing_top:
        raise ValueError(f"compare_summary_latest.json missing keys: {missing_top}")

    samples = compare_summary.get("samples", [])
    workload_mode = str(compare_summary.get("workload_mode", ""))
    warmup_runs = int(compare_summary.get("warmup_runs", -1))
    measured_runs = int(compare_summary.get("measured_runs", 0))
    workloads = compare_summary.get("workloads", [])
    workload_count_requested = int(compare_summary.get("workload_count_requested", -1))
    workload_count_effective = int(compare_summary.get("workload_count_effective", -1))
    if warmup_runs < 0:
        raise ValueError("compare_summary_latest.json has negative warmup_runs")
    if measured_runs <= 0:
        raise ValueError("compare_summary_latest.json has non-positive measured_runs")
    if workload_mode not in {"single", "matrix"}:
        raise ValueError(f"compare_summary_latest.json has invalid workload_mode={workload_mode!r}")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("compare_summary_latest.json has no workloads")
    if not isinstance(samples, list):
        raise ValueError("compare_summary_latest.json samples must be a list")
    if workload_count_requested <= 0:
        raise ValueError("compare_summary_latest.json has non-positive workload_count_requested")
    if workload_count_effective <= 0:
        raise ValueError("compare_summary_latest.json has non-positive workload_count_effective")
    if workload_count_effective != len(workloads):
        raise ValueError(
            f"workload_count_effective mismatch: expected {len(workloads)}, got {workload_count_effective}"
        )
    if workload_count_requested < workload_count_effective:
        raise ValueError(
            "workload_count_requested must be >= workload_count_effective after clamping"
        )

    if workload_mode == "matrix":
        if str(compare_summary.get("workload_generation", "")) != "seeded_random_unique":
            raise ValueError("matrix workload mode must use workload_generation=seeded_random_unique")
        if "workload_seed" not in compare_summary:
            raise ValueError("matrix workload mode missing workload_seed")
        int(compare_summary.get("workload_seed"))

    workload_keys: set[tuple[int, int]] = set()
    for idx, workload in enumerate(workloads):
        if "TokenId" not in workload or "Position" not in workload:
            raise ValueError(f"workloads[{idx}] missing TokenId/Position keys")
        workload_key = (int(workload["TokenId"]), int(workload["Position"]))
        if workload_key in workload_keys:
            raise ValueError(f"duplicate workload entry detected at index {idx}: {workload_key}")
        workload_keys.add(workload_key)

    expected_samples = measured_runs * len(workloads)
    if len(samples) != expected_samples:
        raise ValueError(
            f"paired sample count mismatch: expected {expected_samples}, got {len(samples)}"
        )

    required_sample = (
        "workload_index",
        "run_index",
        "phase",
        "token_id",
        "position",
        "base_cycles",
        "imprint_cycles",
        "base_tokens_per_second",
        "imprint_tokens_per_second",
        "base_medusa_toks",
        "imprint_medusa_toks",
        "speedup_x",
    )
    seen_run_keys: set[tuple[int, int]] = set()
    per_workload_counts: dict[int, int] = {}
    for idx, sample in enumerate(samples):
        missing = [k for k in required_sample if k not in sample]
        if missing:
            raise ValueError(f"sample[{idx}] missing keys: {missing}")
        if str(sample["phase"]) != "measured":
            raise ValueError(f"sample[{idx}] has non-measured phase: {sample['phase']}")
        workload_index = int(sample["workload_index"])
        run_index = int(sample["run_index"])
        if workload_index < 1 or workload_index > len(workloads):
            raise ValueError(f"sample[{idx}] has out-of-range workload_index={workload_index}")
        if run_index <= warmup_runs:
            raise ValueError(
                f"sample[{idx}] run_index={run_index} is not a measured run for warmup_runs={warmup_runs}"
            )
        run_key = (workload_index, run_index)
        if run_key in seen_run_keys:
            raise ValueError(f"duplicate sample run tuple detected: workload={workload_index}, run={run_index}")
        seen_run_keys.add(run_key)
        per_workload_counts[workload_index] = per_workload_counts.get(workload_index, 0) + 1
        if float(sample["base_cycles"]) <= 0 or float(sample["imprint_cycles"]) <= 0:
            raise ValueError(f"sample[{idx}] has non-positive cycles")
        if float(sample["base_tokens_per_second"]) <= 0 or float(sample["imprint_tokens_per_second"]) <= 0:
            raise ValueError(f"sample[{idx}] has non-positive token throughput")
        if float(sample["base_medusa_toks"]) <= 0 or float(sample["imprint_medusa_toks"]) <= 0:
            raise ValueError(f"sample[{idx}] has non-positive MEDUSA throughput")
        if float(sample["speedup_x"]) <= 0:
            raise ValueError(f"sample[{idx}] has non-positive speedup")

    for workload_index in range(1, len(workloads) + 1):
        sample_count = per_workload_counts.get(workload_index, 0)
        if sample_count != measured_runs:
            raise ValueError(
                f"workload_index={workload_index} sample count mismatch: expected {measured_runs}, got {sample_count}"
            )

    system_environment = compare_summary.get("system_environment", {})
    required_env = ("powershell_version", "computer_name", "processor", "iverilog_path", "vvp_path")
    missing_env = [k for k in required_env if not system_environment.get(k)]
    if missing_env:
        raise ValueError(f"compare_summary_latest.json missing system_environment fields: {missing_env}")

    file_integrity = compare_summary.get("file_integrity", {})
    for key in ("top_log_sha256", "base_log_sha256", "imprint_log_sha256"):
        value = file_integrity.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{64}", value):
            raise ValueError(f"compare_summary_latest.json invalid {key}: {value!r}")

    run_quality = compare_summary.get("run_quality", {})
    required_quality = (
        "warmup_meets_recommendation",
        "measured_meets_recommendation",
        "workload_diversity_ok",
        "workload_coverage_pct",
    )
    missing_quality = [k for k in required_quality if k not in run_quality]
    if missing_quality:
        raise ValueError(f"compare_summary_latest.json missing run_quality fields: {missing_quality}")
    coverage = float(run_quality.get("workload_coverage_pct"))
    if coverage < 0.0 or coverage > 100.0:
        raise ValueError(f"run_quality.workload_coverage_pct out of range: {coverage}")


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        raise ValueError("summarize_numeric received empty values")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = statistics.fmean(sorted_vals)
    median = statistics.median(sorted_vals)
    min_v = sorted_vals[0]
    max_v = sorted_vals[-1]

    stdev = statistics.stdev(sorted_vals) if n > 1 else None
    cv_pct = ((stdev / mean) * 100.0) if (stdev is not None and mean != 0.0) else None
    stderr = (stdev / math.sqrt(n)) if stdev is not None else None
    ci95_lower = (mean - 1.96 * stderr) if stderr is not None else None
    ci95_upper = (mean + 1.96 * stderr) if stderr is not None else None

    outlier_count = 0
    if n >= 4:
        q1, _, q3 = statistics.quantiles(sorted_vals, n=4, method="inclusive")
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = sum(1 for v in sorted_vals if v < lower or v > upper)

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": min_v,
        "max": max_v,
        "stdev": round(stdev, 4) if stdev is not None else None,
        "cv_pct": round(cv_pct, 4) if cv_pct is not None else None,
        "stderr": round(stderr, 4) if stderr is not None else None,
        "ci95_lower": round(ci95_lower, 4) if ci95_lower is not None else None,
        "ci95_upper": round(ci95_upper, 4) if ci95_upper is not None else None,
        "outlier_count": outlier_count,
    }


def parse_gpu_top(text: str) -> dict:
    baseline = find(r"MATMUL completed via optimized path in (\d+) cycles", text)
    mini = find(r"MINI imprint profile engaged in (\d+) cycles", text)
    gemma = find(r"GEMMA exported profile engaged in (\d+) cycles", text)
    speedup = find(r"speedup=([0-9]+\.[0-9]+)x", text, cast=float)
    passed, total = find_pair(r"Results:\s+(\d+)\s+PASSED,\s+(\d+)\s+FAILED", text)
    return {
        "benchmark": "gpu_system_top_v2_tb",
        "status": f"{passed}/{passed + total} PASS" if passed is not None else "unknown",
        "baseline_cycles": baseline,
        "mini_cycles": mini,
        "gemma_cycles": gemma,
        "mini_speedup_x": speedup,
    }


def parse_e2e(text: str) -> dict:
    total_cycles = find(r"TOTAL \(end-to-end\)\s+(\d+)\s+cy", text)
    passed, total = find_pair(r"End-to-End Pipeline Tests:\s+(\d+)\s+/\s+(\d+)\s+PASSED", text)
    return {
        "benchmark": "end_to_end_pipeline_tb",
        "status": f"{passed}/{total} PASS" if passed is not None else "unknown",
        "total_cycles": total_cycles,
    }


def parse_full_model(text: str) -> dict:
    total_cycles = find(r"TOTAL INFERENCE\s+(\d+)\s+cy", text)
    tps = find(r"Tokens/second:\s+~([0-9]+)", text)
    medusa_tps = find(r"effective\s+([0-9]+)\s+tok/s", text)
    passed, total = find_pair(r"12-Layer Inference Emulation Tests:\s+(\d+)\s+/\s+(\d+)\s+PASSED", text)
    return {
        "benchmark": "full_model_inference_tb",
        "status": f"{passed}/{total} PASS" if passed is not None else "unknown",
        "total_cycles": total_cycles,
        "tokens_per_second": tps,
        "medusa_effective_tokens_per_second": medusa_tps,
    }


def parse_full_model_imprint(text: str) -> dict:
    total_cycles = find(r"TOTAL INFERENCE\s+(\d+)\s+cy", text)
    tps = find(r"Tokens/second:\s+~([0-9]+)", text)
    medusa_tps = find(r"effective\s+([0-9]+)\s+tok/s", text)
    passed, total = find_pair(r"MINI Imprint Inference Tests:\s+(\d+)\s+/\s+(\d+)\s+PASSED", text)
    return {
        "benchmark": "full_model_inference_imprint_tb",
        "status": f"{passed}/{total} PASS" if passed is not None else "unknown",
        "total_cycles": total_cycles,
        "tokens_per_second": tps,
        "medusa_effective_tokens_per_second": medusa_tps,
    }


def parse_integration_speed(text: str) -> dict:
    passed, total = find_pair(r"Tests Passed:\s+(\d+)\s+/\s+(\d+)", text)
    gqa_cycles = find(r"GQA:\s+(\d+)\s+cycles", text)
    kv_cycles = find(r"KV Q\+DQ roundtrip:\s+(\d+)\s+cycles", text)
    comp_cycles = find(r"Compression:\s+(\d+)\s+cycles", text)
    softmax_cycles = find(r"Softmax:\s+(\d+)\s+cycles", text)
    return {
        "benchmark": "integration_speed_benchmark_tb",
        "status": f"{passed}/{total} PASS" if passed is not None else "unknown",
        "gqa_cycles": gqa_cycles,
        "kv_qdq_cycles": kv_cycles,
        "activation_compression_cycles": comp_cycles,
        "softmax_cycles": softmax_cycles,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    sim = root / "sim"

    logs = {
        "gpu_top": pick_existing(
            sim / "measured_gpu_system_top_v2_bench.log",
            sim / "phase3_gpu_system_top_v2_bench.log",
        ),
        "e2e": pick_existing(
            sim / "measured_e2e_pipeline_bench.log",
            sim / "phase3_e2e_pipeline_bench.log",
        ),
        "full_model": pick_existing(
            sim / "measured_full_model_base.log",
            sim / "phase3_full_model_bench.log",
        ),
        "integration_speed": pick_existing(
            sim / "measured_integration_speed_bench.log",
            sim / "phase3_integration_speed_bench.log",
        ),
        "full_model_imprint": pick_existing(
            sim / "measured_full_model_imprint.log",
            sim / "phase3_full_model_imprint.log",
        ),
    }

    required = ("gpu_top", "e2e", "full_model", "integration_speed")
    missing = [k for k in required if logs[k] is None]
    if missing:
        raise FileNotFoundError("Missing required benchmark logs for: " + ", ".join(missing))

    report = [
        parse_gpu_top(read_text(logs["gpu_top"])),
        parse_e2e(read_text(logs["e2e"])),
        parse_full_model(read_text(logs["full_model"])),
        parse_integration_speed(read_text(logs["integration_speed"])),
    ]

    if logs["full_model_imprint"] is not None:
        imprint_row = parse_full_model_imprint(read_text(logs["full_model_imprint"]))
        report.append(imprint_row)

        base_row = report[2]
        base_cycles = base_row.get("total_cycles")
        imprint_cycles = imprint_row.get("total_cycles")
        base_tps = base_row.get("tokens_per_second")
        imprint_tps = imprint_row.get("tokens_per_second")
        if base_cycles and imprint_cycles and base_tps and imprint_tps:
            report.append({
                "benchmark": "base_vs_imprint_full_model",
                "status": "measured",
                "base_cycles": base_cycles,
                "imprint_cycles": imprint_cycles,
                "speedup_x": round(base_cycles / imprint_cycles, 4),
                "base_tokens_per_second": base_tps,
                "imprint_tokens_per_second": imprint_tps,
            })

    compare_summary_latest = sim / "compare_summary_latest.json"
    if compare_summary_latest.exists():
        compare_summary = json.loads(read_text(compare_summary_latest))
        validate_compare_summary(compare_summary)
        samples = compare_summary.get("samples", [])
        if samples:
            base_cycles_vec = [float(s["base_cycles"]) for s in samples]
            imprint_cycles_vec = [float(s["imprint_cycles"]) for s in samples]
            speedup_vec = [float(s["speedup_x"]) for s in samples]
            base_stats = summarize_numeric(base_cycles_vec)
            imprint_stats = summarize_numeric(imprint_cycles_vec)
            speedup_stats = summarize_numeric(speedup_vec)
            grouped = {}
            for sample in samples:
                key = (sample.get("token_id"), sample.get("position"))
                grouped.setdefault(key, []).append(sample)

            if len(grouped) == 1:
                (token_id, position) = next(iter(grouped.keys()))
                report.append({
                    "benchmark": "base_vs_imprint_full_model_paired",
                    "status": "measured_paired",
                    "run_id": compare_summary.get("run_id"),
                    "warmup_runs": compare_summary.get("warmup_runs"),
                    "measured_runs": compare_summary.get("measured_runs"),
                    "token_id": token_id,
                    "position": position,
                    "sample_count": base_stats["count"],
                    "base_cycles_mean": base_stats["mean"],
                    "base_cycles_median": base_stats["median"],
                    "base_cycles_min": base_stats["min"],
                    "base_cycles_max": base_stats["max"],
                    "base_cycles_stdev": base_stats["stdev"],
                    "base_cycles_cv_pct": base_stats["cv_pct"],
                    "base_cycles_ci95_lower": base_stats["ci95_lower"],
                    "base_cycles_ci95_upper": base_stats["ci95_upper"],
                    "base_cycles_outlier_count": base_stats["outlier_count"],
                    "imprint_cycles_mean": imprint_stats["mean"],
                    "imprint_cycles_median": imprint_stats["median"],
                    "imprint_cycles_min": imprint_stats["min"],
                    "imprint_cycles_max": imprint_stats["max"],
                    "imprint_cycles_stdev": imprint_stats["stdev"],
                    "imprint_cycles_cv_pct": imprint_stats["cv_pct"],
                    "imprint_cycles_ci95_lower": imprint_stats["ci95_lower"],
                    "imprint_cycles_ci95_upper": imprint_stats["ci95_upper"],
                    "imprint_cycles_outlier_count": imprint_stats["outlier_count"],
                    "speedup_mean_x": speedup_stats["mean"],
                    "speedup_median_x": speedup_stats["median"],
                    "speedup_min_x": speedup_stats["min"],
                    "speedup_max_x": speedup_stats["max"],
                    "speedup_stdev_x": speedup_stats["stdev"],
                    "speedup_cv_pct": speedup_stats["cv_pct"],
                    "speedup_ci95_lower_x": speedup_stats["ci95_lower"],
                    "speedup_ci95_upper_x": speedup_stats["ci95_upper"],
                    "speedup_outlier_count": speedup_stats["outlier_count"],
                })
            else:
                report.append({
                    "benchmark": "base_vs_imprint_full_model_paired_matrix",
                    "status": "measured_paired_matrix",
                    "run_id": compare_summary.get("run_id"),
                    "warmup_runs": compare_summary.get("warmup_runs"),
                    "measured_runs": compare_summary.get("measured_runs"),
                    "workload_count": len(grouped),
                    "sample_count": base_stats["count"],
                    "base_cycles_mean": base_stats["mean"],
                    "base_cycles_median": base_stats["median"],
                    "base_cycles_min": base_stats["min"],
                    "base_cycles_max": base_stats["max"],
                    "base_cycles_stdev": base_stats["stdev"],
                    "base_cycles_cv_pct": base_stats["cv_pct"],
                    "base_cycles_ci95_lower": base_stats["ci95_lower"],
                    "base_cycles_ci95_upper": base_stats["ci95_upper"],
                    "base_cycles_outlier_count": base_stats["outlier_count"],
                    "imprint_cycles_mean": imprint_stats["mean"],
                    "imprint_cycles_median": imprint_stats["median"],
                    "imprint_cycles_min": imprint_stats["min"],
                    "imprint_cycles_max": imprint_stats["max"],
                    "imprint_cycles_stdev": imprint_stats["stdev"],
                    "imprint_cycles_cv_pct": imprint_stats["cv_pct"],
                    "imprint_cycles_ci95_lower": imprint_stats["ci95_lower"],
                    "imprint_cycles_ci95_upper": imprint_stats["ci95_upper"],
                    "imprint_cycles_outlier_count": imprint_stats["outlier_count"],
                    "speedup_mean_x": speedup_stats["mean"],
                    "speedup_median_x": speedup_stats["median"],
                    "speedup_min_x": speedup_stats["min"],
                    "speedup_max_x": speedup_stats["max"],
                    "speedup_stdev_x": speedup_stats["stdev"],
                    "speedup_cv_pct": speedup_stats["cv_pct"],
                    "speedup_ci95_lower_x": speedup_stats["ci95_lower"],
                    "speedup_ci95_upper_x": speedup_stats["ci95_upper"],
                    "speedup_outlier_count": speedup_stats["outlier_count"],
                })

                for (token_id, position), workload_samples in grouped.items():
                    wl_base = [float(s["base_cycles"]) for s in workload_samples]
                    wl_imprint = [float(s["imprint_cycles"]) for s in workload_samples]
                    wl_speedup = [float(s["speedup_x"]) for s in workload_samples]
                    wl_base_stats = summarize_numeric(wl_base)
                    wl_imprint_stats = summarize_numeric(wl_imprint)
                    wl_speedup_stats = summarize_numeric(wl_speedup)
                    report.append({
                        "benchmark": "base_vs_imprint_full_model_paired_workload",
                        "status": "measured_paired_workload",
                        "run_id": compare_summary.get("run_id"),
                        "token_id": token_id,
                        "position": position,
                        "sample_count": wl_base_stats["count"],
                        "base_cycles_mean": wl_base_stats["mean"],
                        "base_cycles_median": wl_base_stats["median"],
                        "base_cycles_min": wl_base_stats["min"],
                        "base_cycles_max": wl_base_stats["max"],
                        "base_cycles_stdev": wl_base_stats["stdev"],
                        "base_cycles_cv_pct": wl_base_stats["cv_pct"],
                        "base_cycles_ci95_lower": wl_base_stats["ci95_lower"],
                        "base_cycles_ci95_upper": wl_base_stats["ci95_upper"],
                        "base_cycles_outlier_count": wl_base_stats["outlier_count"],
                        "imprint_cycles_mean": wl_imprint_stats["mean"],
                        "imprint_cycles_median": wl_imprint_stats["median"],
                        "imprint_cycles_min": wl_imprint_stats["min"],
                        "imprint_cycles_max": wl_imprint_stats["max"],
                        "imprint_cycles_stdev": wl_imprint_stats["stdev"],
                        "imprint_cycles_cv_pct": wl_imprint_stats["cv_pct"],
                        "imprint_cycles_ci95_lower": wl_imprint_stats["ci95_lower"],
                        "imprint_cycles_ci95_upper": wl_imprint_stats["ci95_upper"],
                        "imprint_cycles_outlier_count": wl_imprint_stats["outlier_count"],
                        "speedup_mean_x": wl_speedup_stats["mean"],
                        "speedup_median_x": wl_speedup_stats["median"],
                        "speedup_min_x": wl_speedup_stats["min"],
                        "speedup_max_x": wl_speedup_stats["max"],
                        "speedup_stdev_x": wl_speedup_stats["stdev"],
                        "speedup_cv_pct": wl_speedup_stats["cv_pct"],
                        "speedup_ci95_lower_x": wl_speedup_stats["ci95_lower"],
                        "speedup_ci95_upper_x": wl_speedup_stats["ci95_upper"],
                        "speedup_outlier_count": wl_speedup_stats["outlier_count"],
                    })

    out_json = sim / "phase3_benchmark_proof_pack.json"
    out_csv = sim / "phase3_benchmark_proof_pack.csv"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    all_keys = sorted({k for row in report for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(report)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

