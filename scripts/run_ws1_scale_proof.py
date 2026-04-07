#!/usr/bin/env python3
"""
run_ws1_scale_proof.py

WS1 deterministic dim-sweep + parity harness orchestrator.

Outputs:
  - sim/dim_sweep_report.json
  - sim/parity_report.json

Exit codes:
  0 => success (and gate passed if enforced)
  1 => gate failure when --enforce-gate is set
  2 => fatal error
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_dims(raw: str) -> list[int]:
    dims: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim <= 0:
            raise ValueError(f"Invalid dim: {dim}")
        if dim not in seen:
            seen.add(dim)
            dims.append(dim)
    if not dims:
        raise ValueError("No dimensions provided")
    return dims


def deterministic_workloads(
    token_id: int,
    position: int,
    token_space: int,
    position_space: int,
    workload_count: int,
    workload_seed: int,
) -> list[dict]:
    if token_space < 1:
        raise ValueError("token_space must be >= 1")
    if position_space < 1:
        raise ValueError("position_space must be >= 1")
    if workload_count < 1:
        raise ValueError("workload_count must be >= 1")

    max_unique = token_space * position_space
    effective_count = min(workload_count, max_unique)

    workloads: list[dict] = []
    seen: set[tuple[int, int]] = set()

    first = (token_id % token_space, position % position_space)
    workloads.append({"token_id": first[0], "position": first[1]})
    seen.add(first)

    if effective_count > 1:
        rng = random.Random(workload_seed)
        attempts = 0
        max_attempts = max(64, max_unique * 8)

        while len(workloads) < effective_count and attempts < max_attempts:
            cand = (rng.randrange(token_space), rng.randrange(position_space))
            if cand not in seen:
                seen.add(cand)
                workloads.append({"token_id": cand[0], "position": cand[1]})
            attempts += 1

        if len(workloads) < effective_count:
            for t in range(token_space):
                if len(workloads) >= effective_count:
                    break
                for p in range(position_space):
                    if len(workloads) >= effective_count:
                        break
                    cand = (t, p)
                    if cand not in seen:
                        seen.add(cand)
                        workloads.append({"token_id": cand[0], "position": cand[1]})

    if len(workloads) < effective_count:
        raise RuntimeError(
            f"Insufficient unique workloads: have {len(workloads)}, requested {effective_count}"
        )

    return workloads


def run_dim_case(
    root: Path,
    run_id: str,
    dim: int,
    args: argparse.Namespace,
    workloads: list[dict],
) -> dict:
    token_seq = ",".join(str(w["token_id"]) for w in workloads)
    position_seq = ",".join(str(w["position"]) for w in workloads)

    report_rel = f"sim/ws1_cosim_dim{dim}_{run_id}.txt"
    json_rel = f"sim/ws1_cosim_dim{dim}_{run_id}.json"

    cmd = [
        sys.executable,
        str(root / "scripts" / "run_scaled_cosim.py"),
        "--dim",
        str(dim),
        "--ffn-mult",
        str(args.ffn_mult),
        "--vocab",
        str(args.token_space),
        "--seq-len",
        str(args.seq_len),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--token-seq",
        token_seq,
        "--position-seq",
        position_seq,
        "--emit-checkpoints",
        "--logit-tolerance",
        str(args.logit_tolerance),
        "--checkpoint-tolerance",
        str(args.checkpoint_tolerance),
        "--json-report",
        json_rel,
        "--report",
        report_rel,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "run_scaled_cosim failed for dim={dim}\nstdout:\n{stdout}\nstderr:\n{stderr}".format(
                dim=dim,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )

    json_path = root / json_rel
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON report for dim={dim}: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    parity = payload.get("parity", {})
    parity_tokens = parity.get("tokens", [])
    pass_count = sum(1 for token in parity_tokens if token.get("pass"))

    return {
        "dim": dim,
        "ffn_dim": dim * int(args.ffn_mult),
        "report_path": report_rel,
        "json_path": json_rel,
        "aggregate": payload.get("aggregate", {}),
        "parity": {
            "overall_pass": bool(parity.get("overall_pass", False)),
            "token_count": len(parity_tokens),
            "token_pass_count": pass_count,
            "token_fail_count": len(parity_tokens) - pass_count,
            "tokens": parity_tokens,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WS1 deterministic dim sweep and parity harness")
    parser.add_argument("--dims", default="16,32,64", help="Comma-separated embedding dimensions")
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=5)
    parser.add_argument("--position", type=int, default=2)
    parser.add_argument("--token-space", type=int, default=16)
    parser.add_argument("--position-space", type=int, default=8)
    parser.add_argument("--workload-count", type=int, default=8)
    parser.add_argument("--workload-seed", type=int, default=20260331)
    parser.add_argument("--logit-tolerance", type=float, default=2.0)
    parser.add_argument("--checkpoint-tolerance", type=float, default=2.0)
    parser.add_argument("--dim-sweep-report", default="sim/dim_sweep_report.json")
    parser.add_argument("--parity-report", default="sim/parity_report.json")
    parser.add_argument("--enforce-gate", action="store_true", help="Exit non-zero if overall parity gate fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    dims = parse_dims(args.dims)
    if args.position_space > args.seq_len:
        raise ValueError("position_space must be <= seq_len for deterministic replay")

    workloads = deterministic_workloads(
        token_id=args.token_id,
        position=args.position,
        token_space=args.token_space,
        position_space=args.position_space,
        workload_count=args.workload_count,
        workload_seed=args.workload_seed,
    )

    if len(workloads) > args.seq_len:
        raise ValueError(
            f"workload_count/effective workloads ({len(workloads)}) exceeds seq_len ({args.seq_len})"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    generated_at = datetime.now(timezone.utc).isoformat()

    print(f"[WS1] run_id={run_id}")
    print(f"[WS1] workloads={len(workloads)} dims={dims}")

    dim_rows: list[dict] = []
    for dim in dims:
        print(f"[WS1] running dim={dim}...")
        row = run_dim_case(root, run_id, dim, args, workloads)
        dim_rows.append(row)

    overall_pass = all(row["parity"]["overall_pass"] for row in dim_rows)

    dim_sweep_payload = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "workload_mode": "matrix",
        "workload_generation": "seeded_random_unique",
        "workload_seed": int(args.workload_seed),
        "workload_count_requested": int(args.workload_count),
        "workload_count_effective": len(workloads),
        "token_space": int(args.token_space),
        "position_space": int(args.position_space),
        "dimensions": dims,
        "workloads": workloads,
        "results": [
            {
                "dim": row["dim"],
                "ffn_dim": row["ffn_dim"],
                "report_path": row["report_path"],
                "json_path": row["json_path"],
                "total_tokens": int(row["aggregate"].get("total_tokens", 0)),
                "total_cycles": int(row["aggregate"].get("total_cycles", 0)),
                "avg_cycles_per_token": float(row["aggregate"].get("avg_cycles_per_token", 0.0)),
                "zero_skip_rate_pct": float(row["aggregate"].get("zero_skip_rate_pct", 0.0)),
                "parity_overall_pass": bool(row["parity"]["overall_pass"]),
            }
            for row in dim_rows
        ],
        "overall_parity_pass": bool(overall_pass),
    }

    parity_payload = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "logit_tolerance": float(args.logit_tolerance),
        "checkpoint_tolerance": float(args.checkpoint_tolerance),
        "workloads": workloads,
        "dimensions": [
            {
                "dim": row["dim"],
                "overall_pass": bool(row["parity"]["overall_pass"]),
                "token_count": int(row["parity"]["token_count"]),
                "token_pass_count": int(row["parity"]["token_pass_count"]),
                "token_fail_count": int(row["parity"]["token_fail_count"]),
                "tokens": row["parity"]["tokens"],
            }
            for row in dim_rows
        ],
        "overall_pass": bool(overall_pass),
    }

    dim_sweep_path = root / args.dim_sweep_report
    parity_path = root / args.parity_report
    dim_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.parent.mkdir(parents=True, exist_ok=True)

    dim_sweep_path.write_text(json.dumps(dim_sweep_payload, indent=2), encoding="utf-8")
    parity_path.write_text(json.dumps(parity_payload, indent=2), encoding="utf-8")

    print(f"[WS1] wrote {dim_sweep_path}")
    print(f"[WS1] wrote {parity_path}")
    print(f"[WS1] overall_parity_pass={overall_pass}")

    if args.enforce_gate and not overall_pass:
        print("[WS1][FAIL-CLOSE] parity gate failed")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - fail-closed entrypoint
        print(f"[WS1][FATAL] {exc}", file=sys.stderr)
        raise SystemExit(2)
