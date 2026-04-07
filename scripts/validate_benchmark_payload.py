#!/usr/bin/env python3
"""
validate_benchmark_payload.py

Fail-closed validator for benchmark payloads and proof-pack consistency.

Usage:
  python scripts/validate_benchmark_payload.py \
    --input sim/compare_summary_latest.json \
    --schema sim/benchmark_schema.json \
    --proof-pack sim/phase3_benchmark_proof_pack.json

Exit codes:
  0 => validation passed
  1 => validation failed
  2 => fatal/unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_phase3_benchmark_proof_pack import validate_compare_summary


TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "object": dict,
    "array": list,
    "boolean": bool,
    "null": type(None),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _check_type(value, type_decl) -> bool:
    if isinstance(type_decl, list):
        return any(_check_type(value, item) for item in type_decl)
    py_type = TYPE_MAP.get(type_decl)
    if py_type is None:
        return True
    if type_decl == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, py_type)


def _validate_required(data: dict, required: list[str], context: str) -> None:
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{context} missing required keys: {missing}")


def _validate_top_types(data: dict, schema: dict) -> None:
    properties = schema.get("properties", {})
    for key, rule in properties.items():
        if key not in data:
            continue
        type_decl = rule.get("type")
        if type_decl is None:
            continue
        if not _check_type(data[key], type_decl):
            raise ValueError(f"Top-level key '{key}' has invalid type: expected {type_decl}, got {type(data[key]).__name__}")


def _validate_nested_required(data: dict, schema: dict) -> None:
    properties = schema.get("properties", {})
    for section in ("run_quality", "system_environment", "file_integrity"):
        if section not in properties or section not in data:
            continue
        section_schema = properties[section]
        required = section_schema.get("required", [])
        if isinstance(data[section], dict):
            _validate_required(data[section], required, section)

    samples_schema = properties.get("samples", {}).get("items", {})
    sample_required = samples_schema.get("required", [])
    for idx, sample in enumerate(data.get("samples", [])):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{idx}] must be an object")
        _validate_required(sample, sample_required, f"samples[{idx}]")


def validate_against_schema(data: dict, schema: dict) -> None:
    required = schema.get("required", [])
    _validate_required(data, required, "root")
    _validate_top_types(data, schema)
    _validate_nested_required(data, schema)


def validate_proof_pack_consistency(compare_summary: dict, proof_pack_path: Path) -> None:
    proof_rows = read_json(proof_pack_path)
    if not isinstance(proof_rows, list):
        raise ValueError(f"Proof-pack must be a JSON array: {proof_pack_path}")

    run_id = compare_summary.get("run_id")
    if not run_id:
        raise ValueError("compare summary missing run_id")

    run_rows = [row for row in proof_rows if isinstance(row, dict) and row.get("run_id") == run_id]
    if not run_rows:
        raise ValueError(f"No proof-pack rows found for run_id={run_id}")

    paired_rows = [
        row
        for row in run_rows
        if str(row.get("benchmark", "")).startswith("base_vs_imprint_full_model_paired")
    ]
    if not paired_rows:
        raise ValueError(
            f"Proof-pack missing paired benchmark row for run_id={run_id}; expected benchmark starting with 'base_vs_imprint_full_model_paired'"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark payload schema and proof-pack consistency.")
    parser.add_argument("--input", required=True, help="Path to compare summary JSON")
    parser.add_argument("--schema", required=True, help="Path to benchmark schema JSON")
    parser.add_argument("--proof-pack", help="Optional path to proof-pack JSON for run_id consistency check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compare_path = Path(args.input)
    schema_path = Path(args.schema)

    compare_summary = read_json(compare_path)
    schema = read_json(schema_path)

    # Schema-level structural checks.
    validate_against_schema(compare_summary, schema)

    # Deep semantic checks from canonical validator already used in proof-pack generation.
    validate_compare_summary(compare_summary)

    # Optional proof-pack consistency check.
    if args.proof_pack:
        validate_proof_pack_consistency(compare_summary, Path(args.proof_pack))

    print("[OK] benchmark payload validation passed")
    print(f"[OK] run_id={compare_summary.get('run_id')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAIL-CLOSE] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"[FATAL] Unexpected validator error: {exc}", file=sys.stderr)
        raise SystemExit(2)