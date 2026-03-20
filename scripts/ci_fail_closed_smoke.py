#!/usr/bin/env python3
"""
ci_fail_closed_smoke.py - lightweight negative smoke checks for fail-closed CI behavior.

Usage:
  python scripts/ci_fail_closed_smoke.py

This script validates that scripts/run_tests.py fails closed in two controlled paths:
1) Forced compile failure -> non-zero exit, sentinel emitted by compile stub, and "COMPILE FAIL" marker.
2) Forced simulator timeout -> non-zero exit, sentinel emitted by simulator stub, and "TIMEOUT" marker.
3) Explicit invalid env overrides -> deterministic hard-fail markers (no silent PATH fallback).
4) Simulator launch errors -> deterministic non-zero handling.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_tests.py"


def _write_cmd(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def _run_case(
    name: str,
    env_overrides: dict,
    expected_exit: int,
    expected_marker: str,
    expected_sentinel: str = "",
) -> bool:
    env = os.environ.copy()
    env.update(env_overrides)
    completed = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    has_marker = expected_marker in output
    has_sentinel = (not expected_sentinel) or (expected_sentinel in output)
    ok = completed.returncode == expected_exit and has_marker and has_sentinel

    verdict = "PASS" if ok else "FAIL"
    print(f"[{verdict}] {name}")
    print(f"        exit={completed.returncode} expected={expected_exit} marker={expected_marker!r}")
    if expected_sentinel:
        print(f"        sentinel={expected_sentinel!r}")
    if not ok:
        print(
            f"        details: has_marker={has_marker} has_sentinel={has_sentinel}"
        )
        print("-------- runner output (failure) --------")
        print(output.strip())
        print("-----------------------------------------")
    return ok


def main() -> int:
    print("[INFO] Running fail-closed smoke checks for scripts/run_tests.py")
    checks = []

    with tempfile.TemporaryDirectory(prefix="bitbybit_fail_closed_") as temp_dir:
        temp = Path(temp_dir)
        compile_sentinel = "SMOKE_COMPILE_STUB_SENTINEL_FC03_A9E1"
        timeout_sentinel = "SMOKE_TIMEOUT_STUB_SENTINEL_FC03_B4C2"

        iverilog_fail = temp / "iverilog_fail.cmd"
        iverilog_pass = temp / "iverilog_pass.cmd"
        iverilog_not_executable = temp / "iverilog_not_executable.txt"
        vvp_pass = temp / "vvp_pass.cmd"
        vvp_sleep = temp / "vvp_sleep.cmd"
        vvp_not_executable = temp / "vvp_not_executable.txt"
        missing_iverilog = temp / "missing_iverilog.exe"
        missing_vvp = temp / "missing_vvp.exe"

        _write_cmd(
            iverilog_fail,
            f"@echo off\r\necho {compile_sentinel} 1>&2\r\necho smoke compile failure 1>&2\r\nexit /b 1\r\n",
        )
        _write_cmd(iverilog_pass, "@echo off\r\nexit /b 0\r\n")
        _write_cmd(vvp_pass, "@echo off\r\necho smoke sim pass\r\nexit /b 0\r\n")
        _write_cmd(
            vvp_sleep,
            f"@echo off\r\necho {timeout_sentinel}\r\ntimeout /t 5 /nobreak >nul\r\nexit /b 0\r\n",
        )
        iverilog_not_executable.write_text("this is not an executable", encoding="utf-8")
        vvp_not_executable.write_text("this is not an executable", encoding="utf-8")

        common_env = {"BITBYBIT_MAX_TESTS": "1"}

        checks.append(
            _run_case(
                name="forced compile failure returns non-zero",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(iverilog_fail),
                    "BITBYBIT_VVP": str(vvp_pass),
                },
                expected_exit=1,
                expected_marker="COMPILE FAIL",
                expected_sentinel=compile_sentinel,
            )
        )

        checks.append(
            _run_case(
                name="forced timeout returns non-zero",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(iverilog_pass),
                    "BITBYBIT_VVP": str(vvp_sleep),
                    "BITBYBIT_SIM_TIMEOUT_SEC": "1",
                },
                expected_exit=1,
                expected_marker="TIMEOUT",
                expected_sentinel=timeout_sentinel,
            )
        )

        checks.append(
            _run_case(
                name="explicit invalid BITBYBIT_IVERILOG hard-fails",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(missing_iverilog),
                    "BITBYBIT_VVP": str(vvp_pass),
                },
                expected_exit=1,
                expected_marker="BITBYBIT_IVERILOG is explicitly set",
            )
        )

        checks.append(
            _run_case(
                name="explicit invalid BITBYBIT_VVP hard-fails",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(iverilog_pass),
                    "BITBYBIT_VVP": str(missing_vvp),
                },
                expected_exit=1,
                expected_marker="BITBYBIT_VVP is explicitly set",
            )
        )

        checks.append(
            _run_case(
                name="iverilog launch errors return deterministic marker",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(iverilog_not_executable),
                    "BITBYBIT_VVP": str(vvp_pass),
                },
                expected_exit=1,
                expected_marker="COMPILE LAUNCH FAIL",
            )
        )

        checks.append(
            _run_case(
                name="vvp launch errors return deterministic marker",
                env_overrides={
                    **common_env,
                    "BITBYBIT_IVERILOG": str(iverilog_pass),
                    "BITBYBIT_VVP": str(vvp_not_executable),
                },
                expected_exit=1,
                expected_marker="SIM LAUNCH FAIL",
            )
        )

    if all(checks):
        print("[OK] All fail-closed smoke checks passed.")
        return 0

    print("[ERROR] Fail-closed smoke checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
