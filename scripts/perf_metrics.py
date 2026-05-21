#!/usr/bin/env python3
"""
BitbyBit — performance metric derivation (single source of truth).

All throughput values MUST satisfy:
    tokens_per_second = clock_hz / cycles_per_token

Simulation-measured constants come from phase3 benchmark proof pack
(full_model_inference_imprint_tb @ 100 MHz).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

# Default: cycle-accurate sim @ 100 MHz (full_model_inference_*_tb)
DEFAULT_CLOCK_HZ = 100_000_000
DEFAULT_BASE_CYCLES_PER_TOKEN = 358
DEFAULT_IMPRINT_CYCLES_PER_TOKEN = 112
DEFAULT_MEDUSA_DRAFT_HEADS = 3

PROOF_PACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "sim",
    "phase3_benchmark_proof_pack.json",
)


def throughput_from_cycles(clock_hz: float, cycles_per_token: float) -> float:
    """Sustained tokens/sec from clock and measured cycles per token."""
    if cycles_per_token <= 0:
        raise ValueError(f"cycles_per_token must be positive, got {cycles_per_token}")
    return clock_hz / cycles_per_token


def cycles_per_token_from_throughput(clock_hz: float, tokens_per_second: float) -> float:
    """Inverse of throughput_from_cycles."""
    if tokens_per_second <= 0:
        raise ValueError(f"tokens_per_second must be positive, got {tokens_per_second}")
    return clock_hz / tokens_per_second


@dataclass(frozen=True)
class SimMeasuredMetrics:
    """RTL simulation metrics (mini full-model path, 100 MHz)."""

    clock_hz: float
    base_cycles_per_token: int
    imprint_cycles_per_token: int
    medusa_draft_heads: int
    source: str

    @property
    def base_throughput_tps(self) -> float:
        return throughput_from_cycles(self.clock_hz, self.base_cycles_per_token)

    @property
    def imprint_throughput_tps(self) -> float:
        return throughput_from_cycles(self.clock_hz, self.imprint_cycles_per_token)

    @property
    def medusa_effective_throughput_tps(self) -> float:
        """Speculative draft heads only — not single-token sustained IPC."""
        return self.imprint_throughput_tps * self.medusa_draft_heads

    @property
    def imprint_speedup_vs_base(self) -> float:
        return self.base_cycles_per_token / self.imprint_cycles_per_token


def _find_proof_entry(pack: list[dict[str, Any]], benchmark: str) -> dict[str, Any] | None:
    for row in pack:
        if row.get("benchmark") == benchmark:
            return row
    return None


def load_sim_metrics(
    proof_pack_path: str | None = None,
    clock_hz: float = DEFAULT_CLOCK_HZ,
) -> SimMeasuredMetrics:
    """Load measured sim cycles from proof pack, with sane defaults."""
    path = proof_pack_path or PROOF_PACK_PATH
    base_cy = DEFAULT_BASE_CYCLES_PER_TOKEN
    imprint_cy = DEFAULT_IMPRINT_CYCLES_PER_TOKEN
    source = "defaults (proof pack missing)"

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
        paired = _find_proof_entry(pack, "base_vs_imprint_full_model")
        if paired:
            base_cy = int(paired["base_cycles"])
            imprint_cy = int(paired["imprint_cycles"])
            source = path
        else:
            imprint_row = _find_proof_entry(pack, "full_model_inference_imprint_tb")
            base_row = _find_proof_entry(pack, "full_model_inference_tb")
            if imprint_row and "total_cycles" in imprint_row:
                imprint_cy = int(imprint_row["total_cycles"])
            if base_row and "total_cycles" in base_row:
                base_cy = int(base_row["total_cycles"])
            source = path

    return SimMeasuredMetrics(
        clock_hz=clock_hz,
        base_cycles_per_token=base_cy,
        imprint_cycles_per_token=imprint_cy,
        medusa_draft_heads=DEFAULT_MEDUSA_DRAFT_HEADS,
        source=source,
    )


@dataclass(frozen=True)
class ArchitecturalMultipliers:
    """
    Analytical throughput multipliers vs dense INT8 MAC @ 1 MAC/cycle.
    Use only for projection after sim or silicon establishes a baseline.
    """

    int4_mac_density: float = 4.0  # 4 INT4 MACs per cycle (documented RTL mode)
    zero_skip_fraction: float = 0.26  # measured OPT-125M FFN ReLU + Q8.8 blend
    variable_precision_gain: float = 1.0  # set >1 when mixed prec path validated

    @property
    def combined_compute_multiplier(self) -> float:
        zero_factor = 1.0 / (1.0 - self.zero_skip_fraction)
        return self.int4_mac_density * zero_factor * self.variable_precision_gain
