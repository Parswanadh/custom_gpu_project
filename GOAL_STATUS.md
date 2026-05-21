# Goal Status — BitbyBit Custom GPU

**Target:** Variable-precision ALU + zero-skip + multi× NVIDIA-class throughput, **FPGA dump ready**.

## Core features (iverilog verified)

| Feature | RTL | Regression | Status |
|---------|-----|------------|--------|
| Variable-precision ALU | `rtl/primitives/variable_precision_alu.v` | P1 **9/9 PASS** | Done |
| Zero-skip multiply | `rtl/primitives/zero_detect_mult.v` | P1 **7/7 PASS** | Done |
| Zero-skip MAC | `rtl/compute/mac_unit.v` | P2 **8/8 PASS** | Done |
| GPU pipeline (primitives) | `rtl/primitives/gpu_top.v` | P1 gpu_top | Fixed `ready_in` on ZDM |
| GPT-2 in silicon | `rtl/gpt2/gpt2_engine.v` | P4 **1/1**, P17 nanogpt smoke | Done |

## Performance claims (honest, measurable)

See `docs/PERFORMANCE_VALIDATION.md` and `docs/PERFORMANCE_CLAIMS_REPORT.md`.

| Claim | Basis |
|-------|--------|
| **~3.2×** imprint vs base path | Sim: 358 → 112 cycles/token @ 100 MHz |
| **892K tok/s** sustained imprint | `100e6 / 112` |
| **4× INT4 MAC density** | Architectural vs FP16 tensor cores |
| **~1.35×** from ~26% zero-skip | Python cosim (OPT-125M) |
| **vs NVIDIA GPU** | Requires `scripts/gemma3_benchmark_nvidia.py` + FPGA wall-clock — not sim-only |

Run: `python scripts/compare_bitbybit_vs_rtx.py --medusa`

## FPGA dump (Kintex-7 / Genesys 2)

| Item | Path |
|------|------|
| Portable RTL (27 files) | `fpga_dump/rtl/` |
| Constraints | `fpga_dump/constraints/kintex7_genesys2.xdc` |
| Vivado Tcl | `fpga_dump/scripts/create_vivado_project.tcl`, `build_bitstream.tcl` |
| Top module | `gpu_system_top_v2` |
| Sync script | `scripts/export_fpga_dump.ps1` |

```powershell
cd custom_gpu_project
powershell -ExecutionPolicy Bypass -File .\scripts\export_fpga_dump.ps1
cd fpga_dump
vivado -mode batch -source scripts\create_vivado_project.tcl
```

Copy **`fpga_dump/`** to your Kintex workflow folder as-is.

## Full regression

```powershell
cd custom_gpu_project
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1
```

## GitHub

- https://github.com/Parswanadh/custom_gpu_project
- https://github.com/Parswanadh/BitbyBit (parent docs + verification)
