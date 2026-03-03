# BitbyBit Custom GPU — Project Context

## Overview
Custom GPU designed for efficient transformer/LLM inference, featuring hardware zero-skip
for sparse multiplications, signed Q8.8 fixed-point arithmetic, and a proper standalone
GPU architecture with command processing, DMA, scratchpad SRAM, and AXI interfaces.

## Architecture (Post-Audit v2.0)
- **Pipeline**: 16-bit signed datapath, Q8.8 fixed-point (signed throughout)
- **Zero-Skip**: `zero_detect_mult` with signed operands skips MACs when either is zero
- **Compute Cores**: Parameterized N-lane gpu_core with pipeline stall/ready handshake
- **Systolic Array**: Proper NxN PE mesh with weight-stationary dataflow and skewed inputs
- **Math LUTs**: 256-entry exp, GELU, inv_sqrt LUTs (replace crude approximations)
- **Memory**: SRAM scratchpad, DMA engine, AXI4-Lite weight memory with parity protection
- **Control**: Command processor with FIFO, config registers, performance counters
- **Precision**: Q8.8 default, BF16 multiplier available, INT4 pack/unpack support
- **Error Detection**: Parity bits on all weight memory, error counters exposed via AXI

## Key Fixes Applied (28 Issues Resolved)
1. **Signed Arithmetic (#1)**: All operands/results `signed` — Q8.8 compatible
2. **Accumulator Clear (#2)**: `acc_clear` input resets accumulator without full reset
3. **inv_sqrt LUT (#3)**: 256-entry LUT replaces 5-bucket step function in LayerNorm
4. **Pipeline Stall (#4)**: `ready/valid` handshaking with `downstream_ready` backpressure
5. **Division Removal (#5)**: Shift-based division for power-of-2 DIM; reciprocal LUT for softmax
6. **Synchronous Reset (#6)**: All `always @(posedge clk or posedge rst)` → `always @(posedge clk)`
7. **SRAM Weights (#7)**: Weight matrices stored in SRAM, loaded via write interface
8. **Per-Layer Weights (#8)**: Each transformer layer has independent weight banks
9. **Real Systolic Array (#9)**: PE mesh with registered inter-PE data flow
10. **Per-Lane Activation (#11)**: `activation_in` is now `8*LANES`-bit vector
11. **Real Attention (#12)**: Q·K^T/√d_k softmax, weighted V sum (not trivial output=V)
12. **GELU LUT (#13)**: 256-entry LUT replaces 3-piece linear approximation
13. **exp LUT (#14)**: 256-entry exp LUT used in softmax
14. **Clock Gate (#15)**: Zero-skipped lanes don't toggle multiplier inputs
15. **Parity (#16)**: Parity bits on all weight SRAM with error flags
16. **DMA Engine (#17)**: AXI4 master for bulk weight transfers
17. **Command Processor (#18)**: FIFO-based command queue with opcodes
18. **Scratchpad (#19)**: Dual-port SRAM for intermediate activations
19. **Config Registers (#20)**: AXI4-Lite slave for runtime GPU configuration
20. **Tiled MatMul (#21)**: Tile controller for large matrices via systolic array
21. **BF16 (#22)**: BF16 multiply unit with zero-skip and flush-to-zero
22. **Perf Counters (#23)**: 8 hardware performance counters (cycles, MACs, stalls, etc.)
23. **Causal Mask (#24)**: Configurable attention mask for autoregressive generation
24. **Reset Sync (#6b)**: 2-FF reset synchronizer for safe async-to-sync reset

## Project Structure
```
rtl/
  primitives/     Core GPU pipeline (gpu_core, gpu_top variants, zero_detect_mult, etc.)
  compute/        Math units (MAC, systolic_array, softmax, GELU, exp LUTs, BF16, tiled_matmul)
  transformer/    Transformer blocks (attention, FFN, layer_norm, linear_layer)
  memory/         Memory (axi_weight_memory, scratchpad, DMA, sparse_memory_ctrl)
  gpt2/           GPT-2 engine (embedding, transformer_block, gpt2_engine)
  top/            Top-level (command_processor, config_regs, perf_counters, reset_sync)
tb/               Testbenches
sim/              Simulation outputs
scripts/          Python cosim and inference
weights/          Cached model weights
docs/             Documentation
```

## Key Scripts

### `scripts/chat_opt.py` — OPT-125M Chat (Primary Demo)
Meta's OPT-125M model with **native ReLU** activation → 92%+ activation sparsity.
Demonstrates the GPU's `zero_detect_mult` hardware feature with real AI inference.

```bash
# Default: FFN ReLU + Q8.8 quantization (best quality + good sparsity)
python scripts/chat_opt.py

# Options
python scripts/chat_opt.py --max-tokens 50 --temperature 0.7
python scripts/chat_opt.py --relu-mode all    # ReLU everywhere (max sparsity, lower quality)
python scripts/chat_opt.py --no-q88           # Disable Q8.8 quantization
```

**Performance (FFN ReLU + Q8.8, default mode):**
- ReLU sparsity: **92%** (exact zeros from ReLU activation)
- Overall zero-skip: **26%** (blended across all matmuls)
- Throughput boost: **1.35x** (vs no skip)
- Q8.8 quantization adds ~5% more zero-skip vs float32

### `scripts/chat_gpt2.py` — GPT-2 Chat (Alternative Demo)
GPT-2-small (124M params) with optional `--relu` flag to swap GELU for ReLU.

```bash
python scripts/chat_gpt2.py              # GELU (2% skip)
python scripts/chat_gpt2.py --relu       # ReLU (23% skip)
```

## Dependencies
- Python 3.10+
- NumPy (inference engine)
- No PyTorch required for inference (pure NumPy)
- Icarus Verilog (for RTL simulation)
- Weights auto-download from HuggingFace on first run

## Zero-Skip Analysis
The overall zero-skip rate is ~26% (not 92%) because only 1 of 6 matmuls per
transformer layer has ReLU-zeroed inputs (the fc2 matmul after FFN activation).
The other 5 matmuls (Q/K/V/out_proj/fc1) receive dense inputs from LayerNorm.

| Source | Skip Rate | Notes |
|--------|-----------|-------|
| FFN ReLU activation | 92% | Exact zeros for negative inputs |
| Q8.8 weight quantization | ~5% | Small weights round to zero |
| Attention projections | ~2% | Dense (LayerNorm input) |
| Overall (blended) | ~26% | Weighted average across all ops |

## Conventions
- RTL files use `snake_case` naming
- Python scripts use `argparse` for CLI
- All inference is pure NumPy — no PyTorch dependency
- Weights are cached in `weights/` directory (gitignored)
- Q8.8 threshold for zero detection: `0.5/256 ≈ 0.00195`
