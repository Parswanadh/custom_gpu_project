# BitbyBit Custom GPU — Project Context

## Overview
Custom GPU designed for efficient transformer/LLM inference, featuring hardware zero-skip
for sparse multiplications and Q8.8 fixed-point arithmetic. The GPU is implemented in
SystemVerilog/Verilog and simulated with Python-based inference engines.

## Architecture
- **Pipeline**: 16-bit datapath, Q8.8 and Q4.12 fixed-point modes
- **Zero-Skip**: `zero_detect_mult` module skips multiplications when either operand is zero
- **Compute Cores**: MAC units, GELU/ReLU activation, softmax, layer normalization
- **Memory**: SRAM controller with sparse memory optimization

## Project Structure
```
rtl/              Verilog/SystemVerilog RTL source
  core/           Core modules (ALU, register file, control)
  compute/        Compute modules (MAC, activation, softmax, layernorm)
  memory/         Memory controller, SRAM, sparse memory
  integration/    Top-level integration, transformer blocks
tb/               Testbenches
sim/              Simulation scripts and cosimulation
scripts/          Python inference engines and utilities
  chat_gpt2.py   GPT-2 chat with optional --relu flag
  chat_opt.py    OPT-125M chat with native ReLU + Q8.8 quantization
weights/          Cached model weights (gitignored)
docs/             Documentation and visualizations
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
