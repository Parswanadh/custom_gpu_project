# BitbyBit GPU — System Architecture (Post-Audit v2.0)

## Overview

The BitbyBit GPU is a custom hardware accelerator for transformer-based neural network
inference, implemented in synthesizable Verilog. It targets INT4/INT8/Q8.8 fixed-point
arithmetic with hardware-level zero-skip optimization for sparse activations.

## Architecture Block Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     HOST / SoC Bus (AXI4-Lite)                   │
└───────┬───────────────────┬───────────────────┬──────────────────┘
        │                   │                   │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │ AXI Wt  │        │ Config  │        │  DMA    │
   │ Memory  │        │  Regs   │        │ Engine  │
   │ (parity)│        │         │        │ (AXI M) │
   └────┬────┘        └────┬────┘        └────┬────┘
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────────────────────────────────────────────┐
   │              Command Processor (FIFO)            │
   │  Opcodes: LOAD_WEIGHT, RUN_LAYER, READ_RESULT   │
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────┐
   │              GPU Multi-Core Array                │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
   │  │ gpu_core │ │ gpu_core │ │ gpu_core │ × N    │
   │  │ (LANES)  │ │ (LANES)  │ │ (LANES)  │        │
   │  └──────────┘ └──────────┘ └──────────┘        │
   │  ┌──────────────────────────────────────┐       │
   │  │ Systolic Array (NxN PE mesh)         │       │
   │  │ + Tiled MatMul Controller            │       │
   │  └──────────────────────────────────────┘       │
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────┐
   │              Scratchpad SRAM (Dual-Port)         │
   │         Intermediate activations, KV cache       │
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼──────────────────────────┐
   │           GPT-2 / Transformer Engine             │
   │  Embedding → N × TransformerBlock → LN → Argmax │
   │                                                   │
   │  TransformerBlock:                                │
   │    LN1 → Attention(SRAM Wq/Wk/Wv/Wo, KV cache) │
   │    → Residual → LN2 → FFN(SRAM W1/W2, GELU LUT)│
   │    → Residual                                     │
   └──────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Signed Q8.8 Fixed-Point (Issue #1)
All arithmetic uses `signed` Verilog types. Q8.8 means 8 integer bits + 8 fractional bits
in a 16-bit signed word. Range: [-128.0, +127.996]. Multiply produces Q16.16 (32-bit),
shifted right by 8 to return to Q8.8.

### 2. Zero-Skip Hardware (Core Feature)
The `zero_detect_mult` module checks both operands before multiply. If either is zero,
the multiply is skipped and output is forced to zero. This saves dynamic power and
(with clock gating) reduces switching activity. With ReLU activations, ~92% of FFN
activations are exactly zero.

### 3. SRAM-Based Weight Storage (Issues #7, #8)
All weights are stored in internal SRAM arrays, loaded via a write interface before
inference. This replaces the original flat wire ports that would require millions of
I/O pins for realistic models. Each transformer layer has its own weight bank.

### 4. Pipeline Stall / Backpressure (Issue #4)
All pipeline stages respect `downstream_ready` and expose a `ready` signal. When the
consumer can't accept data, the pipeline stalls. This prevents data loss.

### 5. Parity Error Detection (Issue #16)
All weight memories include a parity bit per word. On read, parity is checked and
errors are flagged via `parity_error` outputs. The AXI weight memory exposes parity
error counts via a status register.

## Module Inventory

### Primitives (`rtl/primitives/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `gpu_core` | N-lane compute core | Parameterized LANES, signed, acc_clear, stall, parity |
| `gpu_multicore` | Multi-core wrapper | NUM_CORES × gpu_core, round-robin/broadcast |
| `gpu_top` | Original FSM pipeline | 4-module chain (sparse→dequant→zero→ALU) |
| `gpu_top_pipelined` | 5-stage pipeline | 1 result/cycle after 4-cycle fill |
| `gpu_top_integrated` | 4-wide pipeline | 4 results/cycle, signed, stall, parity |
| `fused_dequantizer` | INT4→INT8 converter | Signed, sync reset |
| `zero_detect_mult` | Zero-skip multiplier | Signed, single-cycle |
| `variable_precision_alu` | Multi-mode ALU | 4×4bit, 2×8bit, 1×16bit modes, signed |
| `sparse_memory_ctrl` | Direct-mapped sparse mem | O(1) reads, sync reset |

### Compute (`rtl/compute/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `mac_unit` | Multiply-Accumulate | Signed, zero-skip, sign-extended accumulation |
| `systolic_array` | NxN PE mesh | Weight-stationary, skewed input, registered PEs |
| `tiled_matmul` | Tile controller | Breaks large matmul into systolic-sized tiles |
| `softmax_unit` | Softmax via exp LUT | 256-entry exp LUT, reciprocal normalization |
| `gelu_activation` | GELU via LUT | 256-entry LUT, replaces 3-piece approximation |
| `bf16_multiply` | BF16 multiplier | Zero-skip, flush-to-zero, IEEE-like rounding |
| `int4_pack_unit` | INT4 pack/unpack | 4×4bit parallel with zero detection |
| `exp_lut_256` | e^x LUT | 256 entries, Q8.8 input → Q0.8 output |
| `gelu_lut_256` | GELU LUT | 256 entries, signed Q8.8 |
| `inv_sqrt_lut_256` | 1/√x LUT | 256 entries, used in LayerNorm |

### Transformer (`rtl/transformer/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `attention_unit` | Multi-head attention | SRAM weights, KV cache, causal mask, real Q·K^T softmax |
| `ffn_block` | Feed-forward network | SRAM weights, GELU LUT, zero-skip counting |
| `layer_norm` | Layer normalization | inv_sqrt LUT, shift-based division, Q8.8 |
| `linear_layer` | Matrix-vector multiply | SRAM weights, signed, zero-skip in dot product |
| `accelerated_attention` | Attention (flat wires) | KV cache, real attention, for accelerated path |
| `accelerated_linear_layer` | Linear via gpu_core | Uses pipelined gpu_core for MAC |
| `accelerated_transformer_block` | Transformer + gpu_core | FFN via gpu_core pipeline |

### Memory (`rtl/memory/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `axi_weight_memory` | AXI4-Lite weight SRAM | Parity, status registers, GPU read port |
| `scratchpad` | Dual-port SRAM | Port A (compute), Port B (DMA/cmd) |
| `dma_engine` | AXI4 master DMA | Burst reads, sequential writes, interrupt |
| `sparse_memory_ctrl_wide` | 4-wide sparse mem | Prefetch buffer, parity |

### GPT-2 (`rtl/gpt2/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `gpt2_engine` | Full GPT-2 engine | Per-layer SRAM weights, LN banks, zero-skip counting |
| `accelerated_gpt2_engine` | Accel GPT-2 engine | Per-layer LN banks, uses accel transformer block |
| `transformer_block` | Decoder block | SRAM weight pass-through, residual connections |
| `embedding_lookup` | Token+position embed | SRAM tables, combined output |

### Top-Level (`rtl/top/`)
| Module | Description | Key Features |
|--------|-------------|--------------|
| `command_processor` | FIFO command queue | 8 opcodes, DMA trigger, layer sequencing |
| `gpu_config_regs` | AXI4-Lite config | Runtime parameter tuning |
| `perf_counters` | 8 HW counters | Cycles, MACs, zero-skips, stalls, cache hits |
| `reset_synchronizer` | 2-FF reset sync | Safe async→sync reset |

## Fixed-Point Arithmetic Details

```
Q8.8: 16-bit signed, 8 integer + 8 fractional bits
  Value = raw_bits / 256
  1.0  = 256 (0x0100)
  -1.0 = -256 (0xFF00)
  0.5  = 128 (0x0080)

Multiply: Q8.8 × Q8.8 = Q16.16 (32-bit)
  To get Q8.8 result: shift right by 8, or take bits [23:8]
  acc = acc + (a * b) >>> 8

LayerNorm division: DIM must be power of 2
  mean = sum >>> DIM_LOG2  (shift instead of divide)
```

## Performance Characteristics

| Configuration | Products/Cycle | @ 100MHz FPGA | @ 1GHz ASIC |
|--------------|----------------|---------------|-------------|
| 1-core, 4-lane | 4 | 400 MOPS | 4 GOPS |
| 4-core, 32-lane | 128 | 12.8 GOPS | 128 GOPS |
| 4-core, 64-lane | 256 | 25.6 GOPS | 256 GOPS |

Zero-skip boost: +35% effective throughput with ReLU activations (92% sparsity in FFN).
