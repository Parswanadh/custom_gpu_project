# Q4 Quantization Formats — Technical Reference for BitbyBit GPU

## Table of Contents
1. [GGML/GGUF Q4_0 Format](#1-ggmlgguf-q4_0-format)
2. [GPTQ Q4 Format](#2-gptq-q4-format)
3. [AWQ Q4 Format](#3-awq-q4-format)
4. [Small Q4 Models for 4×4 Systolic Array](#4-small-q4-models-for-4x4-systolic-array)
5. [INT4 × INT8 MAC Arithmetic](#5-int4--int8-mac-arithmetic)
6. [Mapping to BitbyBit Hardware](#6-mapping-to-bitbybit-hardware)

---

## 1. GGML/GGUF Q4_0 Format

### Binary Layout (block_q4_0)

GGML Q4_0 uses **block quantization** with a block size of **32 weights**.

```c
// From ggml-common.h (llama.cpp source)
typedef struct {
    ggml_half d;        // FP16 scale factor (2 bytes)
    uint8_t   qs[16];   // 32 × 4-bit quantized weights packed into 16 bytes
} block_q4_0;
// Total: 18 bytes per block of 32 weights
// Bits per weight: 18 * 8 / 32 = 4.5 bpw
```

### How Weights Are Packed

Each `uint8_t` in `qs[]` stores **two** 4-bit weights:

```
byte qs[i]:
  bits [3:0] = weight[2*i]      (low nibble)
  bits [7:4] = weight[2*i + 1]  (high nibble)
```

The 4-bit values are **unsigned** integers in range `[0, 15]`.

### Dequantization Formula (Exact)

```
float weight = d * (q - 8)
```

Where:
- `d` = FP16 scale factor (per block of 32 weights)
- `q` = unsigned 4-bit value extracted from qs[] (range 0..15)
- `8` = implicit zero-point (hardcoded, NOT stored in the file)

**There is no stored zero-point/offset.** The zero-point is always 8, making this
an **asymmetric** quantization scheme with a fixed midpoint.

### Concrete Example
```
Given: d = 0.125 (FP16), q = 3
Result: 0.125 * (3 - 8) = 0.125 * (-5) = -0.625

Given: d = 0.125, q = 12
Result: 0.125 * (12 - 8) = 0.125 * 4 = 0.5
```

### Q4_0 Dequantization in C (from llama.cpp)
```c
// From ggml-quants.c
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k) {
    for (int i = 0; i < k/32; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < 16; j++) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;   // low nibble
            const int x1 = (x[i].qs[j] >> 4)    - 8;   // high nibble
            y[i*32 + j*2]     = x0 * d;
            y[i*32 + j*2 + 1] = x1 * d;
        }
    }
}
```

### Q4_1 Variant (for comparison)
```c
typedef struct {
    ggml_half d;        // FP16 scale (2 bytes)
    ggml_half m;        // FP16 minimum (2 bytes) — this IS the zero-point
    uint8_t   qs[16];   // 32 × 4-bit weights (16 bytes)
} block_q4_1;
// Total: 20 bytes per block of 32 weights (5.0 bpw)
// Dequant: weight = d * q + m
// q is unsigned [0,15], no subtraction of 8
```

### GGUF File Structure (for weight loading)
```
GGUF Header:
  magic: "GGUF" (4 bytes)
  version: uint32 (currently 3)
  n_tensors: uint64
  n_kv: uint64 (metadata key-value pairs)

KV Metadata: (architecture, vocab, quantization info)

Tensor Descriptors: (name, shape, type=Q4_0, offset)

Tensor Data: (aligned to 32 bytes, blocks of block_q4_0)
```

---

## 2. GPTQ Q4 Format

### Core Algorithm Difference

GPTQ (Generalized Post-Training Quantization) uses **second-order information**
(Hessian inverse) to choose quantization parameters that minimize output error
layer-by-layer. The quantization is done column-by-column with error compensation.

### Binary Layout

GPTQ stores quantized weights in a **different structure** than GGML:

```python
# Typical GPTQ model files (model.safetensors):
# For each linear layer:
qweight:  int32 tensor, shape [in_features // 8, out_features]
           # 8 × 4-bit weights packed per int32
qzeros:   int32 tensor, shape [in_features // group_size, out_features // 8]
           # Per-group zero-points, also packed 4-bit
scales:   float16 tensor, shape [in_features // group_size, out_features]
           # Per-group scale factors
g_idx:    int32 tensor, shape [in_features]
           # Maps each input channel to its quantization group
           # (Optional — "desc_act=True" reorders columns by activation magnitude)
```

### Weight Packing (qweight)

```
int32 word packs 8 weights:
  bits [3:0]   = weight 0
  bits [7:4]   = weight 1
  bits [11:8]  = weight 2
  bits [15:12] = weight 3
  bits [19:16] = weight 4
  bits [23:20] = weight 5
  bits [27:24] = weight 6
  bits [31:28] = weight 7
```

### Group Size

GPTQ uses **configurable group quantization** (not fixed block of 32):
- **group_size = 128** (most common default)
- group_size = 64 (higher quality, more overhead)
- group_size = 32 (highest quality, most overhead)
- group_size = -1 (per-channel — one scale per entire column)

Each group of `group_size` consecutive input channels shares one scale and one zero-point.

### Dequantization Formula (Exact)

```
float weight = scale[group] * (qweight_int4 - qzeros[group])
```

Where:
- `scale[group]` = FP16 per-group scale factor
- `qweight_int4` = unsigned 4-bit integer [0, 15]
- `qzeros[group]` = per-group zero-point (typically around 8, but varies per group)

### Key Difference from GGML Q4_0

| Property          | GGML Q4_0              | GPTQ Q4                    |
|-------------------|------------------------|-----------------------------|
| Block/Group size  | Fixed 32               | Configurable (typically 128)|
| Zero-point        | Hardcoded = 8          | Stored per group (variable) |
| Scale             | FP16, per block        | FP16, per group             |
| Calibration       | MinMax (simple)        | Hessian-based (2nd order)   |
| Error compensation| None                   | Yes (OBQ-style column-wise) |
| Storage overhead  | 4.5 bpw                | ~4.25 bpw (group_size=128)  |
| Activation-aware  | No                     | Yes (uses calibration data) |

### GPTQ Quantization Process (how scale/zero are computed)
```python
# Simplified GPTQ per-column quantization:
for col in range(out_features):
    for row in range(in_features):
        group = row // group_size
        # Quantize using current scale/zero for this group:
        q = clamp(round(w[row,col] / scale[group] + zero[group]), 0, 15)
        # Compute quantization error:
        err = w[row,col] - scale[group] * (q - zero[group])
        # Propagate error to remaining unquantized columns using Hessian inverse:
        w[row, col+1:] -= err * H_inv[col, col+1:] / H_inv[col, col]
```

This error propagation is what makes GPTQ significantly better than naive MinMax
quantization at the same bit width.

---

## 3. AWQ Q4 Format

### Core Innovation

AWQ (Activation-Aware Weight Quantization) observes that **not all weights are
equally important** — weights connected to channels with large activation magnitudes
matter more. Instead of complex Hessian computation, AWQ:

1. Profiles activation magnitudes on calibration data
2. Identifies "salient" weight channels (top ~1% by activation magnitude)
3. Applies per-channel scaling to protect salient weights before quantization

### The AWQ Scaling Trick

```python
# Before quantization, scale weights to protect important channels:
# s = optimal per-channel scale factor
# For salient channel i:
#   w_scaled[i,:] = w[i,:] * s[i]
#   x_scaled[i]   = x[i] / s[i]     # compensate on activation side

# The math preserves: w * x = (w * s) * (x / s)
# But quantization error is reduced because salient weights are amplified
# before rounding, making them more "integer-friendly"

# Optimal scale search:
# s* = argmin_s || Q(W * diag(s)) * (X / s) - W * X ||
# Solved analytically: s_i ≈ (mean(|x_i|))^alpha, alpha ∈ [0, 1]
# Typical: alpha = 0.5 (geometric mean of no-scaling and full-scaling)
```

### Binary Layout

AWQ typically stores in the same packed format as GPTQ:

```python
# AWQ model files (typically safetensors):
qweight:  int32 tensor — 8 × 4-bit weights per int32 (same packing as GPTQ)
qzeros:   int32 tensor — per-group zero-points (same as GPTQ)
scales:   float16 tensor — per-group scales (same as GPTQ)
# AWQ does NOT store g_idx (no column reordering needed)
```

### Dequantization Formula

**Identical to GPTQ at inference time:**
```
float weight = scale[group] * (qweight_int4 - qzeros[group])
```

The difference is entirely in **how** scale/zero/qweight were computed during
quantization (activation-aware scaling), not in inference-time format.

### Key Differences: AWQ vs GPTQ

| Property              | GPTQ                      | AWQ                        |
|-----------------------|---------------------------|----------------------------|
| Quantization method   | Hessian-based (OBQ)       | Activation-aware scaling   |
| Calibration cost      | High (Hessian inverse)    | Low (just activation stats)|
| Group size            | 128 (typical)             | 128 (typical)              |
| Inference format      | int4 + scale + zero       | int4 + scale + zero        |
| Kernel compatibility  | Needs custom CUDA kernel  | Same kernels as GPTQ       |
| Quantization speed    | Slow (hours for 70B)      | Fast (minutes for 70B)     |
| Quality at 4-bit      | Very good                 | Slightly better on average |
| desc_act reordering   | Optional (g_idx)          | Not needed                 |

**Bottom line for hardware:** AWQ and GPTQ are **identical at the inference/hardware
level**. The difference is purely in the offline quantization process.

---

## 4. Small Q4 Models for 4×4 Systolic Array

### What's Available on HuggingFace in GGUF Q4_0

#### Tiny Models (< 200M params, ideal for simulation)

| Model | Params | Q4_0 GGUF Size | HuggingFace Path |
|-------|--------|-----------------|-------------------|
| **SmolLM-135M** | 135M | ~80 MB | `afrideva/smollm-135M-GGUF` or TheBloke variants |
| **TinyLlama-1.1B** | 1.1B | ~600 MB | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |
| **GPT-2 Small** | 124M | ~73 MB | `ggml-org/gpt-2` (official GGUF) |
| **Phi-1** | 1.3B | ~750 MB | Various uploaders |
| **TinyStories-33M** | 33M | ~20 MB | `afrideva/TinyStories-33M-GGUF` |
| **TinyStories-1M** | 1M | ~1 MB | Community uploads (check `tinystories` tag) |
| **Qwen2-0.5B** | 0.5B | ~290 MB | `Qwen/Qwen2-0.5B-GGUF` |
| **pythia-70m** | 70M | ~40 MB | `afrideva/pythia-70m-GGUF` |
| **pythia-160m** | 160M | ~95 MB | `afrideva/pythia-160m-GGUF` |

#### Recommendations for 4×4 Systolic Array Simulation

**Best picks (in order):**

1. **TinyStories-33M** (~20 MB Q4_0)
   - 4 layers, dim=256, 4 heads
   - Weight matrices: 256×256 = 65,536 weights per matrix
   - Tiles perfectly: 256/4 = 64 tiles per dimension
   - Small enough for full RTL simulation

2. **pythia-70m** (~40 MB Q4_0)
   - 6 layers, dim=512, 8 heads
   - Weight matrices: 512×512, 512×2048
   - Still manageable: 512/4 = 128 tiles per dimension

3. **GPT-2 Small** (124M, ~73 MB Q4_0)
   - 12 layers, dim=768, 12 heads
   - Weight matrices: 768×768, 768×3072
   - 768/4 = 192 tiles — doable but slow in simulation

4. **SmolLM-135M** (~80 MB Q4_0)
   - Modern architecture, good quality for size
   - Similar dimensions to GPT-2

### Extracting Weights from GGUF for Simulation

```python
# Using gguf library (pip install gguf)
from gguf import GGUFReader

reader = GGUFReader("model-q4_0.gguf")
for tensor in reader.tensors:
    name = tensor.name           # e.g., "blk.0.attn_q.weight"
    shape = tensor.shape         # e.g., (768, 768)
    qtype = tensor.tensor_type   # Q4_0 = 2
    data = tensor.data           # Raw bytes (block_q4_0 structures)
    
    # Manual Q4_0 decode:
    import numpy as np
    import struct
    
    num_blocks = np.prod(shape) // 32
    weights = np.zeros(np.prod(shape), dtype=np.float32)
    
    for b in range(num_blocks):
        offset = b * 18  # 18 bytes per block
        d = struct.unpack('<e', data[offset:offset+2])[0]  # FP16 scale
        for j in range(16):
            byte = data[offset + 2 + j]
            q0 = (byte & 0x0F) - 8  # low nibble
            q1 = (byte >> 4) - 8     # high nibble
            weights[b*32 + j*2]     = d * q0
            weights[b*32 + j*2 + 1] = d * q1
    
    weights = weights.reshape(shape)
```

### Creating Your Own Q4_0 Weights for Simulation

For RTL simulation, you might want to quantize directly:

```python
import numpy as np

def quantize_q4_0(weights_fp32, block_size=32):
    """Quantize FP32 weights to Q4_0 format.
    Returns: list of (scale_fp16, int4_values[32]) per block.
    """
    flat = weights_fp32.flatten()
    n_blocks = len(flat) // block_size
    blocks = []
    
    for i in range(n_blocks):
        block = flat[i*block_size : (i+1)*block_size]
        
        # Scale = max absolute value / 8
        amax = np.max(np.abs(block))
        d = amax / 8.0  # Scale factor
        
        if d == 0:
            q = np.zeros(block_size, dtype=np.uint8) + 8
        else:
            # Quantize: q = round(x / d) + 8, clamped to [0, 15]
            q = np.clip(np.round(block / d) + 8, 0, 15).astype(np.uint8)
        
        blocks.append((np.float16(d), q))
    
    return blocks

def dequantize_q4_0(blocks, total_elements):
    """Exact inverse of quantize_q4_0."""
    result = np.zeros(total_elements, dtype=np.float32)
    for i, (d, q) in enumerate(blocks):
        for j in range(32):
            result[i*32 + j] = float(d) * (int(q[j]) - 8)
    return result
```

---

## 5. INT4 × INT8 MAC Arithmetic

### Bit Width Analysis

For a single multiply:
```
INT4 weight:  signed [-8, +7]       — 4 bits
INT8 activation: signed [-128, +127] — 8 bits
Product: signed [-1024, +1016]       — needs 12 bits (4+8)
   Exact: -8 × -128 = +1024, +7 × -128 = -896
   Max positive: 7 × 127 = 889
   Max negative: 7 × (-128) = -896 or -8 × 127 = -1016
   Most extreme: -8 × -128 = +1024 → needs 12 bits (sign + 11 magnitude)
```

### Accumulator Width for Dot Product

For accumulating `K` products (dot product of length K):

```
Single product range: [-1024, +1024]  (12 bits)
Sum of K products:    [-1024*K, +1024*K]
Accumulator bits:     12 + ceil(log2(K))
```

| Dot product length K | Accumulator bits needed |
|----------------------|------------------------|
| 4 (4×4 tile)        | 12 + 2 = 14 bits       |
| 32 (Q4_0 block)     | 12 + 5 = 17 bits       |
| 64                   | 12 + 6 = 18 bits       |
| 128 (GPTQ group)    | 12 + 7 = 19 bits       |
| 256 (typical dim)    | 12 + 8 = 20 bits       |
| 512                  | 12 + 9 = 21 bits       |
| 768 (GPT-2 dim)     | 12 + 10 = 22 bits      |
| 3072 (GPT-2 FFN)    | 12 + 12 = 24 bits      |

**Practical recommendation: 24-bit accumulator** handles K up to 4096 without
overflow. A 32-bit accumulator gives maximum safety margin for any realistic model.

### For BitbyBit's 4×4 Systolic Array (Current: 32-bit ACC_WIDTH)

Your current systolic array has `ACC_WIDTH = 32`, which is **more than sufficient**
for INT4 × INT8 MAC. Even accumulating 768 products needs only 22 bits.

However, your current `DATA_WIDTH = 16` is used for both weights and activations.
For INT4 × INT8, you'd want a **mixed-precision** PE:

```
Weight:     4-bit signed  [-8, +7]
Activation: 8-bit signed  [-128, +127]
Product:    12-bit signed
Accumulator: 32-bit signed (your current ACC_WIDTH — perfect)
```

### The Dequantize-Then-Multiply vs Multiply-Then-Scale Approaches

**Approach A: Dequantize first (what your fused_dequantizer does)**
```
int8_activation = dequantize(int4_weight)     // expand INT4 → INT8
result = int8_activation × int8_input         // 8×8 multiply
```
Pro: Uses existing INT8 datapath
Con: Loses precision during dequant step, doesn't save area/power

**Approach B: Integer multiply, then scale (optimal for hardware)**
```
// In the PE, compute entirely in integers:
int12_product = int4_weight × int8_activation   // 4×8 = 12-bit
int32_accumulator += int12_product              // accumulate in integer domain

// AFTER full dot product, apply scale ONCE:
float_result = int32_accumulator × fp16_scale   // one FP multiply per output
```
Pro: Minimal hardware (4×8 multiplier is ~4× smaller than 8×8)
Pro: Scale applied once per block/group, not per element
Pro: Exact integer arithmetic — no rounding until final conversion
**This is how all efficient INT4 inference engines work.**

### Accuracy vs FP16 — Measured Results

From published benchmarks (llama.cpp perplexity measurements on WikiText-2):

| Model | FP16 PPL | Q4_0 PPL | Δ PPL | % Degradation |
|-------|----------|----------|-------|---------------|
| LLaMA-7B | 5.68 | 6.16 | +0.48 | +8.5% |
| LLaMA-13B | 5.09 | 5.36 | +0.27 | +5.3% |
| LLaMA-30B | 4.10 | 4.34 | +0.24 | +5.9% |
| LLaMA-65B | 3.53 | 3.69 | +0.16 | +4.5% |
| GPT-2 Small (124M) | ~29.0 | ~33-35 | +4-6 | +14-20% |

**Key observations:**
- Larger models tolerate Q4 much better (65B loses only 4.5%)
- Small models (124M) have significant degradation (15-20%)
- GPTQ Q4 is typically 1-2 PPL points better than Q4_0 at same bit width
- AWQ Q4 is typically 0.5-1 PPL better than GPTQ Q4

### INT4 × INT8 vs INT8 × INT8 Hardware Cost

```
Multiplier area comparison (approximate):
  4×4 unsigned: ~16 full adders
  4×8 signed:   ~32 full adders + sign handling
  8×8 signed:   ~64 full adders + sign handling
  16×16 signed: ~256 full adders + sign handling

INT4×INT8 multiplier is roughly 2× SMALLER than INT8×INT8
```

---

## 6. Mapping to BitbyBit Hardware

### Current State Assessment

Your hardware already has the building blocks:

| Component | Status | For Q4_0 |
|-----------|--------|----------|
| `int4_pack_unit` | ✅ Has pack/unpack + zero_mask | Perfect for Q4 weight storage |
| `fused_dequantizer` | ⚠️ Dequant to INT8, then multiply | Suboptimal — should do INT4×INT8 directly |
| `variable_precision_alu` | ✅ Mode 00: 4×4 multiplies | But does INT4×INT4, not INT4×INT8 |
| `systolic_array` | ✅ 4×4, ACC_WIDTH=32 | DATA_WIDTH=16 — needs mixed-precision mode |
| `tiled_matmul` | ✅ ARRAY_SIZE=4 tiling | Works with any precision, just change tile load |

### Recommended Architecture for Q4_0 Inference

```
┌─────────────────────────────────────────────────────────┐
│  GGUF Q4_0 Weight Memory (18 bytes per block of 32)     │
│  ┌──────────┬──────────────────────────────────────────┐ │
│  │ scale(d) │ packed_qs[16 bytes] = 32 × INT4 weights  │ │
│  │ FP16     │ {w1,w0}{w3,w2}...{w31,w30}               │ │
│  └──────────┴──────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────┐
│  INT4 Unpack (int4_pack_unit)   │
│  Extract 4-bit weights          │
│  Subtract 8 (hardcoded offset)  │  ← new: subtract constant 8
│  Output: signed 4-bit [-8,+7]   │
└─────────────────┬───────────────┘
                  │ INT4 weights
                  ▼
┌─────────────────────────────────────────────────────────┐
│  4×4 Systolic Array (mixed precision INT4w × INT8act)   │
│                                                         │
│  Each PE:                                               │
│    weight_reg: 4-bit signed (loaded from Q4_0 block)    │
│    act_in:     8-bit signed (from previous layer/embed) │
│    product:    12-bit signed = weight × activation      │
│    psum:       32-bit signed accumulator                 │
│                                                         │
│  Zero-skip: skip if weight==0 OR activation==0          │
└─────────────────┬───────────────────────────────────────┘
                  │ 32-bit integer accumulator (per output)
                  ▼
┌─────────────────────────────────┐
│  Scale Application (one-shot)   │
│  fp16_result = acc × scale(d)   │  ← one FP16 multiply per block
│  OR: q8.8_result = (acc * d)    │     (can use shift if d is power-of-2)
│  >> appropriate shift            │
└─────────────────────────────────┘
```

### Q4_0 Block Walk for Tiled MatMul

For a weight matrix of shape [K, N] stored in Q4_0:
- There are `(K * N) / 32` blocks
- Blocks are stored in **row-major** order along the flattened tensor
- Each block of 32 weights is contiguous in the K (input) dimension

```
Processing a tile (4×4 output tile, K=256 reduction):
  For k_block = 0 to 255 in steps of 32:  // iterate Q4_0 blocks
    Load Q4_0 block → get scale_d + 32 INT4 weights
    For k_sub = 0 to 31 in steps of 4:    // feed 4 at a time to 4×4 array
      Load 4 weights into systolic array column
      Feed 4 activations (INT8) into systolic array row
      Accumulate 4×4 partial products
    After 32 weights: acc[] now has partial sums for this block
  After all blocks: apply scale_d to get final output

  Total MACs per output element: 256
  Accumulator range: [-1024 * 256, +1024 * 256] = [-262144, +262144] → 19 bits
  Your 32-bit accumulator: ample headroom ✓
```

### Memory Bandwidth for TinyStories-33M (Recommended Test Model)

```
Model: TinyStories-33M
  Layers: 4, dim: 256, heads: 4, head_dim: 64, FFN dim: 1024

Per transformer layer:
  Q weight: 256×256 = 65,536 params → 65536/32 × 18 = 36,864 bytes (36 KB)
  K weight: 256×256 = 36 KB
  V weight: 256×256 = 36 KB
  O weight: 256×256 = 36 KB
  FFN up:   256×1024 = 262,144 params → 147,456 bytes (144 KB)
  FFN down: 1024×256 = 144 KB
  Total per layer: 432 KB

4 layers total: ~1.7 MB of Q4_0 weights
Embedding: 50257 × 256 = 12.9M params → ~7.3 MB
Total model: ~9 MB in Q4_0

For 4×4 systolic array simulation:
  256×256 matmul: 64×64 = 4096 tiles × 64 MAC cycles each = 262,144 cycles
  At 100MHz: ~2.6 ms per matmul (very manageable for RTL sim)
```

---

## Appendix A: Format Comparison Cheat Sheet

```
┌──────────────┬────────────┬──────────────┬───────────────┐
│              │ GGML Q4_0  │ GPTQ Q4      │ AWQ Q4        │
├──────────────┼────────────┼──────────────┼───────────────┤
│ Block/Group  │ 32 fixed   │ 128 (config) │ 128 (config)  │
│ Scale        │ FP16/block │ FP16/group   │ FP16/group    │
│ Zero-point   │ 8 (fixed)  │ INT4/group   │ INT4/group    │
│ Weight range │ [0,15]     │ [0,15]       │ [0,15]        │
│ Dequant      │ d*(q-8)    │ s*(q-z)      │ s*(q-z)       │
│ BPW          │ 4.5        │ ~4.25        │ ~4.25         │
│ Calibration  │ MinMax     │ Hessian      │ Act. magnitude│
│ Error comp.  │ None       │ Column-wise  │ Pre-scaling   │
│ File format  │ GGUF       │ safetensors  │ safetensors   │
│ HW inference │ INT4→deq   │ INT4→deq     │ INT4→deq      │
│ Same HW?     │ YES — all three use the same INT4×INT8 MAC│
└──────────────┴────────────┴──────────────┴───────────────┘
```

## Appendix B: For Your fused_dequantizer — Suggested Fix

Your current dequantizer uses 4-bit scale and 4-bit offset, which limits precision.
For proper Q4_0 support, the scale should be FP16 (or at minimum 8-bit fixed-point):

```
Current:   int8_out = (int4_in - offset_4bit) * scale_4bit   // very coarse
Q4_0:      float_out = fp16_scale * (uint4_in - 8)           // standard
Proposed:  int8_out = (int4_in - 4'd8) * scale_8bit >> shift  // hardware-friendly
```

A better hardware approach for Q4_0 (avoids FP16 entirely):

```verilog
// Q4_0 dequant in pure integer arithmetic:
// scale_q88 = Q8.8 representation of the FP16 scale factor
// shifted = int4_weight - 4'sd8  (subtract fixed zero-point)
// result_q88 = shifted * scale_q88  (5-bit × 16-bit = 21-bit product)
// Feed result_q88 directly into INT8 or Q8.8 datapath

wire signed [4:0] shifted = $signed({1'b0, int4_in}) - 5'sd8;
wire signed [20:0] product = shifted * $signed(scale_q88);  // scale_q88 is 16-bit Q8.8
wire signed [15:0] result_q88 = product[20:5];  // Truncate to Q8.8
```
