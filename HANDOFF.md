# BitbyBit Custom GPU — Complete Handoff Document for Continuation

> **Purpose**: This document gives a new Claude Opus instance 100% of the context needed to resume work on this project. Read this FULLY before taking any action.
> **Date**: February 28, 2026
> **Workspace**: `D:\Projects\BitbyBit\custom_gpu_project`
> **OS**: Windows, Shell: PowerShell 5.1

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Complete File Structure](#2-complete-file-structure)
3. [Architecture Summary](#3-architecture-summary)
4. [Current Test Results (After Fixes)](#4-current-test-results)
5. [What Was Already Fixed](#5-what-was-already-fixed)
6. [Remaining Bugs To Fix](#6-remaining-bugs-to-fix)
7. [Critical Improvements Needed (GPU Reliability)](#7-critical-improvements-needed)
8. [Simulation & Tooling](#8-simulation--tooling)
9. [GPT-2 Model Execution](#9-gpt-2-model-execution)
10. [Visualization Requirements](#10-visualization-requirements)
11. [Exact Commands Reference](#11-exact-commands-reference)
12. [Key File Contents Summary](#12-key-file-contents-summary)
13. [Action Plan (What To Do Next)](#13-action-plan)

---

## 1. PROJECT OVERVIEW

A **fully custom GPU accelerator** designed from scratch in Verilog for transformer/LLM inference (GPT-2, OPT-125M). Features:

- **Q8.8 signed fixed-point** arithmetic (16-bit: 8 integer + 8 fractional bits)
- **Zero-skip hardware** (`zero_detect_mult`) — skips multiplications when either operand is zero
- **KV-cached attention** for autoregressive token generation
- **5-stage pipelined compute core** with backpressure/stall support
- **AXI4-Lite bus interface** for SoC integration
- **DMA engine**, **command processor**, **scratchpad SRAM**, **perf counters**
- **Parameterized** — works at EMBED_DIM=4 (test) up to 768 (real GPT-2)

**Simulation only** — uses Icarus Verilog (iverilog 12.0). No FPGA synthesis done yet.

---

## 2. COMPLETE FILE STRUCTURE

```
D:\Projects\BitbyBit\custom_gpu_project\
|
|-- CLAUDE.md                          # Project context (read this too)
|-- HANDOFF.md                         # THIS FILE
|-- cosim_output.txt                   # Previous cosim run output (dim=4)
|-- cosim_report.txt                   # Sentence cosim report (dim=4, "hello")
|-- cosim_report_dim64.txt             # Scaled cosim report (dim=64, "hello")
|-- cosim.vcd                          # VCD waveform from cosim
|
|-- rtl/
|   |-- primitives/                    # Layer 1: Core compute building blocks
|   |   |-- zero_detect_mult.v         # Signed 8-bit multiplier with zero-skip bypass
|   |   |-- variable_precision_alu.v   # 4-bit/8-bit/16-bit parallel signed ALU
|   |   |-- sparse_memory_ctrl.v       # Direct-mapped sparse memory controller
|   |   |-- fused_dequantizer.v        # INT4 -> signed INT8 with scale/offset
|   |   |-- gpu_core.v                 # N-lane pipelined compute core (285 lines, KEY MODULE)
|   |   |-- gpu_multicore.v            # Multi-core wrapper (NUM_CORES x gpu_core)
|   |   |-- gpu_top.v                  # Original FSM pipeline (4-module chain)
|   |   |-- gpu_top_pipelined.v        # 5-stage pipeline version
|   |   |-- gpu_top_integrated.v       # 4-wide pipeline version
|   |
|   |-- compute/                       # Layer 2: Neural-net compute units
|   |   |-- mac_unit.v                 # Multiply-accumulate with acc_clear
|   |   |-- systolic_array.v           # NxN PE mesh, weight-stationary, zero-skip (193 lines)
|   |   |-- gelu_activation.v          # GELU via gelu_lut_256
|   |   |-- gelu_lut_256.v             # 256-entry GELU lookup table
|   |   |-- softmax_unit.v             # Softmax: max-subtract + exp LUT + reciprocal normalize
|   |   |-- exp_lut_256.v              # 256-entry exp() LUT: round(255 * exp(-k/64))
|   |   |-- inv_sqrt_lut_256.v         # 256-entry 1/sqrt(x) LUT for LayerNorm
|   |   |-- bf16_multiply.v            # BF16 multiplier (available but not primary path)
|   |   |-- int4_pack_unit.v           # INT4 pack/unpack
|   |   |-- tiled_matmul.v             # Tile controller for large matrices via systolic array
|   |
|   |-- transformer/                   # Layer 3: Transformer building blocks
|   |   |-- layer_norm.v               # LayerNorm with inv_sqrt_lut_256 (gamma/beta)
|   |   |-- linear_layer.v             # Dense y = Wx + b with weight loading
|   |   |-- attention_unit.v           # Multi-head attention with KV cache + exp LUT
|   |   |-- ffn_block.v                # FFN: Linear -> GELU -> Linear
|   |   |-- accelerated_attention.v    # KV-cached attention for accelerated engine
|   |   |-- accelerated_linear_layer.v # Linear layer for accelerated engine
|   |   |-- accelerated_transformer_block.v  # Full transformer block (accelerated)
|   |
|   |-- gpt2/                          # Layer 4: Full inference engine
|   |   |-- embedding_lookup.v         # Token + position embedding (ROM lookup + add)
|   |   |-- transformer_block.v        # Full decoder block (LN->Attn->Res->LN->FFN->Res)
|   |   |-- gpt2_engine.v              # Original GPT-2 engine (non-accelerated)
|   |   |-- accelerated_gpt2_engine.v  # Accelerated GPT-2 with zero-skip + KV cache
|   |
|   |-- memory/                        # Memory subsystem
|   |   |-- axi_weight_memory.v        # AXI4-Lite slave weight SRAM with parity
|   |   |-- dma_engine.v               # AXI4 master DMA for bulk weight transfers
|   |   |-- scratchpad.v               # Dual-port SRAM for intermediate activations
|   |   |-- sparse_memory_ctrl_wide.v  # Wide sparse memory controller
|   |
|   |-- top/                           # Top-level control
|       |-- command_processor.v         # FIFO-based command queue (8 opcodes)
|       |-- gpu_config_regs.v           # AXI4-Lite config registers
|       |-- perf_counters.v             # 8 hardware performance counters (PMU)
|       |-- reset_synchronizer.v        # 2-FF async-to-sync reset
|
|-- tb/                                # Testbenches (mirror RTL structure)
|   |-- primitives/                    # TBs for all primitives (all working)
|   |-- compute/                       # TBs for compute modules
|   |-- transformer/                   # TBs for transformer blocks
|   |-- gpt2/                          # TBs for full pipeline
|   |   |-- accelerated_gpt2_engine_tb.v  # THE key test (3-token autoregressive)
|   |   |-- gpt2_engine_tb.v           # Original engine test
|   |   |-- embedding_lookup_tb.v
|   |-- memory/
|   |   |-- axi_weight_memory_tb.v
|   |-- demo/
|   |   |-- feature_verification_tb.v
|   |   |-- gpt2_demo_tb.v
|   |-- cocotb/                        # Cocotb-based tests (Python-Verilog cosim)
|   |-- top/                           # EMPTY — no testbenches for top/ modules!
|
|-- scripts/
|   |-- run_all_tests.ps1              # Master test runner (PowerShell)
|   |-- run_cosim.py                   # Python-Verilog cosimulation (dim=4)
|   |-- run_scaled_cosim.py            # Scaled cosim (dim=64, uses $readmemh)
|   |-- run_sentence_cosim.py          # Sentence-level cosim
|   |-- chat_gpt2.py                   # Interactive GPT-2 chat (pure NumPy, 124M params)
|   |-- chat_opt.py                    # Interactive OPT-125M chat (native ReLU)
|   |-- extract_gpt2_weights.py        # Downloads & extracts GPT-2 weights from HuggingFace
|   |-- generate_weights.py            # Generate test weight files
|   |-- benchmark_throughput.py         # Throughput benchmarking
|   |-- compare_inference.py           # Compare GPU vs CPU inference
|   |-- test_quality.py               # Quality testing
|   |-- test_zeroskip.py              # Zero-skip rate analysis
|   |-- run_demo.ps1                   # Demo runner
|   |-- run_cocotb.ps1                 # Cocotb test runner
|
|-- sim/                               # Compiled simulation binaries + outputs
|   |-- waveforms/                     # VCD waveform output directory
|
|-- weights/                           # Model weight caches
|   |-- cache/                         # Downloaded model caches
|   |-- gpt2_dim64/                    # Extracted dim=64 weights
|   |-- gpt2_real/                     # Real GPT-2 weights
|   |-- opt125m/                       # OPT-125M weights
|
|-- website/                           # Project showcase website
|   |-- index.html
|   |-- css/, js/, assets/
|
|-- docs/
    |-- architecture.md                # Architecture diagrams and module hierarchy
    |-- system_architecture.md         # Post-audit v2.0 system architecture
    |-- BitbyBit_Complete_Guide.md     # Complete project guide
    |-- Judge_QA.md                    # 551-line Q&A prep for evaluators
    |-- Simulation_Commands.md         # All individual sim commands with expected output
    |-- gpu_visualization.html         # Interactive architecture visualization (869-line HTML)
    |-- Website_Reference.md
```

---

## 3. ARCHITECTURE SUMMARY

### Data Flow (Token In -> Token Out)

```
Token ID (integer)
    |
    v
[Embedding Lookup] -- token_emb[token_id] + pos_emb[position]
    |
    v
[Transformer Block x N layers]
    |-- LayerNorm 1 (inv_sqrt_lut_256)
    |-- Multi-Head Attention (Q*K^T / sqrt(d_k), softmax, weighted V sum)
    |   |-- KV Cache (stores past K,V vectors)
    |   |-- exp_lut_256 for softmax
    |-- Residual Add
    |-- LayerNorm 2
    |-- FFN (Linear -> GELU/ReLU -> Linear)
    |   |-- gelu_lut_256 for activation
    |   |-- zero_detect_mult skips zero activations
    |-- Residual Add
    |
    v
[Final LayerNorm]
    |
    v
[Argmax] --> Next Token ID (fed back as input for autoregressive generation)
```

### Key Design Points

| Feature | Implementation |
|---------|---------------|
| Arithmetic | Q8.8 signed fixed-point (16-bit), all `signed` Verilog |
| Multiply | `a * b` produces 16-bit result, zero-skip when either operand = 0 |
| Pipeline | 5-stage: FETCH -> DEQUANT -> ZERO_CHECK -> ALU -> WRITEBACK |
| Backpressure | `ready`/`valid` handshaking with `downstream_ready` |
| Memory | SRAM-based weights, parity protection, AXI4-Lite interface |
| Scaling | Parameterized: EMBED_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS, VOCAB_SIZE |
| Clock | Simulation assumes 100 MHz target |
| LUTs | 256-entry tables for exp, GELU, inv_sqrt (replace costly math) |

### Accelerated vs Original Engine

There are TWO parallel implementations:

1. **Original**: `gpt2_engine.v` + `transformer_block.v` + `attention_unit.v` + `ffn_block.v`
   - Uses `layer_norm.v` (with inv_sqrt LUT), `softmax_unit.v`, `gelu_activation.v`
   - Cleaner architecture but no zero-skip integration

2. **Accelerated**: `accelerated_gpt2_engine.v` + `accelerated_transformer_block.v` + `accelerated_attention.v` + `accelerated_linear_layer.v`
   - Uses `gpu_core.v` with zero-skip, KV cache
   - THE primary demo path — this is the one that matters
   - `accelerated_transformer_block.v` does NOT use `layer_norm.v` (has inline simple norm)

---

## 4. CURRENT TEST RESULTS (After Fixes)

Last run of `run_all_tests.ps1` (Feb 28, 2026):

```
Phase Module                 Status    Tests
----- ------                 ------    -----
P1    zero_detect_mult       PASS      7/7     ✅ FIXED (was 5/7)
P1    variable_precision_alu PASS      6/6     ✅ FIXED (was 5/6)
P1    sparse_memory_ctrl     PASS      6/6     ✅
P1    fused_dequantizer      PASS      8/8     ✅ FIXED (was 6/8)
P1    gpu_top                PASS      5/5     ✅
P2    mac_unit               PASS      8/8     ✅
P2    systolic_array         NO OUTPUT 0/0     ❌ NEEDS FIX (TB issue)
P2    gelu_activation        PASS      9/9     ✅ FIXED (was COMPILE FAIL)
P2    softmax_unit           NO OUTPUT 0/0     ❌ NEEDS FIX (RTL bug: sums wrong)
P3    layer_norm             PASS      3/3     ✅ FIXED (was COMPILE FAIL)
P3    linear_layer           PASS      2/2     ✅
P3    attention_unit         NO OUTPUT 0/0     ❌ NEEDS FIX (TB output format)
P3    ffn_block              NO OUTPUT 0/0     ❌ NEEDS FIX (TB output format)
P4    embedding_lookup       PASS      2/2     ✅
P4    gpt2_engine_FULL       PASS      1/1     ✅ FIXED (was COMPILE FAIL)
P4    accel_gpt2_engine      NO OUTPUT 0/0     ❌ NEEDS FIX (TB output format)
P5    axi_weight_memory      PASS      4/4     ✅
```

**Score: 12/17 modules PASS, 5 show "NO OUTPUT"**

**IMPORTANT**: The accelerated GPT-2 engine DOES work correctly when run directly:
```
Token 0: in=1 pos=0 -> out=0  (328 cycles)
Token 1: in=0 pos=1 -> out=3  (326 cycles)
Token 2: in=3 pos=2 -> out=3  (328 cycles)
Total cycles: 1070, Total zero-skips: 42
```
The "NO OUTPUT" in the test runner is because the TB uses lowercase `passed`/`failed` instead of `[PASS]`/`[FAIL]` markers that the regex looks for.

---

## 5. WHAT WAS ALREADY FIXED

### 5a. Compile Failures Fixed (6 modules)

**Root cause**: `run_all_tests.ps1` was missing LUT file dependencies.

| Module | Missing Dependency | Fix Applied |
|--------|--------------------|-------------|
| `gelu_activation` | `gelu_lut_256.v` | Added to Sources array |
| `softmax_unit` | `exp_lut_256.v` | Added to Sources array |
| `layer_norm` | `inv_sqrt_lut_256.v` | Added to Sources array |
| `attention_unit` | `exp_lut_256.v` | Added to Sources array |
| `ffn_block` | `gelu_lut_256.v` | Added to Sources array |
| `gpt2_engine_FULL` | All 3 LUT files | Added to Sources array |

All changes were made in `scripts/run_all_tests.ps1`.

### 5b. Test Failures Fixed (3 modules)

**Root cause**: Testbenches used unsigned constants but RTL modules use signed arithmetic.

1. **`zero_detect_mult_tb.v`** (was 5/7, now 7/7):
   - Test 5: `8'd255 * 8'd255` expected `65025` — but 255 as signed 8-bit = -1, so (-1)*(-1) = 1
   - Test 6: `8'd200` as signed = -56, so `1 * (-56) = -56`
   - **Fix**: Changed to signed test values: `-1 * -1 = 1`, `1 * 100 = 100`

2. **`variable_precision_alu_tb.v`** (was 5/6, now 6/6):
   - Test: `0xFFFF * 0xFFFF` expected `0xFFFE0001` (unsigned) — but signed -1 * -1 = 1
   - **Fix**: Changed expected value to `{32'd0, 32'd1}`

3. **`fused_dequantizer_tb.v`** (was 6/8, now 8/8):
   - Test 4: `(15-0)*15=225` exceeds signed 8-bit max (127), gets clamped. TB expected 225.
   - Test 6: `(3-8)*2=-10` is a valid signed result. TB expected 0 (unsigned clamp).
   - **Fix**: Expected clamped `127` for overflow, expected `-10` for negative

### 5c. Test Suite Expanded

- Added `accel_gpt2_engine` test (Phase 4) — was completely missing from `run_all_tests.ps1`
- Added `axi_weight_memory` test (Phase 5) — was missing
- Added `inv_sqrt_lut_256.v` dependency for accel_gpt2 engine compile

### 5d. Systolic Array TB Partially Fixed

- Added missing `clear_acc` port connection (RTL has it, TB didn't wire it)
- Added wait loops for systolic pipeline delay (TB was checking result immediately after `valid_in` deassert, but systolic array needs `2*ARRAY_SIZE` = 8 cycles)
- **Still shows NO OUTPUT** — needs further investigation (display output may be swallowed)

---

## 6. REMAINING BUGS TO FIX

### Bug 1: "NO OUTPUT" — TB Output Format Mismatch (4 modules)

**Affected**: `attention_unit`, `ffn_block`, `accel_gpt2_engine`, `systolic_array`

**Root Cause**: The `run_all_tests.ps1` regex counts `\[PASS\]` and `\[FAIL\]` markers. These TBs either:
- Print `pass_count = pass_count + 1` silently (no `[PASS]` marker printed on success)
- Print `%0d passed, %0d failed` at end (lowercase, no brackets)

**When run individually, they all work**:
- `attention_unit`: "2 passed, 0 failed" (actually WORKS)
- `ffn_block`: "1 passed, 0 failed" (actually WORKS)
- `accel_gpt2_engine`: Prints full pipeline results (WORKS perfectly)
- `systolic_array`: Valid outputs received but no [PASS] printed

**Fix needed**: Add `$display("[PASS] ...")` lines to these TBs on success, OR update the regex in `run_all_tests.ps1` to also match lowercase `passed`/`Results: N passed`.

Specific files to edit:
- `tb/transformer/attention_unit_tb.v` — ~line 120, add `$display("[PASS] ...");` before `pass_count = pass_count + 1;`
- `tb/transformer/ffn_block_tb.v` — ~line 133, same
- `tb/gpt2/accelerated_gpt2_engine_tb.v` — needs `[PASS]` markers in output
- `tb/compute/systolic_array_tb.v` — already has `[PASS]` in the fixed version but display may be suppressed by VCD $dumpfile error

### Bug 2: Softmax Unit — Probability Sums Are Wrong (REAL RTL BUG)

**File**: `rtl/compute/softmax_unit.v`

**Symptom**: All 5 softmax tests FAIL. Probability sums should be ~256 (Q0.8 format) but actual sums are 390-1016.

**Root Cause Analysis**: The COMPUTE state has a **1-cycle LUT latency bug**:
```verilog
// COMPUTE state (line ~95):
lut_input <= x_buf[idx] - max_val;    // Sets LUT input (registered)
exp_val[idx] <= lut_output;            // Reads LUT output SAME CYCLE
exp_sum <= exp_sum + {8'd0, lut_output};  // Uses STALE lut_output
```

The `lut_input` is a `reg` assigned with `<=` (non-blocking). The LUT is combinational (`always @(*)`). But `lut_input` won't update until next clock edge, so `lut_output` reflects the PREVIOUS `lut_input` value. Element 0 reads garbage, and every subsequent element reads the PREVIOUS element's exp value.

**Fix**: Either:
1. Make `lut_input` a `wire` with combinational assign, OR
2. Add a pipeline stage: compute lut_input in one cycle, read lut_output in the next
3. Split COMPUTE into two sub-states: COMPUTE_ADDR (set lut_input) and COMPUTE_READ (read lut_output)

### Bug 3: Softmax Reciprocal Normalization May Overflow

In the NORMALIZE state:
```verilog
norm_product = {16'd0, exp_val[idx]} * {8'd0, reciprocal};
prob_out[idx*8 +: 8] <= norm_product[15:8];
```
The bit selection `[15:8]` may not be the correct range for the product. If `exp_val` is 8-bit and `reciprocal` is 16-bit, product is 24-bit. The correct normalization shift depends on the reciprocal scaling factor (65536 / sum). Need to verify the math: `exp_val * (65536/sum) >> 16` should yield `exp_val * 256 / sum`. The bit extraction should probably be `[23:16]` or `>> 16`.

### Bug 4: No Testbenches for `rtl/top/` Modules

The following modules have ZERO testbenches:
- `command_processor.v` (218 lines) — FIFO command queue with 8 opcodes
- `gpu_config_regs.v` — AXI4-Lite configuration registers
- `perf_counters.v` (103 lines) — 8 hardware performance counters
- `reset_synchronizer.v` — 2-FF reset synchronizer

These are critical for standalone GPU operation and are completely untested.

### Bug 5: No Testbenches for Memory Modules (Partial)

- `dma_engine.v` (234 lines) — NO testbench
- `scratchpad.v` — NO testbench
- `sparse_memory_ctrl_wide.v` — NO testbench
- Only `axi_weight_memory.v` has a testbench (and it passes 4/4)

### Bug 6: Cosim Accuracy Issues at dim=64

From `cosim_report_dim64.txt`:
- Verilog vs Q8.8 reference match: **0/5 tokens** (0% match!)
- Verilog vs Float32 match: **0/5 tokens**
- Average MSE vs Q8.8: **0.98** (very high)
- Logit values are near-zero in Verilog (~0.03) but should be ~0.5-1.7

This suggests weight loading or accumulation precision loss at larger dimensions. The dim=4 cosim works (5/5 Q8.8 match) but dim=64 breaks.

---

## 7. CRITICAL IMPROVEMENTS NEEDED (GPU Reliability)

### Priority 1: Fix Softmax RTL Bug
The softmax is used in attention scoring. Wrong probabilities = wrong attention = wrong everything. This is the #1 blocking bug for numerical correctness.

### Priority 2: Fix dim=64 Cosim Accuracy
The GPU produces near-zero logits at dim=64 while the CPU reference produces proper values. Likely causes:
- Q8.8 accumulator overflow when summing 64 products (max accumulator is 32-bit but intermediate Q8.8 multiply results need careful shift handling)
- Weight loading via `$readmemh` may have endianness or addressing bugs in `run_scaled_cosim.py`
- LayerNorm numerical issues at larger dimensions

### Priority 3: Add Missing Testbenches
Create TBs for: `command_processor`, `gpu_config_regs`, `perf_counters`, `reset_synchronizer`, `dma_engine`, `scratchpad`. These are needed for standalone GPU operation.

### Priority 4: Top-Level Integration
There is NO single top-level module that wires everything together:
- `command_processor` -> `gpu_core` / `systolic_array`
- `dma_engine` -> `axi_weight_memory` -> `gpu_core`
- `perf_counters` connected to pipeline events
- `gpu_config_regs` connected to all configurable parameters
- `scratchpad` connected as intermediate buffer

A `gpu_system_top.v` module is needed that instantiates ALL of these and provides the external AXI interface. This is what makes it a "standalone GPU."

### Priority 5: Tiled MatMul Verification
`tiled_matmul.v` exists but has no testbench. This is critical for real-world matrix sizes that exceed the systolic array dimensions.

### Priority 6: Multi-Head Attention
Current attention is single-head. Real GPT-2 uses 12 heads. The architecture supports NUM_HEADS parameter but the head-splitting/concatenation logic needs verification at scale.

### Priority 7: BF16 Integration Path
`bf16_multiply.v` exists but is not integrated into any pipeline. For better accuracy than Q8.8, a BF16 mode should be selectable via `gpu_config_regs`.

### Priority 8: Clock Domain Crossing
No CDC logic exists. If the AXI bus runs at a different clock than the compute pipeline, CDC FIFOs are needed.

### Priority 9: Power Optimization
- Clock gating signals exist in `gpu_core.v` (Issue #15) but are not verified
- Zero-skip reduces computation but doesn't gate the clock to idle lanes
- Need power analysis testbench

### Priority 10: FPGA Synthesis Readiness
- All `initial begin` blocks with LUT data need to be converted to `$readmemh` for synthesis
- Memory inference directives may be needed for Xilinx/Intel FPGA tools
- Timing constraints file (.xdc/.sdc) needed
- I/O pin assignment needed

---

## 8. SIMULATION & TOOLING

### Installed Tools

| Tool | Path | Version |
|------|------|---------|
| Icarus Verilog (iverilog) | `D:\Tools\iverilog\bin\iverilog.exe` | 12.0 (devel) |
| VVP (simulator) | `D:\Tools\iverilog\bin\vvp.exe` | 12.0 |
| GTKWave (waveforms) | `D:\Tools\iverilog\bin\gtkwave.exe` | (should be bundled) |
| Python | Needs verification — was not responding in terminal | 3.10+ required |

### Running the Full Test Suite

```powershell
cd D:\Projects\BitbyBit\custom_gpu_project
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1
```

### Running Individual Tests

Each test follows this pattern:
```powershell
# Compile
D:\Tools\iverilog\bin\iverilog.exe -o sim/OUTPUT_NAME [RTL_FILES...] [TB_FILE]

# Simulate
D:\Tools\iverilog\bin\vvp.exe sim/OUTPUT_NAME
```

### Key Individual Test Commands

**Accelerated GPT-2 (THE most important test)**:
```powershell
cd D:\Projects\BitbyBit\custom_gpu_project
D:\Tools\iverilog\bin\iverilog.exe -o sim/gpt2_acc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/gpt2/accelerated_gpt2_engine_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gpt2_acc_test
```
**Expected output**:
```
Token 0: in=1 pos=0 -> out=0  (328 cycles)
Token 1: in=0 pos=1 -> out=3  (326 cycles)
Token 2: in=3 pos=2 -> out=3  (328 cycles)
Total cycles: 1070, Total zero-skips: 42
```

**Softmax (currently FAILING)**:
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/sm_test rtl/compute/exp_lut_256.v rtl/compute/softmax_unit.v tb/compute/softmax_unit_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/sm_test
```

### Viewing Waveforms

```powershell
# Testbenches that generate VCD files write to sim/waveforms/
# Make sure directory exists:
mkdir sim\waveforms -Force

# After running a test with $dumpfile, view with GTKWave:
D:\Tools\iverilog\bin\gtkwave.exe sim\waveforms\zero_detect_mult.vcd
```

---

## 9. GPT-2 MODEL EXECUTION

### Python Cosimulation (Python generates TB + runs iverilog)

**Small scale (dim=4)**:
```powershell
python scripts/run_cosim.py --token 5
```

**Scaled (dim=64)**:
```powershell
python scripts/run_scaled_cosim.py --sentence "hello" --dim 64
```

**Sentence-level**:
```powershell
python scripts/run_sentence_cosim.py
```

### Interactive Chat (Pure NumPy, no Verilog)

**GPT-2 Chat**:
```powershell
python scripts/chat_gpt2.py
python scripts/chat_gpt2.py --max-tokens 50 --relu  # ReLU mode for sparsity
```

**OPT-125M Chat (better sparsity demo)**:
```powershell
python scripts/chat_opt.py
python scripts/chat_opt.py --relu-mode all --max-tokens 50
```

These scripts download weights from HuggingFace on first run and perform inference in pure NumPy while tracking GPU performance metrics (zero-skip rates, cycle estimates).

### Tools Needed for Direct GPT-2 Simulation

The user specifically requested:
1. **A tool to simulate and run GPT-2 directly** — The `run_scaled_cosim.py` script does this (generates a Verilog TB with real weights baked in, compiles with iverilog, runs). But it needs Python working. An alternative is to create a standalone Verilog testbench with weights loaded via `$readmemh` from hex files in `weights/gpt2_dim64/hex_sim/`.

2. **A tool to visualize** — Options:
   - **GTKWave** (already available at `D:\Tools\iverilog\bin\gtkwave.exe`) for VCD waveforms
   - **WaveDrom** (npm package `wavedrom-cli`) for timing diagram generation
   - **Surfer** (modern GTKWave alternative, `cargo install surfer` or download binary)
   - **The existing `docs/gpu_visualization.html`** — 869-line interactive HTML visualization
   - **D3.js / Web-based** — A custom HTML dashboard that parses simulation output and shows pipeline state, zero-skip rates, token flow

---

## 10. VISUALIZATION REQUIREMENTS

The user wants visualization tools. Here is what exists and what needs to be created:

### Already Exists
1. `docs/gpu_visualization.html` — Interactive architecture block diagram (869 lines, D3.js-style)
2. `website/` — Project showcase site
3. VCD waveform output in testbenches (but `sim/waveforms/` dir may need creation)

### Needs to Be Created
1. **Pipeline visualization tool** — Parse VCD or simulation output, show cycle-by-cycle pipeline state
2. **Zero-skip rate dashboard** — Real-time chart showing sparsity per layer/token
3. **Token flow visualization** — Show token going through embedding -> transformer blocks -> output
4. **Integrated simulation runner + visualizer** — Single script that runs simulation AND opens visualization

### Recommended Approach
Create a Python script (`scripts/visualize_simulation.py`) that:
1. Runs the accelerated GPT-2 simulation
2. Parses VCD output with `vcdvcd` Python package
3. Generates an HTML dashboard with:
   - Pipeline state timeline
   - Zero-skip heatmap per layer
   - Token prediction sequence
   - Cycle count breakdown
   - Waveform snapshots for key signals

OR create a standalone HTML file (`docs/simulation_dashboard.html`) that accepts pasted simulation output and renders it.

---

## 11. EXACT COMMANDS REFERENCE

### Full simulation commands for every module

See `docs/Simulation_Commands.md` for the complete list (231 lines). Key ones:

```powershell
# All tests at once:
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1

# Phase 1 - Primitives
D:\Tools\iverilog\bin\iverilog.exe -o sim/zdm_test rtl/primitives/zero_detect_mult.v tb/primitives/zero_detect_mult_tb.v
D:\Tools\iverilog\bin\iverilog.exe -o sim/mc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/primitives/gpu_multicore.v tb/primitives/gpu_multicore_tb.v

# Phase 2 - Compute
D:\Tools\iverilog\bin\iverilog.exe -o sim/sa_test rtl/compute/systolic_array.v tb/compute/systolic_array_tb.v
D:\Tools\iverilog\bin\iverilog.exe -o sim/sm_test rtl/compute/exp_lut_256.v rtl/compute/softmax_unit.v tb/compute/softmax_unit_tb.v

# Phase 3 - Transformer
D:\Tools\iverilog\bin\iverilog.exe -o sim/au_test rtl/compute/exp_lut_256.v rtl/transformer/attention_unit.v tb/transformer/attention_unit_tb.v

# Phase 4 - GPT-2
D:\Tools\iverilog\bin\iverilog.exe -o sim/gpt2_acc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/gpt2/accelerated_gpt2_engine_tb.v

# Phase 5 - Memory
D:\Tools\iverilog\bin\iverilog.exe -o sim/axi_test rtl/memory/axi_weight_memory.v tb/memory/axi_weight_memory_tb.v

# Demo
D:\Tools\iverilog\bin\iverilog.exe -o sim/demo_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/demo/gpt2_demo_tb.v
```

---

## 12. KEY FILE CONTENTS SUMMARY

### `rtl/primitives/zero_detect_mult.v` (40 lines)
Signed 8-bit multiplier. If `a == 0 || b == 0`, output 0 and set `skipped=1`. Single-cycle latency. Synchronous reset.

### `rtl/primitives/gpu_core.v` (285 lines)
Parameterized N-lane compute core. 5-stage pipeline: FETCH weights from SRAM -> DEQUANT (INT4->INT8) -> ZERO_CHECK -> ALU (multiply) -> WRITEBACK (accumulate). Has `acc_clear`, `ready/valid` handshaking, `downstream_ready` backpressure, parity error detection on weight memory, clock-gate enable for zero-skipped lanes.

### `rtl/compute/systolic_array.v` (193 lines)
NxN PE mesh with weight-stationary dataflow. Weights preloaded via `load_weight` interface. Activations flow left-to-right with skewed (staggered) input. Partial sums flow top-to-bottom. Zero-skip in each PE. Result available after `2*ARRAY_SIZE` cycles. Has `clear_acc` for new computations.

### `rtl/compute/softmax_unit.v` (136 lines) — **HAS BUG**
State machine: IDLE -> FIND_MAX -> COMPUTE (exp LUT) -> NORMALIZE (reciprocal) -> OUTPUT. The COMPUTE state has a 1-cycle latency bug where `lut_input` is registered but `lut_output` is read in the same cycle (reads stale data).

### `rtl/transformer/accelerated_attention.v`
KV-cached multi-head attention. Computes Q*K^T scoring with cached K/V from previous tokens. Uses exp_lut_256 for softmax. Reports zero-skip count.

### `rtl/gpt2/accelerated_gpt2_engine.v`
Top-level accelerated inference engine. Embedding lookup -> N transformer blocks -> final norm -> argmax. Supports autoregressive generation (output token fed back as input).

### `rtl/top/command_processor.v` (218 lines)
FIFO-based command queue. 8 opcodes: NOP, LOAD_WEIGHTS, MATMUL, ACTIVATION, LAYERNORM, SOFTMAX, RESIDUAL_ADD, EMBEDDING, FENCE. 64-bit command descriptors. This is what makes the GPU "standalone" — a host CPU pushes commands over AXI and the GPU executes them.

### `rtl/memory/dma_engine.v` (234 lines)
AXI4 master for bulk weight transfers from external memory (DRAM) to local SRAM. Supports burst reads, configurable transfer length, completion interrupt.

### `rtl/top/perf_counters.v` (103 lines)
8 hardware counters: CYCLE_COUNT, ACTIVE_CYCLES, STALL_CYCLES, TOTAL_MACS, ZERO_SKIP_COUNT, MEMORY_READS, MEMORY_WRITES, PARITY_ERRORS. Readable via index.

### `scripts/run_all_tests.ps1` (169 lines)
PowerShell test runner. Compiles and runs each module, counts `[PASS]`/`[FAIL]` regex matches, prints summary table. Needs the regex fix to also match lowercase pass/fail patterns.

### `scripts/chat_gpt2.py` (546 lines)
Full GPT-2-small inference in pure NumPy. BPE tokenizer, 12-layer transformer, downloads weights from HuggingFace. Tracks zero-skip rates and cycle estimates.

### `scripts/chat_opt.py` (568 lines)
OPT-125M inference with native ReLU activation. Shows 92% activation sparsity. Best demo for zero-skip hardware feature.

### `scripts/run_scaled_cosim.py` (668 lines)
Generates a complete Verilog testbench with real GPT-2 weights (quantized to Q8.8), compiles with iverilog, runs simulation, parses output, compares against Python float32 reference. Uses `$readmemh` for weight loading at dim=64.

---

## 13. ACTION PLAN (What To Do Next)

### Phase A: Fix Remaining Test Issues (IMMEDIATE)

1. **Fix softmax RTL bug** (`rtl/compute/softmax_unit.v`):
   - Split COMPUTE state into two sub-states to handle LUT latency
   - Verify probability sums equal ~256 for all test vectors

2. **Fix TB output format** for 4 modules:
   - `tb/transformer/attention_unit_tb.v`: Add `$display("[PASS] ...");` on success
   - `tb/transformer/ffn_block_tb.v`: Same
   - `tb/gpt2/accelerated_gpt2_engine_tb.v`: Same
   - `tb/compute/systolic_array_tb.v`: Verify display output works

3. **Alternative**: Update `run_all_tests.ps1` regex to also match `passed` and `Results: N passed`:
   ```powershell
   $passes = ([regex]::Matches($simOutput, '\[PASS\]|(?<=Results:\s)\d+(?=\s+passed)')).Count
   ```

### Phase B: Create Missing Testbenches (HIGH PRIORITY)

4. Create `tb/top/command_processor_tb.v` — test all 8 opcodes
5. Create `tb/top/perf_counters_tb.v` — verify all 8 counters
6. Create `tb/top/gpu_config_regs_tb.v` — test AXI register read/write
7. Create `tb/top/reset_synchronizer_tb.v` — test async-to-sync
8. Create `tb/memory/dma_engine_tb.v` — test burst transfers
9. Create `tb/memory/scratchpad_tb.v` — test dual-port access

### Phase C: Create Top-Level Integration (CRITICAL for Standalone GPU)

10. Create `rtl/top/gpu_system_top.v` that wires:
    - `command_processor` -> `gpu_core` / `systolic_array` / activation units
    - `dma_engine` -> `axi_weight_memory`
    - `scratchpad` as intermediate buffer
    - `perf_counters` connected to all pipeline events
    - `gpu_config_regs` for runtime configuration
    - `reset_synchronizer` on the input reset
    - External AXI4-Lite slave interface for host

11. Create `tb/top/gpu_system_top_tb.v` — end-to-end test

### Phase D: Fix Numerical Accuracy (HIGH PRIORITY)

12. Debug dim=64 cosim accuracy:
    - Add overflow detection in accumulator path
    - Verify Q8.8 multiply shift-right-by-8 is happening correctly in all layers
    - Check `$readmemh` hex file format matches Verilog expectations
    - Consider wider accumulators (40-bit or 48-bit) for larger dimensions

### Phase E: Simulation & Visualization Tools (USER REQUESTED)

13. **Set up simulation runner**:
    - Verify Python is working (was not responding in terminal)
    - Install `numpy`, `safetensors` if needed
    - Test `python scripts/run_cosim.py` end-to-end
    - Test `python scripts/run_scaled_cosim.py --sentence "hello" --dim 64`

14. **Create visualization dashboard**:
    - Install `pip install vcdvcd` for VCD parsing
    - Create `scripts/visualize_simulation.py` that runs sim + generates HTML dashboard
    - OR create standalone `docs/simulation_dashboard.html`

15. **Verify GTKWave** works:
    ```powershell
    D:\Tools\iverilog\bin\gtkwave.exe sim\waveforms\zero_detect_mult.vcd
    ```

16. **Consider Surfer** as modern alternative to GTKWave:
    - Download from https://surfer-project.org/
    - Supports VCD, FST, GHW formats
    - Better UI than GTKWave

### Phase F: Advanced Improvements (STRETCH)

17. Create formal verification properties (SVA assertions) for critical modules
18. Add clock gating verification testbench
19. Create FPGA synthesis script (Vivado TCL or Quartus QSF)
20. Add CDC (Clock Domain Crossing) FIFOs if needed
21. Integrate BF16 multiply as configurable precision mode
22. Add multi-head attention verification at NUM_HEADS > 2
23. Power estimation using switching activity from VCD files
24. Create a CI pipeline that runs all tests automatically

---

## QUICK-START FOR THE NEXT AGENT

```powershell
# 1. Navigate to project
cd D:\Projects\BitbyBit\custom_gpu_project

# 2. Run full test suite (see current state)
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1

# 3. Run the key GPT-2 accelerated test directly
D:\Tools\iverilog\bin\iverilog.exe -o sim/gpt2_acc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/gpt2/accelerated_gpt2_engine_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gpt2_acc_test

# 4. Start fixing: softmax bug first, then TB output formats, then missing TBs
# 5. Read CLAUDE.md for additional project context
# 6. Read docs/Simulation_Commands.md for all individual test commands
```

---

*End of handoff document. The next agent should start with Phase A (fix softmax bug + TB output formats), then proceed through B-F in order.*

