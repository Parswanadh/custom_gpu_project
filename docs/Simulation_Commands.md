# BitbyBit — Individual Simulation Commands & Expected Behavior

> Copy-paste any command below directly into PowerShell. No setup needed.
> All commands assume you are in `d:\Projects\BitbyBit\custom_gpu_project`.

---

## Demo-day one-command launcher (recommended)

Use the new launcher to run measured base vs imprint model demos from simulation logs.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1 -Mode compare -TokenId 5 -Position 2
```

By default, this now runs paired compare with a small workload matrix (`WorkloadMode=matrix`, `WarmupRuns=1`, `MeasuredRuns=3`).
For single-workload paired benchmarking with explicit knobs:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1 -Mode compare -WorkloadMode single -TokenId 5 -Position 2 -WarmupRuns 1 -MeasuredRuns 3
```

This writes a run bundle summary to:
- `sim\bench_runs\<run_id>\compare_summary.json`
- `sim\compare_summary_latest.json`

Available modes:
- `top`: top-level command-latency demo (`gpu_system_top_v2_tb`)
- `base`: base full-model inference demo
- `imprint`: MINI imprint full-model inference demo
- `compare`: base + imprint + top summaries (default)
- `all`: same as compare (explicit)

---

## Phase 1: Core Compute Primitives

### 1. Zero-Detect Multiplier
> Checks if weight or activation is zero and skips the multiply.
**Expected Behavior:** The test applies various inputs (normal numbers, zero weight, zero activation, both zero). When a zero is present, the testbench confirms that the internal `zero_skip_flag` goes high and the multiplier output immediately goes to 0 without calculation. Expect to see `7/7 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/zdm_test rtl/primitives/zero_detect_mult.v tb/primitives/zero_detect_mult_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/zdm_test
```

### 2. Variable Precision ALU
> Supports different bit-width operations (8-bit, 16-bit).
**Expected Behavior:** The module executes addition, subtraction, and multiplication at different precisions. You'll see verification of exact fixed-point math outputs. Expect `6/6 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/vpa_test rtl/primitives/variable_precision_alu.v tb/primitives/variable_precision_alu_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/vpa_test
```

### 3. Sparse Memory Controller
> Skips zero-valued entries during memory reads.
**Expected Behavior:** The test writes a mix of zero and non-zero values into memory. When reading, it simulates reading a block of data but verifies that the output correctly flags zero entries, allowing downstream components to skip processing. Expect `6/6 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/smc_test rtl/primitives/sparse_memory_ctrl.v tb/primitives/sparse_memory_ctrl_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/smc_test
```

### 4. Fused Dequantizer
> Scales quantized weights using configurable scale/offset.
**Expected Behavior:** The test passes in 4-bit and 8-bit quantized values along with a scale factor. The output will show the correctly reconstructed 16-bit fixed-point (Q8.8) values. Expect `8/8 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/fd_test rtl/primitives/fused_dequantizer.v tb/primitives/fused_dequantizer_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/fd_test
```

### 5. GPU Top (Integrated Primitives)
> Full GPU top-level with all primitives wired together.
**Expected Behavior:** This integrates the dequantizer, ALU, and sparse memory. The test streams a sequence of operations into the compute unit. Expect `5/5 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gt_test rtl/primitives/zero_detect_mult.v rtl/primitives/variable_precision_alu.v rtl/primitives/sparse_memory_ctrl.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_top.v tb/primitives/gpu_top_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gt_test
```

### 6. GPU Multi-Core (4 cores x 32 lanes = 128-wide) ⭐
> The standalone multi-core benchmark. 256 products, 96 zero-skips.
**Expected Behavior:** The testbench simulates 4 parallel GPU cores operating on a broad set of weights. It tests the "Broadcast" topology (sending the same activation to all cores). The output will print a **Multi-Core Performance Report**, showing that it computed 256 products in just 2 cycles, successfully skipped 96 zero operations, and achieved a theoretical 12800 MOPS at 100MHz. It will display `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/mc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/primitives/gpu_multicore.v tb/primitives/gpu_multicore_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/mc_test
```

### 7. GPU Top Pipelined
> Pipelined version of GPU top module.
**Expected Behavior:** Verifies the 5-stage pipeline behavior. It pushes operations into the pipeline and checks that the final correct result emerges exactly 5 clock cycles later. Expect `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gtp_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/variable_precision_alu.v rtl/primitives/gpu_top_pipelined.v tb/primitives/gpu_top_pipelined_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gtp_test
```

### 8. GPU Top Integrated
> Integrated version with all sub-modules connected.
**Expected Behavior:** The full integration test of the early primitive designs. Expect `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gti_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/variable_precision_alu.v rtl/primitives/sparse_memory_ctrl.v rtl/primitives/gpu_top_integrated.v tb/primitives/gpu_top_integrated_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gti_test
```

---

## Phase 2: Extended Compute Modules

### 9. MAC Unit (Multiply-Accumulate)
> Basic multiply-accumulate unit in Q8.8.
**Expected Behavior:** Performs multiple Q8.8 multiplications and additions into a running accumulator. Tests overflow, negative numbers, and standard MAC operations. Expect `8/8 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/mac_test rtl/compute/mac_unit.v tb/compute/mac_unit_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/mac_test
```

### 10. Systolic Array
> Matrix multiply array.
**Expected Behavior:** Pumps a small matrix multiplication through a 2D array of MAC units. The output verifies the final computed matrix matches the mathematical expectation. Expect `3/3 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/sa_test rtl/compute/systolic_array.v tb/compute/systolic_array_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/sa_test
```

### 11. GELU Activation
> Gaussian Error Linear Unit with piecewise approximation.
**Expected Behavior:** The testbench provides inputs like -5.0, -1.0, 0.0, 1.0, and 5.0 to the hardware. The output will show the corresponding approximated GELU results (e.g., input=1.0 will output ~0.84). Expect `9/9 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gelu_test rtl/compute/gelu_activation.v tb/compute/gelu_activation_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gelu_test
```

### 12. Softmax Unit
> Probability normalization using exp() lookup.
**Expected Behavior:** Passes a set of 4 arbitrary logit values into the softmax unit. The test verifies that the unit successfully reads the 256-entry exponential LUT, correctly calculates the sum denominator, and outputs 4 normalized probabilities that sum to ~1.0. Expect `5/5 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/sm_test rtl/compute/softmax_unit.v tb/compute/softmax_unit_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/sm_test
```

---

## Phase 3: Transformer Building Blocks

### 13. Layer Normalization
> Mean/variance normalization in Q8.8.
**Expected Behavior:** Takes an input vector, computes its mean and variance in hardware, and normalizes it. Outputs will show the vector centered around 0 with a standard deviation of 1. Expect `3/3 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/ln_test rtl/transformer/layer_norm.v tb/transformer/layer_norm_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/ln_test
```

### 14. Linear Layer
> Simple matrix multiply layer.
**Expected Behavior:** Performs a basic $W \times X + b$ operation using an un-pipelined matrix multiplier. It verifies the dot product matches perfectly. Expect `2/2 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/ll_test rtl/transformer/linear_layer.v tb/transformer/linear_layer_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/ll_test
```

### 15. Attention Unit (Original)
> Basic attention without KV-cache.
**Expected Behavior:** Simulates passing standard Q, K, and V vectors into an inline dot-product block. Tests the mathematical calculation (not performance). Expect `1/1 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/au_test rtl/transformer/attention_unit.v tb/transformer/attention_unit_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/au_test
```

### 16. Accelerated Attention (KV-Cached) ⭐
> Real Q*K^T scoring with KV cache and 256-entry exp LUT.
**Expected Behavior:** Generates outputs for 2 successive tokens. 
- Token 0 goes in, its K and V are written to memory, and it self-attends. 
- Token 1 goes in, and the hardware reads Token 0's K and V from the cache to compute cross-attention scores. 
The output will show the explicit input and output vectors, the cycle counts, and the amount of zero skips that occurred during scoring. Expect `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/accel_attn_test rtl/compute/exp_lut_256.v rtl/transformer/accelerated_attention.v tb/transformer/accelerated_attention_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/accel_attn_test
```

### 17. FFN Block
> Feed-forward network block.
**Expected Behavior:** A standard 2-layer neural network test (Linear -> ReLU -> Linear). Verifies the hardware executes the non-linear transformation correctly. Expect `1/1 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/ffn_test rtl/transformer/ffn_block.v tb/transformer/ffn_block_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/ffn_test
```

---

## Phase 4: GPT-2 Full Pipeline

### 18. Embedding Lookup
> Token ID to embedding vector with position encoding.
**Expected Behavior:** Looks up token ID = 5 and position = 2. Verifies that the hardware correctly fetches the token embedding from ROM, fetches the position embedding from ROM, and outputs the sum as a single vector. Expect `2/2 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/emb_test rtl/gpt2/embedding_lookup.v tb/gpt2/embedding_lookup_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/emb_test
```

### 19. GPT-2 Engine (Original)
> Original GPT-2 pipeline with basic components (no GPU pipeline).
**Expected Behavior:** The old version of the architecture. Outputs the generation of 1 token using un-accelerated `attention_unit` and `ffn_block`. Expect `1/1 PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gpt2_test rtl/gpt2/embedding_lookup.v rtl/gpt2/transformer_block.v rtl/gpt2/gpt2_engine.v rtl/transformer/layer_norm.v rtl/transformer/attention_unit.v rtl/transformer/ffn_block.v rtl/transformer/linear_layer.v rtl/compute/gelu_activation.v rtl/compute/softmax_unit.v tb/gpt2/gpt2_engine_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gpt2_test
```

### 20. Accelerated GPT-2 Engine (1 Layer) ⭐⭐⭐
> Full accelerated pipeline: embedding → transformer → argmax. 
**Expected Behavior:** **THE MOST IMPORTANT TEST.** This is the full system running autoregressive generation. 
The testbench first loads token/position embeddings into the hardware's memory via the initialization interface. Next, it kicks off a **3-token autoregressive generation cycle**, where the output prediction is fed directly back into the hardware as the next token input.
Output will explicitly show:
- `Token 0: in=1 pos=0 --> out=0`
- `Token 1: in=0 pos=1 --> out=3`
- `Token 2: in=3 pos=2 --> out=3` (Generation feedback loop)
It also outputs the pre-argmax probability logits and prints a summary block stating the total simulation cycles (~1070) and total zero-multiplies skipped (~42) to prove hardware power efficiency.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/gpt2_acc_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/gpt2/accelerated_gpt2_engine_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/gpt2_acc_test
```

---

## Phase 5: Memory Interface

### 21. AXI4-Lite Weight Memory ⭐
> Standard bus interface connection for an SoC (like a CPU loading weights).
**Expected Behavior:** Testbench acts like an external processor (CPU) writing data over an AXI4-Lite bus. It writes the magic memory values `0xDEADBEEF` and `0x12345678` into the GPU memory via standard AXI protocols. It then reads them back to verify memory stability. It also tests the "Start Inference" control switch and "Zero Skip Count" data registers. Expect `ALL PASSED`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/axi_test rtl/memory/axi_weight_memory.v tb/memory/axi_weight_memory_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/axi_test
```

---

## Phase 6: Demo / Integration Tests

### 22. Feature Verification
> End-to-end feature verification across multiple subsystems.
**Expected Behavior:** A sanity wrapper script simulating varied inputs to ensure zero-skip accumulation and pipeline draining edge cases respond as defined. Expect `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/feat_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/demo/feature_verification_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/feat_test
```

### 23. GPT-2 Demo
> Demo testbench showcasing the full GPT-2 inference flow.
**Expected Behavior:** Identical internal test vector to the Accelerated GPT-2 Engine (command 20), formatted specifically for broad viewer-friendly console output during a real presentation. Displays identical Token-in to Token-out functionality with zero-skip counts. Expect `PASS`.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -o sim/demo_test rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v tb/demo/gpt2_demo_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/demo_test
```

### 24. Unified Top-Level GPU (`gpu_system_top_v2`) ⭐
> Integrates Phase 7 command/DMA/control path with the optimized transformer-layer compute pipeline.
**Expected Behavior:** All `gpu_system_top_v2_tb` checks PASS (currently `16/16`). MATMUL should route through optimized/hardwired paths with multi-cycle latency (not 1-cycle stub), including error-path rejection and recovery checks.
```powershell
D:\Tools\iverilog\bin\iverilog.exe -g2012 -o sim/sys_v2_test rtl/top/reset_synchronizer.v rtl/top/gpu_config_regs.v rtl/top/command_processor.v rtl/top/perf_counters.v rtl/memory/scratchpad.v rtl/memory/dma_engine.v rtl/memory/imprinted_embedding_rom.v rtl/integration/imprinted_mini_transformer_core.v rtl/transformer/rope_encoder.v rtl/transformer/grouped_query_attention.v rtl/compute/parallel_softmax.v rtl/compute/exp_lut_256.v rtl/compute/recip_lut_256.v rtl/compute/gelu_lut_256.v rtl/compute/gelu_activation.v rtl/memory/kv_cache_quantizer.v rtl/compute/activation_compressor.v rtl/integration/optimized_transformer_layer.v rtl/top/gpu_system_top_v2.v tb/top/gpu_system_top_v2_tb.v
D:\Tools\iverilog\bin\vvp.exe sim/sys_v2_test
```

---

## Run Everything At Once

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1
```
**Expected Behavior:** The suite should finish with:
- non-zero exit on any compile/sim/timeout/no-output failure;
- `Total FAIL : 0` in summary;
- no `COMPILE FAIL`, `SIM FAIL`, `SIM LAUNCH FAIL`, `TIMEOUT`, or `NO OUTPUT` statuses.

## Canonical Production Demo Flow

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1
```

**Expected Behavior:** Runs top-level + full-model compare benchmark flow in one command (recommended default: `matrix`, `warmup=3`, `measured=10`) and regenerates:
- `sim\compare_summary_latest.json`
- `sim\phase3_benchmark_proof_pack.json`
- `sim\phase3_benchmark_proof_pack.csv`

For broader benchmark coverage with reproducible workload sampling, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_production_demo.ps1 `
  -WorkloadMode matrix -WarmupRuns 3 -MeasuredRuns 10 `
  -WorkloadCount 8 -WorkloadSeed 20260317
```

## Synthesis-Readiness Check (Conda Yosys Tooling)

```powershell
D:\Anaconda\Scripts\conda.exe run -n yosys-tools yowasp-yosys -V
D:\Anaconda\Scripts\conda.exe run -n yosys-tools yowasp-yosys -p "read_verilog rtl/compute/exp_lut_256.v; synth -top exp_lut_256; stat"
```

**Expected Behavior:** Yosys version prints successfully and synthesis/stat completes without frontend hierarchy errors.
