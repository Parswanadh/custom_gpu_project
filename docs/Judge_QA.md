# BitbyBit — Judge & Evaluator Q&A

> Complete preparation document with every possible question a judge, evaluator, or professor might ask about the BitbyBit custom GPU accelerator for LLM inference, along with honest, verified answers.

---

## 🔹 1. HIGH-LEVEL PROJECT QUESTIONS

### Q: What is this project in one sentence?

**A:** A custom GPU accelerator — designed entirely from scratch in Verilog — that can run GPT-2 inference on-device by processing tokens through 28 hand-designed RTL modules including a 5-stage pipelined compute engine, KV-cached attention, and an AXI bus interface.

---

### Q: Why did you build this instead of using an NVIDIA GPU or existing framework?

**A:** The goal was to understand hardware design from the ground up. We didn't use CUDA, OpenCL, or any existing GPU IP. Every gate, every pipeline stage, every memory interface was designed by hand in Verilog. This is a learning and research project that demonstrates full-stack hardware/software co-design — from Python weight extraction all the way down to combinational logic.

---

### Q: What problem does this solve?

**A:** Running large language models currently requires powerful GPUs that consume massive power. Edge devices (phones, IoT, embedded systems) can't run LLMs efficiently. This project explores what a purpose-built, LLM-specific accelerator looks like at the hardware level — one that exploits sparsity, uses fixed-point math, and caches attention state to minimize redundant computation.

---

### Q: Is this running on real hardware (FPGA/ASIC)?

**A:** No. This project is **simulation-only** using Icarus Verilog. We have not performed FPGA synthesis, place-and-route, or timing analysis. All performance numbers in clock cycles are verified from simulation, but any MHz or tokens/second claims are theoretical projections assuming a 100 MHz clock. We are transparent about this distinction.

---

### Q: What language model does it run?

**A:** GPT-2 Small (117M parameters). We extracted real weights from the HuggingFace model using a Python script, quantized them to Q8.8 fixed-point, and loaded them into the hardware via `$readmemh`.

---

### Q: Can it generate coherent text?

**A:** Our current test configuration uses EMBED_DIM=4 and VOCAB_SIZE=8 (a scaled-down version for simulation feasibility). It correctly performs autoregressive token generation — each token feeds back as input for the next. With real GPT-2 weights loaded (quantized to fit our test dimensions), the engine produces deterministic outputs and all tests pass. Scaling to the full GPT-2 dimensions (768 embedding, 50257 vocab) would require significantly more memory and simulation time but the architecture itself is parameterized to support it.

---

## 🔹 2. ARCHITECTURE QUESTIONS

### Q: Walk me through how a token goes from input to output.

**A:**
1. **Token Input** — An integer token ID enters (e.g., token 5)
2. **Embedding Lookup** — Token embedding + position embedding are added to create a vector
3. **Layer Norm 1** — Normalize to mean=0, std=1
4. **Attention with KV Cache** — Current token's Query scores against all past Keys; softmax converts to probabilities; output is weighted sum of past Values; K and V are cached
5. **Residual Add** — Add original input back
6. **Layer Norm 2** — Normalize again
7. **FFN via GPU Core** — 2-layer neural network (linear → ReLU → linear) processed through the pipelined compute engine
8. **Residual Add** — Add input back again
9. **Repeat** for N transformer layers
10. **Final Layer Norm → Argmax** → Predicted next token ID

---

### Q: What does "5-stage pipeline" mean?

**A:** The GPU core processes multiply-accumulate operations through 5 sequential stages, each taking 1 clock cycle:

| Stage | Name | What Happens |
|:-----:|------|-------------|
| 1 | FETCH | Read N weights from on-chip memory simultaneously |
| 2 | DEQUANT | Scale weights using configurable scale/offset |
| 3 | ZERO_CHECK | Detect zero-valued weights or activations |
| 4 | ALU | Multiply weight × activation — or skip if zero |
| 5 | WRITEBACK | Sum all lane products and accumulate |

The pipeline takes 5 cycles to fill, then produces one result per cycle. This is the same concept as CPU instruction pipelining — multiple operations are in-flight simultaneously at different stages.

---

### Q: What is the KV cache and why is it important?

**A:** In autoregressive generation, each new token needs to attend to ALL previous tokens. Without a cache, you'd recompute the Key and Value projections for every past token at every step — O(n²) work. The KV cache stores past K and V vectors, so each new token only computes its own Q, K, V, then scores Q against the cached Keys. This reduces per-token work from O(n·d) to O(d) for the projection step. Our implementation stores up to 32 positions (configurable MAX_SEQ parameter).

---

### Q: How does multi-core scaling work?

**A:** The multi-core wrapper uses a **broadcast topology**. The same activation vector is sent to all cores simultaneously, but each core has different weights. Each core independently computes its portion of the output, and results are accumulated. We tested 4 cores × 32 lanes = 128 parallel multiply-accumulate operations per cycle.

**Important caveat:** The multi-core system is verified standalone. The current transformer integration uses a single gpu_core with 4 lanes (matching the test EMBED_DIM=4). Connecting multi-core to the transformer would require scaling to larger embedding dimensions.

---

### Q: Why Pre-LayerNorm instead of Post-LayerNorm?

**A:** Pre-LayerNorm (normalizing before attention and FFN, rather than after) is the standard in modern transformer implementations like GPT-2 and LLaMA. It provides more stable gradients during training and more predictable value ranges during inference. Since we use fixed-point arithmetic with limited range (±127), stable value ranges are critical to prevent overflow.

---

## 🔹 3. TECHNICAL DEEP-DIVE QUESTIONS

### Q: Why Q8.8 fixed-point instead of floating-point?

**A:** Fixed-point has three advantages for hardware inference:
1. **Simpler multiplier** — A 16×16 integer multiply is much smaller in gates than a float32 FPU
2. **Half the memory** — 2 bytes vs 4 bytes per weight, so 2× more weights fit in SRAM
3. **Deterministic** — No rounding mode surprises, no denormals, no NaN handling

The tradeoff is precision: Q8.8 gives ~2.4 decimal digits vs float32's ~7. Research papers (e.g., LLM.int8(), GPTQ) have shown that even 4-bit quantization maintains reasonable LLM accuracy, so 8-bit is well within the acceptable range for inference.

| Property | Float32 | Q8.8 (Ours) |
|----------|:-------:|:-----------:|
| Total bits | 32 | 16 |
| Memory/weight | 4 bytes | 2 bytes |
| Value range | ±3.4×10³⁸ | ±127.996 |
| Precision | ~7 digits | ~2.4 digits |
| Multiplier complexity | High | Low |

---

### Q: How does zero-skip detection work in hardware?

**A:** It's a single combinational NOR gate applied to all 8 bits of the weight value. If all 8 bits are zero, the gate outputs 1, and the ALU stage skips the multiplication entirely — producing zero without consuming any multiplier resources.

This is not a clock cycle cost — the NOR evaluates within nanoseconds (well within a clock period). It's purely combinational logic that sits at pipeline stage 3 (ZERO_CHECK).

**Why zeros are common:**
- ReLU activation outputs zero for all negative inputs (typically 30-50% of values)
- Small weights round to zero after quantization

**Measured zero-skips:**

| Configuration | Zero-Skips | Source |
|--------------|:---------:|--------|
| 1-layer, synthetic weights | 42 | accelerated_gpt2_engine_tb |
| 2-layer, synthetic weights | 132 | multi_layer_test |
| 1-layer, real GPT-2 weights | 32 | real_weight_test |
| Multi-core (128-wide) | 96/256 (37.5%) | gpu_multicore_tb |

---

### Q: How does your softmax work without floating-point?

**A:** We use a 256-entry lookup table (LUT) where each entry is precomputed as:

```
LUT[k] = round(255 × exp(-k/64))
```

All 256 values were generated by Python `math.exp()` — it's a mathematical constant, the same for every model and deployment.

The softmax procedure:
1. Find the maximum score among all attention scores
2. Compute difference: `diff = max_score - score[i]` (always ≥ 0)
3. Look up `exp_value = LUT[diff]` (bounded 0-255)
4. Sum all exp_values → denominator
5. Normalize: `probability[i] = exp_value[i] * 255 / sum`

Max rounding error: ±1 out of 255 (±0.4%).

**We replaced an earlier linear approximation** (`exp(x) ≈ 255 + x×89/256`) which had 5-80% errors, especially at distribution tails.

---

### Q: What is the AXI4-Lite interface for?

**A:** It makes the GPU a standard SoC peripheral. Any CPU or DMA controller that speaks AXI4-Lite can load model weights into the GPU's on-chip SRAM, start inference with a control register write, poll a status register, and read back results.

**Register map:**

| Address | Type | Name | Description |
|---------|:----:|------|-------------|
| 0x0000–0x0FFF | R/W | Weight Memory | 4KB of 8-bit weights |
| 0x1000 | Write | Control | Bit 0: Start inference |
| 0x1004 | Read | Status | Bit 0: Busy, Bit 1: Done |
| 0x1008 | Read | Weight Count | Number loaded |
| 0x100C | Read | Zero-Skip Count | Total during inference |

This was verified with write/readback tests (0xDEADBEEF, 0x12345678) — all passed.

---

### Q: What is the embedding dimension and vocabulary size?

**A:** The current test configuration uses **EMBED_DIM=4** and **VOCAB_SIZE=8**. These are small values chosen for simulation feasibility (Icarus Verilog is an interpreted simulator, so larger values would take hours). The architecture is fully parameterized — `EMBED_DIM`, `VOCAB_SIZE`, `NUM_LAYERS`, and `MAX_SEQ` are all Verilog parameters that can be changed at instantiation time.

Full GPT-2 Small dimensions would be: EMBED_DIM=768, VOCAB_SIZE=50257, NUM_LAYERS=12, MAX_SEQ=1024.

---

### Q: How do you load real model weights?

**A:** We wrote a Python script (`extract_gpt2_weights.py`) that:
1. Loads GPT-2 Small from HuggingFace Transformers
2. Extracts specific weight matrices (attention QKV, FFN, embeddings)
3. Quantizes each float32 value to Q8.8 fixed-point (multiply by 256, round to int16)
4. Writes values as hex strings to `.hex` files
5. Verilog testbench loads these via `$readmemh`

This is the same approach real ASIC/FPGA teams use for weight loading during simulation.

---

## 🔹 4. SIMULATION & VERIFICATION QUESTIONS

### Q: How do you verify this works?

**A:** We have 6 comprehensive test suites, all passing:

1. **Multi-Core Pipeline** — 4 cores × 32 lanes, 256 products, 96 zero-skips, PASSED ✅
2. **Accelerated Attention** — KV cache, 2 tokens, verified input/output values, PASSED ✅
3. **Full GPT-2 (1 Layer)** — 3 autoregressive tokens, 1070 cycles, PASSED ✅
4. **Multi-Layer (2 Layers)** — 4 autoregressive tokens, 2614 cycles, PASSED ✅
5. **Real GPT-2 Weights** — 117M model weights quantized, 4 tokens, PASSED ✅
6. **AXI4-Lite Interface** — Write/readback, control/status registers, ALL PASSED ✅

Every `[PASS]`/`[FAIL]` marker is printed by the testbench itself and counted automatically by our test runner script.

---

### Q: What simulator do you use?

**A:** Icarus Verilog (iverilog) — an open-source, IEEE 1364-2005 compliant Verilog simulator. It compiles Verilog to an intermediate form, then `vvp` executes it. We chose it because it's free, widely used in academia, and supports all the Verilog constructs we need.

---

### Q: How do I run the simulations?

**A:**

```powershell
cd d:\Projects\BitbyBit\custom_gpu_project
.\scripts\run_all_tests.ps1
```

This script:
1. Compiles each test with `iverilog`
2. Runs each with `vvp`
3. Counts `[PASS]` and `[FAIL]` markers in output
4. Prints a summary table with status for all modules

---

### Q: What does "autoregressive" mean in your test results?

**A:** Autoregressive generation means each predicted token is fed back as input for the next prediction. In our 1-layer test:
- Token 0: input=1 → predict=0
- Token 1: input=0 (the prediction) → predict=3
- Token 2: input=3 (the prediction) → predict=3

This is exactly how GPT-2 generates text in practice — one token at a time, feeding each output back as the next input.

---

### Q: Why do the 2-layer results show "out=0" for every token?

**A:** With synthetic weights and only EMBED_DIM=4, the 2-layer configuration produces logit distributions where token 0 consistently wins the argmax. This doesn't mean it's broken — it means the synthetic weight pattern, after two layers of transformation, produces a distribution heavily favoring token 0. The test verifies the pipeline mechanics (tokens flow through 2 layers, KV cache grows correctly, zero-skips accumulate). With real weights and larger dimensions, the outputs would be more varied.

---

## 🔹 5. PERFORMANCE & COMPARISON QUESTIONS

### Q: How fast is it?

**A:** Verified performance (from simulation):

| Metric | Value | Source |
|--------|:-----:|--------|
| Token latency (1 layer) | 328 cycles | accelerated_gpt2_engine_tb |
| Token latency (2 layers) | 628 cycles | multi_layer_test |
| Multi-core throughput | 128 products/cycle | gpu_multicore_tb |
| Zero-skips per token (synthetic) | 42 | accelerated_gpt2_engine_tb |
| Zero-skips per token (real wts) | 32 | real_weight_test |

**Projected (NOT verified):** At a hypothetical 100 MHz clock, this would give ~305,000 tokens/sec for 1 layer. But we have NOT done FPGA synthesis or timing analysis, so the actual achievable clock frequency is unknown.

---

### Q: How does this compare to a real GPU?

**A:** This is **not a fair comparison** and we don't make one. An NVIDIA A100 or H100 has billions of transistors, thousands of CUDA cores, HBM memory, and decades of engineering behind it. Our project is:
- A learning/research exercise
- Simulation-only (no actual hardware)
- Running a tiny slice of GPT-2 (EMBED_DIM=4 vs 768)

The value is in the **architecture design** — showing how you'd build an LLM accelerator from scratch — not in competing with production silicon.

---

### Q: What would it take to run on a real FPGA?

**A:** Several steps:
1. **Synthesis** — Run through Vivado/Quartus to generate a netlist
2. **Timing analysis** — Determine achievable clock frequency
3. **Memory scaling** — The full GPT-2 (117M params × 2 bytes = ~234MB) vastly exceeds on-chip SRAM; would need external DRAM via AXI
4. **Dimension scaling** — Increase EMBED_DIM from 4 to 768, NUM_LAYERS from 1-2 to 12
5. **Weight loading** — Replace `$readmemh` with actual AXI DMA transfers
6. **Testing** — Verify timing closure, functional correctness on hardware

This is significant engineering work but the RTL architecture is designed with FPGA deployment in mind (no exotic constructs, parameterized, synchronous design).

---

### Q: What's the power consumption?

**A:** We don't know. Power analysis requires synthesis results (gate count, toggle rates) which we haven't produced. The zero-skip optimization would reduce dynamic power by avoiding unnecessary multiplier switching, but we can't quantify this without synthesis data.

---

## 🔹 6. DESIGN DECISION QUESTIONS

### Q: Why ReLU instead of GELU in the FFN?

**A:** ReLU (`max(0, x)`) is trivial in hardware — literally one comparator and a mux. GELU requires computing `x × Φ(x)` where Φ is the Gaussian CDF — requiring either another LUT or polynomial approximation. More importantly, ReLU produces exact zeros for all negative inputs, which directly feeds our zero-skip optimization. GELU produces near-zero but non-zero values, reducing zero-skip opportunities.

We do have a `gelu_activation.v` module implemented (using a piecewise approximation), but the accelerated transformer block uses ReLU for the practical benefits.

---

### Q: Why not use a systolic array like Google's TPU?

**A:** We actually built one (`systolic_array.v`). But systolic arrays are optimized for large matrix multiplications with predictable data flow. At EMBED_DIM=4, the overhead of setting up a systolic array exceeds its benefits. The pipelined approach with parallel lanes gives us flexibility — we can change LANES without restructuring the pipeline. For larger dimensions, a systolic array would likely be more efficient, and the module exists for future integration.

---

### Q: Why AXI4-**Lite** and not full AXI4?

**A:** AXI4-Lite is simpler — no burst transfers, no write strobes, no variable-length transactions. For loading model weights before inference, we don't need burst performance. The simplicity means fewer gates, less verification effort, and easier integration into any SoC. If we ever needed high-bandwidth weight streaming, we'd upgrade to full AXI4 with burst support.

---

### Q: What are residual connections and why do they matter?

**A:** Residual connections add the original input back to the output of each sub-layer:

```
output = sublayer(input) + input
```

This serves two purposes:
1. **Gradient flow** — During training, gradients can flow directly through the skip connection, preventing vanishing gradients
2. **Information preservation** — The original signal isn't destroyed by each layer; layers learn to add refinements rather than completely transform the data

In our fixed-point implementation, this is literally one addition per element — very cheap in hardware.

---

## 🔹 7. BUG & DEBUGGING QUESTIONS

### Q: What bugs did you find and fix?

**A:** 7 critical bugs:

| # | Bug | Impact | Fix |
|:-:|-----|--------|-----|
| 1 | Accumulator race condition | Multi-core produced wrong results | Changed to blocking assignment with sequential accumulation |
| 2 | Dequantizer truncation | Only using 4 of 8 weight bits | Changed to use full 8-bit weight value |
| 3 | Fake attention | Attention output = input (no real scoring) | Rewrote with Q·K^T computation and KV cache |
| 4 | Softmax overflow | Normalization overflowed to 0 | Changed scaling factor, added clamp |
| 5 | Disconnected pipeline | GPU core existed but FFN didn't use it | Rewrote FFN to instantiate and use gpu_core |
| 6 | No pipeline drain | Read results before pipeline flushed | Added 6-cycle DRAIN wait states |
| 7 | Wrong LUT values | Manually typed with 5-80% errors | Regenerated all 256 from Python math.exp() |

---

### Q: How did you find the "fake attention" bug?

**A:** During code review, we noticed that the original `attention_unit.v` simply set `output = V` — meaning no attention computation actually happened. Query-Key scoring was never performed. We replaced it with `accelerated_attention.v` which performs proper Q·Kᵀ dot products, applies softmax via the LUT, and produces weighted sums of Value vectors.

---

### Q: What is the "pipeline drain" bug?

**A:** The GPU core pipeline has 5 stages. After we feed the last data into stage 1 (FETCH), the result doesn't appear at stage 5 (WRITEBACK) until 5 cycles later. The original FFN code read the accumulator immediately after feeding — before the pipeline had flushed — getting a partial/wrong result. The fix was adding explicit DRAIN states that wait 6 cycles (5 pipeline stages + 1 safety margin) before reading the final accumulator value.

---

### Q: What was the LUT error bug?

**A:** The original `exp_lut_256.v` had manually typed values that were wrong by 5% to 80%. For example, where `exp(-1.0) = 0.3679` should give LUT value 94, the manually typed value might have been 50 or 120. We regenerated all 256 entries programmatically from Python's `math.exp()` function, guaranteeing mathematical correctness with only ±1 integer rounding error.

---

## 🔹 8. ORIGINALITY & CONTRIBUTION QUESTIONS

### Q: What exactly did YOU build vs reuse?

**A:** **Everything is original.** All 28 Verilog modules were written from scratch. No IP cores, no vendor libraries, no generated code. The only external tools used are:
- **Icarus Verilog** — open-source simulator (tool, not code we wrote)
- **HuggingFace Transformers** — Python library to extract weights (not our code, but we wrote the extraction script)
- **Python** — for weight extraction and LUT generation scripts

The RTL design, testbenches, and system architecture are entirely our own work.

---

### Q: What's novel about this project?

**A:** Several things:

1. **Hardware-level zero-skip detection** integrated into the compute pipeline — not just sparsity at the software level, but the hardware itself detects and skips zero multiplications
2. **LUT-based softmax** in fixed-point — replacing the typical floating-point exp() with a precomputed table that adds only 256 bytes of ROM
3. **End-to-end LLM hardware** — most student projects cover individual components; we have the full pipeline from token input to token output, including attention with KV cache
4. **Real model weights** — we actually loaded quantized GPT-2 weights and ran them through the hardware
5. **Bug discovery and honesty** — we documented 7 critical bugs we found and fixed, demonstrating engineering rigor

---

### Q: What did you learn from this project?

**A:**
1. **Hardware is unforgiving** — bugs that would be easy to debug in software (like race conditions in accumulators) can produce subtly wrong results that are hard to trace
2. **Pipelining is powerful but tricky** — you have to account for fill/drain latency, which software doesn't have
3. **Fixed-point requires careful range management** — overflow is silent and ruins everything; every operation needs scaling analysis
4. **Verification is half the work** — writing testbenches that catch real bugs (not just smoke tests) was as much effort as writing the RTL
5. **The gap between "it compiles" and "it works" is enormous** — our initial version compiled and synthesized perfectly but produced completely wrong results (fake attention, disconnected pipeline, etc.)

---

## 🔹 9. SCALING & FUTURE QUESTIONS

### Q: Can this scale to full GPT-2?

**A:** The architecture is **parameterized** — EMBED_DIM, VOCAB_SIZE, NUM_LAYERS, MAX_SEQ are all configurable parameters. However, scaling to full GPT-2 (768-dim, 12 layers, 50K vocab) requires:
- **Memory**: 117M params × 2 bytes = ~234MB → far exceeds FPGA on-chip SRAM → need external DDR
- **Compute**: 768-wide lanes or multi-core → FPGA resource limits become relevant
- **Simulation time**: Icarus Verilog is too slow for full-scale simulation → would need Verilator or FPGA emulation

The design supports it architecturally; the constraint is physical resources, not design limitations.

---

### Q: What would you add next?

**A:**
1. **FPGA synthesis** — target a Xilinx Artix-7 or Zynq to get real timing and resource numbers
2. **Multi-head attention** — current implementation is single-head; adding multiple parallel attention heads
3. **External DRAM controller** — AXI4 full burst mode for streaming weights from DDR
4. **Larger test dimensions** — EMBED_DIM=64 or 128 for more realistic intermediate testing
5. **Power analysis** — Estimate dynamic power savings from zero-skip
6. **DMA-based weight loading** — Replace `$readmemh` with a hardware DMA engine

---

### Q: Could this be manufactured as an ASIC?

**A:** In theory, yes. The RTL is synthesizable (no non-synthesizable constructs in the main modules — only testbenches use `$display`, `$readmemh`, etc.). An ASIC implementation would give much higher clock speeds (potentially 500MHz-1GHz) and lower power than FPGA. However, ASIC fabrication is expensive ($100K+ for an older node) and requires extensive verification beyond what we've done.

---

## 🔹 10. TRICKY / GOTCHA QUESTIONS

### Q: Your "128-wide parallel processing" — is that actually used in inference?

**A:** **No**, and we're transparent about this. The 128-wide (4 cores × 32 lanes) configuration was tested in a **standalone benchmark** (`gpu_multicore_tb`). The actual transformer inference uses 1 core × 4 lanes (matching EMBED_DIM=4). The multi-core architecture is verified as working, but integrating it into the transformer at scale is future work.

---

### Q: Your projected tokens/second numbers seem high. Are they real?

**A:** **They are NOT verified.** The projected 305,000 tokens/sec assumes 100 MHz clock and 1-layer transformer with EMBED_DIM=4. We have NOT done FPGA synthesis, timing analysis, or any real-world measurement. The only verified numbers are cycle counts from simulation. We clearly label these as "projected" and "not verified" on the website.

---

### Q: With EMBED_DIM=4, how meaningful are your results?

**A:** The results are meaningful for **architectural verification** — proving that tokens flow correctly through embedding → attention → FFN → argmax, that KV cache grows correctly, that pipeline drain works, and that zero-skip detection functions. They are NOT meaningful for **inference quality** — a 4-dimensional embedding can't capture real language semantics. The architecture is parameterized and scales to larger dimensions without redesign.

---

### Q: Why does your test output always pick the same token?

**A:** In the 2-layer test with synthetic weights and small dimensions, the weight patterns consistently drive the logit distribution toward token 0. This is expected behavior — the test validates pipeline mechanics, not language modeling quality. In the 1-layer test with different weights, we see varied outputs (0, 3, 3), and with real GPT-2 weights we see (0, 1, 1, 1) — showing the system does produce different outputs based on different weight configurations.

---

### Q: What happens if a value overflows Q8.8 range?

**A:** It wraps around silently (standard Verilog behavior for signed integers). This is a known limitation. Layer normalization helps mitigate this by keeping values near zero, and the limited range of Q8.8 (±127) is usually sufficient for inference values. If overflow were a critical concern, we could add saturation logic (clamp to ±127 instead of wrapping), but this adds hardware cost per operation.

---

### Q: Do you support multi-head attention?

**A:** Not yet. Our current attention implementation is single-head. Multi-head attention would require splitting Q, K, V into multiple heads (each with dimension `EMBED_DIM / NUM_HEADS`), processing each head independently through the attention mechanism, and concatenating the results. The attention module's architecture supports this — we'd instantiate multiple copies with different weight slices — but it's not implemented in the current version.

---

### Q: How does your design compare to published accelerator papers?

**A:** Academic LLM accelerators (FlexGen, SpAtten, etc.) operate at much larger scales and include optimizations we haven't implemented (tiling, operator fusion, memory hierarchy). Our project is closer in spirit to educational processor designs (like RISC-V cores in university courses) — the value is in demonstrating understanding of the full architecture, not in competing with research papers.

---

## 🔹 11. PRESENTATION-SPECIFIC QUESTIONS

### Q: Can you show me a live demo?

**A:** Yes — we can run the full test suite live:

```powershell
cd d:\Projects\BitbyBit\custom_gpu_project
.\scripts\run_all_tests.ps1
```

This compiles all modules, runs all 6 test suites, and shows a pass/fail summary in ~30 seconds. We can also show the website at [https://bitbybit-gpu.vercel.app](https://bitbybit-gpu.vercel.app).

---

### Q: Show me the source code for [specific module].

**A:** All RTL source is in `custom_gpu_project/rtl/` organized by category:

```
rtl/
├── primitives/     ← gpu_core.v, gpu_multicore.v, zero_detect_mult.v, etc.
├── transformer/    ← accelerated_attention.v, layer_norm.v, etc.
├── compute/        ← exp_lut_256.v, mac_unit.v, softmax_unit.v, etc.
├── gpt2/           ← accelerated_gpt2_engine.v, embedding_lookup.v
└── memory/         ← axi_weight_memory.v, sparse_memory_ctrl.v
```

All 28 modules are readable, commented, and parameterized.

---

### Q: What's in the testbenches?

**A:** Each testbench:
1. Instantiates the module under test with specific parameters
2. Generates clock and reset signals
3. Provides stimulus (input tokens, weights, control signals)
4. Checks outputs against expected values
5. Prints `[PASS]` or `[FAIL]` for each check
6. Reports summary statistics (cycles, zero-skips, etc.)

Testbenches are in `custom_gpu_project/tb/` mirroring the RTL directory structure.

---

### Q: How long did this project take?

**A:** This involved:
- Designing 28 Verilog modules
- Writing 6 test suites
- Debugging 7 critical bugs
- Building weight extraction and LUT generation scripts
- Creating a professional website with interactive visualizations
- Documenting all verified and projected claims

The iterative debugging process (discovering fake attention, pipeline drain issues, LUT errors, etc.) was particularly time-consuming and educational.
