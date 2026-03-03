# BitByBit GPU - Hardware Improvement Research Report

## Executive Summary

This report analyzes 10 areas of cutting-edge hardware acceleration research and identifies concrete, incremental improvements for the BitByBit custom GPU accelerator. The top 5 most impactful improvements ranked by effort-to-benefit ratio are:

| Rank | Improvement | Expected Gain | Complexity | Priority |
|------|------------|---------------|------------|----------|
| 1 | Online/Streaming Softmax (single-pass) | 2× softmax throughput, eliminates 2-pass bottleneck | Low | P0 |
| 2 | 2:4 Structured Sparsity in Systolic Array | 2× effective MAC throughput at 50% sparsity | Medium | P0 |
| 3 | Tiled FlashAttention-style Memory Access | 3-5× attention speedup, O(n) memory | Medium | P0 |
| 4 | Sub-4-bit Quantization (INT2/Ternary) | 2-4× weight compression, 2× MAC density | Medium | P1 |
| 5 | KV Cache Compression & Paging | 4-8× KV cache memory reduction | Medium | P1 |

These five improvements alone could yield a **4-8× overall inference throughput improvement** while staying within the existing architecture's framework.

---

## Detailed Analysis

### 1. Tiled/FlashAttention-Style Attention (Sparse Attention Mechanisms)

**Source**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (arXiv:2205.14135, 2022); "FlashAttention-2" (arXiv:2307.08691, 2023)

**Description**: FlashAttention reformulates the attention computation using *tiling* — processing Q, K, V in blocks that fit in on-chip SRAM, computing partial softmax results per tile, and accumulating with online softmax correction. This avoids materializing the full N×N attention matrix in memory, reducing memory from O(N²) to O(N) and significantly reducing data movement between scratchpad and external memory. FlashAttention-2 further improves by optimizing work partitioning within tiles for higher utilization.

**Implementation Plan**:
- **Modify `attention_unit.v`**: Restructure to process Q×K^T in tiles of size B_r × B_c (e.g., 4×4 matching the systolic array dimension). Instead of computing the full attention matrix, compute one tile of scores at a time.
- **Add `tiled_attention_ctrl.v`**: New FSM that sequences tile loads: for each Q block, iterate over K/V blocks, accumulating partial softmax numerator/denominator using the online softmax algorithm (see Section 3 below).
- **Modify `scratchpad.v`**: Add double-buffering support — while one tile is being computed, the next tile's K/V data is being loaded via DMA. Requires partitioning the existing 4096×16-bit scratchpad into 2 banks (or expanding slightly).
- **Estimated effort**: ~2-3 weeks for a single-head implementation. The key challenge is the online softmax correction factor that must be applied when accumulating partial results across tiles.

**Expected Gain**: 3-5× attention speedup for sequence lengths >64. Memory usage drops from O(N²) to O(N) — critical since the current 8KB scratchpad limits sequence length to ~64 tokens with a full attention matrix. With tiling, sequences of 256-1024 become feasible.

**Complexity**: Medium
**Priority**: P0

---

### 2. Mixed-Precision & Sub-4-bit Quantization

**Source**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (arXiv:2210.17323, ICLR 2023); Lin et al., "AWQ: Activation-aware Weight Quantization" (arXiv:2306.00978, MLSys 2024 Best Paper); Ma et al., "BitNet b1.58" (arXiv:2402.17764, 2024); Huang et al., "BiLLM" (arXiv:2402.04291, 2024)

**Description**: GPTQ uses approximate second-order information (Hessian) to quantize weights to 3-4 bits with minimal accuracy loss, achieving 3.25-4.5× speedup. AWQ identifies that only ~1% of weights are "salient" (determined by activation magnitudes) and applies per-channel scaling before uniform quantization, achieving hardware-friendly quantization without mixed precision. BitNet b1.58 goes further: ternary {-1, 0, 1} weights replace all multiplications with additions/subtractions, matching FP16 performance at the same model size while drastically reducing energy. BiLLM achieves 1.08-bit post-training quantization through binary residual approximation.

**Implementation Plan**:

#### Phase A: INT3 Support (extend existing Q4 path)
- **Modify `block_dequantizer.v`**: Add INT3 dequantization mode. Each INT3 weight occupies 3 bits; pack 5 weights per 16-bit word with 1 bit padding. Dequantize as: `value = (int3_raw - 4) * scale + zero_point`.
- **Modify `variable_precision_alu` (in systolic PE)**: Add INT3×INT8 multiply mode — treat INT3 as sign-extended to 4 bits and reuse the INT4×INT8 datapath.
- **Estimated effort**: ~1 week (minor extension of existing Q4 infrastructure).

#### Phase B: Ternary/Binary Weight Support (BitNet-style)
- **Add `ternary_mac_unit.v`**: For {-1, 0, 1} weights, replace multiplier with a MUX: output = (weight==1) ? activation : (weight==-1) ? -activation : 0. This eliminates the multiplier entirely — pure addition.
- **Modify `systolic_array.v`**: Add a `PRECISION_MODE` port. When in ternary mode, bypass the multiplier in each PE, using only the adder tree. This could double the effective clock rate or halve power consumption.
- **Add `ternary_weight_decoder.v`**: Decode packed ternary weights (2 bits per weight = 8 weights per 16-bit word).
- **Estimated effort**: ~2 weeks. The ternary MAC is dramatically simpler than INT8 — area per PE drops by ~60%.

#### Phase C: AWQ-style Salient Weight Scaling
- **Add `awq_scale_unit.v`**: Per-channel scale factors (one 16-bit scale per channel, stored in a small LUT). Before dequantization, multiply the corresponding activation channel by the inverse scale. This protects salient weights without mixed precision.
- **Estimated effort**: ~1 week. Small area overhead (one extra multiplier + scale LUT).

**Expected Gain**:
- INT3: ~25% additional weight compression over INT4 (1.33× more weights in same memory)
- Ternary: 8× weight compression vs INT8; eliminates multiplier power; 2× effective throughput (since MAC → add)
- AWQ scaling: ~0.5 perplexity improvement at INT4 vs naive quantization

**Complexity**: Medium (Phase A: Low, Phase B: Medium, Phase C: Low)
**Priority**: P1

---

### 3. Efficient Online Softmax Hardware

**Source**: Milakov & Gimelshein, "Online normalizer calculation for softmax" (arXiv:1805.02867, 2018); Dao et al., FlashAttention (arXiv:2205.14135) — Section 3.1 on online softmax

**Description**: Standard safe softmax requires two passes: (1) find max over all elements, (2) compute exp(x_i - max) and sum, (3) normalize. The online softmax algorithm computes the softmax in a single streaming pass by maintaining a running maximum and correction factor. As each new element arrives: update max, apply correction to running sum, accumulate new exp term. This eliminates the need to buffer all attention scores before normalization — critical for tiled attention.

The algorithm maintains running state: `m` (running max), `d` (running denominator), and `acc` (running weighted sum of values). When a new score `s_j` arrives:
```
m_new = max(m, s_j)
d_new = d * exp(m - m_new) + exp(s_j - m_new)
acc_new = acc * exp(m - m_new) + exp(s_j - m_new) * v_j
```

**Implementation Plan**:
- **Replace `softmax_unit.v`** with `online_softmax_unit.v`:
  - **State registers**: `running_max` (16-bit), `running_denom` (16-bit), `running_acc[EMBED_DIM]` (16-bit each).
  - **Datapath**: One comparator (for max update), two `exp_lut_256` lookups (for the correction factor `exp(m_old - m_new)` and `exp(s_j - m_new)`), two multipliers (for correction and accumulation), one adder.
  - **Pipeline**: Can accept one new score per cycle after 2-3 cycle pipeline latency. No buffering of intermediate scores needed.
- **Modify `exp_lut_256.v`**: The existing 256-entry LUT covers a limited range. For online softmax, the correction factor `exp(m_old - m_new)` is always ≤ 1 (since m only increases), so we only need the [−range, 0] domain. Partition the LUT: 128 entries for exp(x) where x ∈ [−8, 0] (for corrections) and 128 entries for x ∈ [0, 8] (for scores).
- **Integration with tiled attention**: The online softmax state persists across tiles. After processing all K tiles for a given Q row, the accumulator contains the correct attention-weighted V output.
- **Estimated effort**: ~1 week. The core logic is simple (comparator + 2 LUT lookups + 2 multiplies + 2 adds); most effort is in the control FSM.

**Expected Gain**: 2× softmax throughput (single-pass vs two-pass). Eliminates the need to store N attention scores before normalizing — reduces scratchpad pressure by N×16 bits per attention head. Enables tiled attention (improvement #1) which is otherwise impossible with two-pass softmax.

**Complexity**: Low
**Priority**: P0 (prerequisite for tiled attention)

---

### 4. Systolic Array Optimizations: Structured Sparsity & Reconfigurable Dataflow

**Source**: Pool & Yu, "Accelerating Sparse Deep Neural Networks" (arXiv:2104.08378, NVIDIA 2021); NVIDIA Developer Blog, "Accelerating Inference with Sparsity Using Ampere and TensorRT" (2021)

**Description**: NVIDIA's Ampere architecture introduced 2:4 structured sparsity in Tensor Cores: in every contiguous block of 4 weights, exactly 2 must be zero. This 50% sparsity pattern is highly regular, requiring only 2 bits of metadata per 4 elements to encode which elements are nonzero. Sparse Tensor Cores skip the zero multiplications, achieving 2× throughput over dense. The key insight is that 2:4 pruning with magnitude-based selection + fine-tuning recovers accuracy to match dense baselines across BERT, ResNet, Transformers, etc.

**Implementation Plan**:

#### Phase A: 2:4 Sparse Systolic Array
- **Add `sparse_pe.v`**: Modified processing element that receives compressed weights (2 nonzero values + 2-bit index mask per group of 4). The PE uses the index mask to select the corresponding activation elements, multiplies only the 2 nonzero weight-activation pairs, and accumulates. Each PE effectively computes 4 outputs in the time of 2 MACs.
- **Add `sparsity_encoder.v`**: Offline preprocessing unit that takes a 4-element weight group, selects the 2 largest by magnitude, stores them plus a 2-bit mask. Format: [val0:16][val1:16][mask:2][pad:14] = 48 bits per group of 4 weights (vs 64 bits dense).
- **Modify `systolic_array.v`**: Add `SPARSE_MODE` input. When active, each PE reads compressed weight format and uses the selection MUX. The array processes 2× more effective weight groups per cycle.
- **Estimated effort**: ~2 weeks. Main challenge is the activation selection MUX within each PE — need to route 4 possible activation values to 2 multipliers based on the 2-bit mask.

#### Phase B: Output-Stationary Mode
- **Add `os_systolic_ctrl.v`**: Alternative control FSM for output-stationary dataflow. In OS mode, each PE accumulates one output element, with weights and activations both flowing through the array. OS is better for attention computation (short inner dimension) while WS is better for FFN (reuse of weights across many inputs).
- **Modify `systolic_array.v`**: Parameterize dataflow. Add a `DATAFLOW_MODE` register (00=WS, 01=OS). MUX the PE interconnect to route data horizontally vs vertically based on mode.
- **Estimated effort**: ~2-3 weeks. Requires re-thinking the PE interconnect, but can share the same multiplier/accumulator hardware.

**Expected Gain**:
- 2:4 sparsity: 2× effective throughput at 50% sparsity with no accuracy loss (proven across many architectures)
- OS dataflow: 20-40% improvement for attention computation (better activation reuse for short sequences)

**Complexity**: Medium
**Priority**: P0 (Phase A), P2 (Phase B)

---

### 5. Memory Hierarchy: PagedAttention & KV Cache Compression

**Source**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arXiv:2309.06180, SOSP 2023); Ge et al., "Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs" (arXiv:2310.01801, ICLR 2024 — FastGen); Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (arXiv:2309.17453, ICLR 2024 — StreamingLLM); Oren et al., "Transformers are Multi-State RNNs — TOVA" (arXiv:2401.06104)

**Description**: PagedAttention (vLLM) borrows virtual memory concepts: KV cache is stored in non-contiguous fixed-size "pages" (blocks), with a page table mapping logical token positions to physical memory locations. This eliminates fragmentation (near-zero waste) and enables KV sharing across requests. StreamingLLM shows that keeping only the initial "attention sink" tokens + a sliding window of recent tokens is sufficient for stable generation. FastGen profiles attention heads to adaptively evict KV entries — heads focused on local context evict long-range KVs, while heads attending to special tokens keep only those.

**Implementation Plan**:

#### Phase A: Paged KV Cache
- **Add `kv_page_table.v`**: Small SRAM table (e.g., 64 entries × 8 bits = 64B) mapping logical token positions to physical page IDs in the scratchpad. Each page stores one token's K and V vectors for one head.
- **Modify `attention_unit.v`**: Replace sequential KV address generation with page table lookup. Instead of `kv_addr = head_offset + token_id * embed_dim`, compute `kv_addr = page_table[token_id] * PAGE_SIZE + offset_within_page`.
- **Add `page_allocator.v`**: Simple free-list allocator for pages. On new token: pop a free page, write K/V. On eviction: push page back to free list.
- **Estimated effort**: ~1-2 weeks. Simple address translation layer.

#### Phase B: Sliding Window + Attention Sink (StreamingLLM)
- **Modify `attention_unit.v`**: Add `WINDOW_SIZE` and `SINK_SIZE` parameters. During attention computation, only iterate over: (1) the first `SINK_SIZE` tokens (attention sinks), and (2) the most recent `WINDOW_SIZE` tokens. Skip all tokens in between.
- **Add eviction logic**: When KV cache is full and a new token arrives, evict the oldest non-sink token by freeing its page.
- **Estimated effort**: ~1 week (simple address range control in the attention loop).

#### Phase C: KV Cache Quantization
- **Add `kv_quantizer.v`**: Quantize KV cache entries from Q8.8 (16-bit) to INT8 or INT4 before storing. Use per-token or per-head scale factors. On read, dequantize back to Q8.8 for attention computation.
- **Estimated effort**: ~1 week. Reuse existing `block_dequantizer.v` infrastructure.

**Expected Gain**:
- Paging: Near-zero memory waste (currently any fragmentation wastes precious scratchpad space)
- Sliding window: Attention complexity drops from O(N) to O(W+S) where W=window size, S=sink count. For 256-token context with W=64, S=4: 3.8× fewer K/V reads
- KV quantization (INT8): 2× more tokens fit in KV cache; with INT4: 4× more tokens
- Combined: 4-8× effective KV cache capacity improvement

**Complexity**: Medium
**Priority**: P1

---

### 6. Activation Compression Between Layers

**Source**: Dryden et al., "Optimizing Data Movement in Transformers" (arXiv:2007.00072, MLSys 2021); Xi et al., "Training Transformers with 4-bit Activations" (arXiv:2306.11987, 2023); Bondarenko et al., "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing" (arXiv:2306.12929, 2023)

**Description**: Inter-layer activations (the residual stream between transformer blocks) are a major data movement bottleneck. Research shows that activations can be quantized to INT8 or even INT4 with minimal quality loss if outliers are handled. Bondarenko et al. identified that attention heads trying to compute "no-ops" push softmax inputs to extreme values, causing outliers. Solutions include clipped softmax and gated attention to suppress outliers, enabling full INT8 activation quantization.

**Implementation Plan**:
- **Add `activation_compressor.v`**: Placed between each transformer block's output and the next block's input. Compresses 16-bit Q8.8 activations to INT8 using per-row (per-token) dynamic scaling:
  1. Find max absolute value across the embedding dimension (single-pass with comparator tree)
  2. Compute scale = max_val / 127
  3. Quantize: `int8_val = round(activation / scale)`
  4. Store scale factor (16-bit) alongside INT8 activations
  - On read (next layer input): `activation = int8_val * scale`
- **Modify inter-block buses**: The flat wire bus between attention→FFN and between transformer blocks currently carries EMBED_DIM × 16-bit = 256 bits (for DIM=16). With INT8 compression: 128 bits + 16-bit scale = 144 bits. This directly addresses weakness #3 (flat buses don't scale) — activation compression cuts bus width by ~44%.
- **Add `clipped_softmax.v`**: Modified softmax that clamps extreme pre-softmax values to a configurable range [−C, C], reducing outliers and enabling better activation quantization downstream.
- **Estimated effort**: ~1-2 weeks. The compressor is a simple scale-and-quantize unit; the main effort is integrating it into the pipeline.

**Expected Gain**: 44% reduction in inter-layer bus width (16-bit → INT8+scale). Directly enables scaling to larger EMBED_DIM (32, 64) on the same wire buses. Reduces scratchpad pressure for storing intermediate activations by ~2×.

**Complexity**: Low-Medium
**Priority**: P1

---

### 7. Speculative Decoding Hardware Support

**Source**: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192, ICML 2023 Oral); Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv:2302.01318, DeepMind 2023); Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (arXiv:2401.10774, 2024)

**Description**: Speculative decoding uses a small "draft" model to generate K candidate tokens quickly, then verifies all K tokens in parallel using the full model in a single forward pass. Using modified rejection sampling, the output distribution exactly matches the target model. This converts K sequential autoregressive steps into ~1 parallel verification step. Medusa simplifies this by adding small prediction heads to the main model itself, avoiding a separate draft model.

**Implementation Plan**:

#### Phase A: Multi-Token Verification Pipeline
- **Add `spec_decode_ctrl.v`**: Controller that manages speculative execution:
  1. Accept K draft tokens (from software or a future draft model)
  2. Construct a causal attention mask for all K tokens (triangular + draft positions)
  3. Run a single forward pass through the transformer with all K tokens
  4. Extract logits for each position, compare with draft token choices
  5. Apply rejection sampling: accept tokens until the first rejection
- **Modify `attention_unit.v`**: Support batch attention where multiple query positions are processed simultaneously. Currently processes one token at a time; extend to K tokens with proper causal masking.
- **Add `rejection_sampler.v`**: Compares target model probability p(x) with draft model probability q(x) for each drafted token. Accepts with probability min(1, p(x)/q(x)). Requires a simple LFSR-based PRNG for the random threshold.

#### Phase B: Medusa-style Prediction Heads
- **Add `medusa_head.v`**: Small linear layer (EMBED_DIM → VOCAB_SIZE) that predicts the next token in parallel with the main model's prediction. Each head predicts token at position t+k for k=1,2,...,K. These heads share the final hidden state and are just extra linear projections.
- **Estimated effort**: Phase A: ~2-3 weeks. Phase B: ~1-2 weeks (reuses existing `linear_layer.v`).

**Expected Gain**: 2-3× token generation throughput. The key insight: verification of K tokens costs approximately the same as generating 1 token (since the forward pass through the model is the bottleneck, and processing K tokens in parallel only marginally increases compute for small K).

**Complexity**: High (Phase A), Medium (Phase B)
**Priority**: P2

---

### 8. Hardware-Aware Neural Architecture Search / Co-Design

**Source**: Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" (arXiv:2303.06865, 2023); Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (arXiv:2305.13245, EMNLP 2023)

**Description**: Rather than pure NAS, the most practical approach for BitByBit is *hardware-aware model co-design*: choosing model architecture parameters that match the hardware's capabilities. GQA (Grouped-Query Attention) reduces KV cache size by sharing K/V heads across multiple query heads (e.g., 4 query heads sharing 1 K/V head), cutting KV memory by 4× with minimal quality loss. FlexGen demonstrates that the optimal compute/memory tradeoff depends heavily on hardware characteristics, and solving a linear program can find the best configuration.

**Implementation Plan**:
- **Add `gqa_attention.v`**: Modified attention unit supporting grouped-query attention. Parameters: `NUM_Q_HEADS`, `NUM_KV_HEADS` where `NUM_KV_HEADS` divides `NUM_Q_HEADS`. During attention, each KV head serves `NUM_Q_HEADS/NUM_KV_HEADS` query heads. This reduces KV cache reads proportionally.
- **Add `hw_config_space.v`**: Parameterizable top-level that allows compile-time selection of: number of heads, head dimension, FFN ratio, number of layers, quantization mode. This enables rapid exploration of different model configurations matched to the hardware.
- **Modify `config_registers.v`**: Add runtime-configurable parameters for GQA grouping, enabling software to select the number of KV groups without resynthesis.
- **Estimated effort**: ~2 weeks. GQA is a straightforward modification to the attention loop's KV indexing.

**Expected Gain**: GQA with 4:1 ratio → 4× KV cache reduction with <1% quality loss. Hardware-aware model selection can improve overall utilization by 20-50% by matching model dimensions to the 4×4 systolic array.

**Complexity**: Medium
**Priority**: P1

---

### 9. Emerging Dataflow Architectures

**Source**: Groq LPU (Language Processing Unit) architecture (groq.com/technology); Dryden et al., "Optimizing Data Movement in Transformers" (arXiv:2007.00072, MLSys 2021)

**Description**: Groq's LPU represents a deterministic spatial architecture where computation is laid out in space rather than time — each operation has a fixed physical location on the chip, and data flows between them on a predetermined schedule. This eliminates the overhead of instruction fetch/decode and memory hierarchy management. For the BitByBit GPU, the applicable insight is: data movement dominates compute in LLM inference (memory-bound workload), so optimizing the dataflow is more impactful than adding more MACs. Key principles: (1) eliminate unnecessary data movement, (2) use spatial pipelining (overlap computation of different stages), (3) keep data on-chip as long as possible.

**Implementation Plan**:

#### Phase A: Inter-Stage Pipeline Registers (addresses weakness #2)
- **Modify `accelerated_transformer_block.v`**: Insert pipeline register stages between attention output and FFN input, and between FFN output and the residual addition. This allows attention for token t+1 to begin while FFN processes token t.
  - Add `pipe_reg_attn_ffn.v`: EMBED_DIM × 16-bit register bank with valid/ready handshake.
- **Estimated effort**: ~1 week. Standard pipeline register insertion.

#### Phase B: Spatial Activation Pipeline
- **Add `activation_fifo.v`**: Small FIFOs (depth 2-4) between each major compute stage (embedding → attn → ffn → layernorm → next block). Replace the current flat wire buses with these registered FIFOs, enabling spatial pipelining where each stage operates independently.
- **Modify `gpt2_engine.v`**: Add pipeline control: each transformer block can process a different token simultaneously, giving up to N× throughput improvement (where N = number of pipeline stages).
- **Estimated effort**: ~2-3 weeks. Requires careful handling of the residual connections across pipeline stages.

**Expected Gain**: 
- Phase A: 30-50% throughput improvement from overlapping attention and FFN
- Phase B: Up to 4× throughput for multi-token inference (one token per pipeline stage)

**Complexity**: Low (Phase A), High (Phase B)
**Priority**: P0 (Phase A), P2 (Phase B)

---

### 10. Power/Area Optimization

**Source**: NVIDIA Ampere 2:4 sparsity (arXiv:2104.08378); BitNet b1.58 (arXiv:2402.17764); general VLSI design principles

**Description**: Power optimization in custom accelerators comes from three main strategies: (1) **Clock gating** — disable clock to unused modules (e.g., FFN units during attention, attention units during FFN), (2) **Operand isolation** — gate inputs to multipliers when they're inactive to prevent dynamic power from toggling, (3) **Approximate computing** — use lower-precision or approximate arithmetic for non-critical paths (e.g., softmax normalization, GELU activation). BitNet b1.58's ternary weights eliminate multipliers entirely, reducing energy per operation by ~10× compared to INT8 multiplication.

**Implementation Plan**:

#### Phase A: Clock Gating
- **Add `clock_gate_cell.v`**: Standard latch-based clock gating cell: `gated_clk = clk & (enable_latched)`. Use a latch on the enable signal to prevent glitches.
- **Modify `accelerated_transformer_block.v`**: Add clock gating for:
  - Systolic array clock: gated off during softmax, layernorm, GELU computation
  - Softmax unit clock: gated off during matmul and FFN phases
  - FFN block clock: gated off during attention phase
  - KV cache memory: gated off during FFN phase
- **Add `power_state_fsm.v`**: Simple FSM that tracks the current computation phase (ATTN_QK, ATTN_SOFTMAX, ATTN_SV, FFN_LINEAR1, FFN_GELU, FFN_LINEAR2, LAYERNORM) and generates clock enable signals.
- **Estimated effort**: ~1 week. Standard power optimization technique with well-known implementation.

#### Phase B: Operand Isolation
- **Modify `mac_unit.v`**: AND-gate the multiplier inputs with an `active` signal. When inactive, both inputs are forced to 0, preventing dynamic power from input toggling.
- **Modify `systolic_array.v`**: Add `pe_active[3:0][3:0]` mask. When processing a matrix smaller than 4×4, deactivate unused PEs.
- **Estimated effort**: ~1 week. Simple gating logic at multiplier inputs.

#### Phase C: Approximate GELU/Softmax
- **Modify `gelu_lut_256.v`**: For EMBED_DIM positions far from the critical region (|x| > 3), use a simpler approximation: GELU(x) ≈ x for x > 3, GELU(x) ≈ 0 for x < −3. Only use the full LUT for |x| ≤ 3. This reduces LUT reads by ~50% for typical activations.
- **Estimated effort**: ~0.5 weeks.

**Expected Gain**:
- Clock gating: 30-50% dynamic power reduction (each unit is active only ~25-40% of the time)
- Operand isolation: Additional 10-15% power reduction on multiplier-heavy modules
- Approximate GELU: 5-10% reduction in GELU computation time
- Combined: **40-60% total dynamic power reduction**

**Complexity**: Low
**Priority**: P0 (Phase A), P1 (Phase B), P2 (Phase C)

---

## Cross-Cutting Concerns

### Addressing Known Weaknesses

| Weakness | Addressed By | Improvement(s) |
|----------|-------------|-----------------|
| #1: Q8.8 precision ceiling | §2 (Sub-4-bit quant) + §6 (Activation compression) | More precision where needed, less where not |
| #2: No pipeline registers attn↔FFN | §9 Phase A (Pipeline registers) | 30-50% throughput boost |
| #3: Flat wire buses don't scale | §6 (Activation compression) + §9 Phase B (FIFOs) | 44% bus width reduction; registered buses |
| #4: Single 4×4 systolic array (16 MACs) | §4 Phase A (2:4 sparsity = effective 32 MACs) | 2× effective throughput |
| #5: Softmax LUT limited range | §3 (Online softmax) | Streaming computation, better range utilization |
| #6: No activation checkpointing | §6 (Activation compression) | Compressed inter-layer activations |
| #7: No speculative decoding | §7 (Spec decode hardware) | 2-3× token throughput |
| #8: Sequential layer norm | §9 Phase A (Pipeline) | Overlapped with next stage |

---

## Recommended Roadmap

### Phase 1: Foundations (Weeks 1-4) — High Impact, Low Risk

| Week | Task | Builds On | Addresses |
|------|------|-----------|-----------|
| 1 | Online softmax unit (`online_softmax_unit.v`) | Existing `exp_lut_256.v` | Weakness #5, enables §1 |
| 1-2 | Clock gating for all major modules | Existing modules | 40%+ power reduction |
| 2 | Inter-stage pipeline registers (attn↔FFN) | Existing `accelerated_transformer_block.v` | Weakness #2 |
| 2-3 | INT3 dequantization mode | Existing `block_dequantizer.v`, Q4 path | Weakness #1 |
| 3-4 | Tiled attention controller | New online softmax, existing systolic array | Weakness #5, enables long sequences |

**Phase 1 Expected Result**: ~2-3× overall throughput improvement, 40% power reduction, longer sequence support.

### Phase 2: Compute Density (Weeks 5-10) — Medium Effort, High Reward

| Week | Task | Builds On | Addresses |
|------|------|-----------|-----------|
| 5-6 | 2:4 structured sparsity in systolic PEs | Existing `systolic_array.v` | Weakness #4 |
| 6-7 | Ternary weight MAC (BitNet-style) | Existing PE infrastructure | Weakness #1, #4 |
| 7-8 | Activation compressor between layers | Existing inter-block buses | Weakness #3, #6 |
| 8-9 | KV cache paging + sliding window | Existing `attention_unit.v`, `scratchpad.v` | Memory capacity |
| 9-10 | GQA support in attention unit | Existing `attention_unit.v` | KV cache reduction |

**Phase 2 Expected Result**: Additional 2-3× throughput (cumulative 4-8×), scalable to larger models.

### Phase 3: Advanced Features (Weeks 11-16) — Higher Effort, Strategic Value

| Week | Task | Builds On | Addresses |
|------|------|-----------|-----------|
| 11-12 | Speculative decode verification pipeline | Phase 1 tiled attention | Weakness #7 |
| 12-13 | KV cache quantization (INT4) | Phase 2 KV paging | Memory capacity |
| 13-14 | Output-stationary systolic mode | Phase 2 sparse systolic | Attention efficiency |
| 14-16 | Spatial activation pipeline (full inter-block pipelining) | Phase 1 pipeline registers | Maximum throughput |

**Phase 3 Expected Result**: Additional 2× throughput (cumulative 8-16×), speculative decoding support, production-ready memory management.

---

## References

1. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — arXiv:2205.14135
2. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — arXiv:2307.08691
3. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" — arXiv:2210.17323
4. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" — arXiv:2306.00978
5. Kim et al., "SqueezeLLM: Dense-and-Sparse Quantization" — arXiv:2306.07629
6. Darvish Rouhani et al., "Microscaling Data Formats for Deep Learning" — arXiv:2310.10537
7. Ma et al., "BitNet: Scaling 1-bit Transformers for Large Language Models" — arXiv:2310.11453
8. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (BitNet b1.58) — arXiv:2402.17764
9. Huang et al., "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs" — arXiv:2402.04291
10. Milakov & Gimelshein, "Online normalizer calculation for softmax" — arXiv:1805.02867
11. Pool & Yu, "Accelerating Sparse Deep Neural Networks" (NVIDIA 2:4 sparsity) — arXiv:2104.08378
12. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM) — arXiv:2309.06180
13. Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM) — arXiv:2309.17453
14. Ge et al., "Model Tells You What to Discard: Adaptive KV Cache Compression" (FastGen) — arXiv:2310.01801
15. Oren et al., "Transformers are Multi-State RNNs" (TOVA) — arXiv:2401.06104
16. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" — arXiv:2211.17192
17. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" — arXiv:2302.01318
18. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" — arXiv:2401.10774
19. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — arXiv:2305.13245
20. Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" — arXiv:2303.06865
21. Dryden et al., "Optimizing Data Movement in Transformers" — arXiv:2007.00072
22. Xi et al., "Training Transformers with 4-bit Activations" — arXiv:2306.11987
23. Bondarenko et al., "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing" — arXiv:2306.12929
24. Frantar & Alistarh, "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression" — arXiv:2306.03078
25. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NF4 data type) — arXiv:2305.14314
