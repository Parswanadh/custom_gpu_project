# IMPLEMENTATION GUIDE: BITBYBIT OPTIMIZATION PHASES

## PHASE 1: WEEK 1 FOUNDATION (Target: 115 cycles, -11% speedup)

### Task 1.1: Fix Softmax exp LUT (PRIORITY: IMMEDIATE, 5 MINUTES)
**File:** \tl/compute/parallel_softmax.v\  
**Lines:** 62-88 (delete fast_exp function)  
**Effort:** 5 minutes

**What to do:**
1. Delete the fast_exp() function entirely (lines 62-88)
2. Line 105 currently calls: \xp_val[i] = fast_exp(diff_calc);\
3. Change it to use existing LUT outputs (already generated lines 54-60)
4. The exp_lut_256 instances already exist!

**Verification:**
\\\ash
# Test: softmax output should now match IEEE exp exactly (within quantization)
run pytest tests/softmax_accuracy_test.py
expected: all outputs match exp_lut_256 reference
\\\

---

### Task 1.2: Pipeline Stage Handoff - FSM Refactor (3-5 days)
**Files:**
- \tl/integration/optimized_transformer_layer.v\ (main refactor)
- Create: \tl/integration/skid_buffer.v\ (new module)

**Current Problem:**
- Lines 268-274: Define 13 states with explicit wait states (S_ROPE_W, S_GQA_W, etc.)
- Lines 289-365: FSM waits for prior stage .done before advancing
- Result: 40 cycles of idle time per token

**Step-by-step implementation:**

**Step 1: Create skid_buffer module**
`erilog
// NEW FILE: rtl/integration/skid_buffer.v (simple 1-entry FIFO)
module skid_buffer #(parameter DATA_WIDTH = 128) (
    input  clk, rst,
    input  valid_in,
    input  [DATA_WIDTH-1:0] data_in,
    input  ready_from_downstream,
    
    output ready_for_upstream,        // Can accept new data?
    output valid_out,                 // Data valid for downstream?
    output [DATA_WIDTH-1:0] data_out
);

    reg [DATA_WIDTH-1:0] buffer_data;
    reg buffer_valid;
    
    // Skid logic: always ready if buffer empty, else only if downstream pulls
    assign ready_for_upstream = !buffer_valid || ready_from_downstream;
    assign valid_out = buffer_valid;
    assign data_out = buffer_data;
    
    always @(posedge clk) begin
        if (rst) begin
            buffer_valid <= 1'b0;
            buffer_data <= 0;
        end else if (ready_from_downstream) begin
            // Downstream pulls data
            if (valid_in) begin
                // New data available: pass through or buffer
                buffer_data <= data_in;
                buffer_valid <= 1'b1;
            end else begin
                buffer_valid <= 1'b0;
            end
        end else if (valid_in && !buffer_valid) begin
            // Buffer is empty, accept upstream
            buffer_data <= data_in;
            buffer_valid <= 1'b1;
        end
    end
endmodule
`

**Step 2: Modify optimized_transformer_layer FSM**

Remove wait states:
`erilog
// BEFORE (lines 268-274):
localparam S_IDLE       = 4'd0;
localparam S_ROPE       = 4'd1;
localparam S_ROPE_W     = 4'd2;   ← DELETE THIS
localparam S_GQA        = 4'd3;   ← RENUMBER
localparam S_GQA_W      = 4'd4;   ← DELETE THIS
...

// AFTER: Remove all _W states
localparam S_IDLE       = 4'd0;
localparam S_ROPE       = 4'd1;
localparam S_GQA        = 4'd2;   ← Renumbered
localparam S_SOFTMAX    = 4'd3;
localparam S_GELU       = 4'd4;
localparam S_KV_QUANT   = 4'd5;
localparam S_COMPRESS   = 4'd6;
localparam S_DONE       = 4'd7;
`

Replace blocking FSM logic:
`erilog
// BEFORE (lines 310-365): Waits for .done before advancing
always @(posedge clk) begin
    case(state)
        S_ROPE: if (rope_done) state <= S_ROPE_W;
        S_ROPE_W: state <= S_GQA;           ← 1 CYCLE WASTED
        S_GQA: if (gqa_done) state <= S_GQA_W;
        S_GQA_W: state <= S_SOFTMAX;        ← 1 CYCLE WASTED
        ...

// AFTER: Ready/valid handshaking (no waits!)
always @(posedge clk) begin
    case(state)
        S_ROPE: if (rope_done && gqa_ready) state <= S_GQA;
        S_GQA: if (gqa_done && softmax_ready) state <= S_SOFTMAX;
        S_SOFTMAX: if (sm_done && gelu_ready) state <= S_GELU;
        ...
        // Each state transition checks downstream ready (combinational)
        // No explicit wait states!
`

**Step 3: Add ready/valid signals to all stage modules**

For each of the 6 stages (rope, gqa, softmax, gelu, kv_quant, compress):

`erilog
// Add to rope_encoder module:
module rope_encoder #(...) (
    ...
    output wire valid_out,
    input  wire ready_downstream,      ← NEW: Can downstream accept?
    ...
);

// Update FSM to drive valid only when ready:
always @(posedge clk) begin
    valid_out <= valid_in && ready_downstream;
    // If downstream not ready, don't output (holds previous)
end
`

**Step 4: Wire skid buffers between stages**

Between RoPE output and GQA input:
`erilog
// In optimized_transformer_layer.v, after rope instantiation:
wire rope_valid_to_gqa;
wire [128-1:0] rope_q_to_gqa;
wire gqa_accepts_rope;

skid_buffer #(.DATA_WIDTH(128)) rope_to_gqa_buf (
    .clk(clk), .rst(rst),
    .valid_in(rope_valid_out),
    .data_in(rope_q_out),
    .ready_from_downstream(gqa_accepts_input),
    .ready_for_upstream(rope_ready_for_next),
    .valid_out(rope_valid_to_gqa),
    .data_out(rope_q_to_gqa)
);

// Connect buffered data to GQA
wire gqa_valid_in = rope_valid_to_gqa;
wire [128-1:0] gqa_input_q = rope_q_to_gqa;
wire gqa_accepts_input = 1'b1;  // GQA always ready (it's stage 2)

// Repeat for all 5 inter-stage connections:
// RoPE→GQA, GQA→Softmax, Softmax→GELU, GELU→KV, KV→Compress
`

**Verification Checklist:**
- [ ] FSM no longer has wait states (grepping for _W should find 0 results)
- [ ] All 6 stages have valid_out and ready input
- [ ] All 5 inter-stage connections use skid buffers
- [ ] cosim_report.txt shows per-stage cycles unchanged (just overlapped)
- [ ] Total cycles reduced: 130 → 115 (±2)
- [ ] Zero-skip rate unchanged (28.7%)

---

### Task 1.3: Expand Softmax Accumulators (1 hour)
**File:** \tl/compute/parallel_softmax.v\, lines 40-41  
**Effort:** 30 minutes

`erilog
// BEFORE:
reg [7:0] exp_val [0:VECTOR_LEN-1];
reg [15:0] exp_sum;

// AFTER:
reg [15:0] exp_val [0:VECTOR_LEN-1];  // Widen to 16-bit (match LUT output)
reg [20:0] exp_sum;                    // Widen to 21-bit (prevent overflow)
`

**Impact:** Eliminates saturation-correction cycles, improves accuracy

---

## PHASE 2: WEEK 2 MEMORY & COMPUTE (Target: 95 cycles, -26% overall)

### Task 2.1: KV Cache Quantization Fusion (2-3 days)
**File:** \tl/memory/kv_cache_quantizer.v\, lines 78-110  
**Effort:** 2 days

**Key changes:**

**Part A: Parallelize min/max tree**
`erilog
// BEFORE: Sequential for-loop (~5 cycles)
for (i = 1; i < VEC_LEN; i = i + 1) begin
    if (val < vmin) vmin = val;
    if (val > vmax) vmax = val;
end

// AFTER: Parallel comparators (1 cycle)
wire [15:0] v0 = kv_in[15:0];
wire [15:0] v1 = kv_in[31:16];
wire [15:0] v2 = kv_in[47:32];
wire [15:0] v3 = kv_in[63:48];

wire [15:0] min_01 = (v0 < v1) ? v0 : v1;
wire [15:0] max_01 = (v0 > v1) ? v0 : v1;
wire [15:0] min_23 = (v2 < v3) ? v2 : v3;
wire [15:0] max_23 = (v2 > v3) ? v2 : v3;

wire [15:0] vmin_parallel = (min_01 < min_23) ? min_01 : min_23;
wire [15:0] vmax_parallel = (max_01 > max_23) ? max_01 : max_23;
`

**Part B: Replace division with reciprocal LUT**
`erilog
// BEFORE: Slow division
q_raw = (shifted_u + (scale_factor >> 1)) / scale_factor;  // ~3-4 cycles

// AFTER: Multiply by precomputed reciprocal (1 cycle!)
wire [15:0] recip_scale = recip_lut_lookup(vmax_parallel - vmin_parallel);
wire [31:0] q_raw = (shifted << 8) * recip_scale;  // Multiply is fast
`

**Part C: Pipeline architecture**
`erilog
// Stage 1: Min/max + scale computation (combinational, ~1 cycle total)
// Stage 2: All 4 elements quantized in parallel using multipliers (1 cycle)
// Total KV quantization: 2 cycles (was 15!)
`

### Task 2.2: DMA Prefetch Engine Integration (1-2 days)
**File:** \tl/top/gpu_system_top_v2.v\, around line 280  
**Effort:** 1 day (simple wiring)

**The module ALREADY EXISTS!** Just instantiate it.

`erilog
// ADD to gpu_system_top_v2 (after line 280):

wire [7:0] prefetch_current_layer;
wire [7:0] prefetch_prefetch_layer;
wire prefetch_dma_request;
wire [31:0] prefetch_dma_src_addr;
wire [15:0] prefetch_dma_length;
wire prefetch_dma_done;

prefetch_engine #(
    .BUFFER_DEPTH(64),
    .DATA_WIDTH(32)
) u_prefetch (
    .clk(clk), .rst(rst),
    .start(start_processing),
    .layer_done(opt_kv_done),         // Signal from compute pipeline
    .total_layers(total_num_layers),  // From command processor
    .dma_request(prefetch_dma_request),
    .dma_src_addr(prefetch_dma_src_addr),
    .dma_length(prefetch_dma_length),
    .dma_done(dma_finished),          // Wire from existing DMA controller
    .compute_ready(weight_buffer_ready),
    .error(prefetch_error)
);

// Wire prefetch requests to DMA (in DMA arbitration logic):
if (prefetch_dma_request && !compute_needs_weights) begin
    dma_src_addr <= prefetch_dma_src_addr;
    dma_length   <= prefetch_dma_length;
    dma_start    <= 1;
end else if (compute_needs_weights) begin
    // Compute takes priority
    dma_src_addr <= compute_dma_src_addr;
    dma_length   <= compute_dma_length;
    dma_start    <= 1;
end
`

**Expected impact:** Hide 8-15 cycles of memory latency

### Task 2.3: GQA Head Replication Pipeline (1 day)
**File:** \tl/integration/optimized_transformer_layer.v\, lines 84-108

Add pipeline stage between RoPE replication muxes and GQA input:
`erilog
reg [256-1:0] gqa_q_in_r_pipelined;
reg gqa_input_valid_delayed;

always @(posedge clk) begin
    if (rope_valid_out) begin
        gqa_q_in_r_pipelined <= gqa_q_in_r;
        gqa_input_valid_delayed <= 1'b1;
    end else begin
        gqa_input_valid_delayed <= 1'b0;
    end
end

// Use pipelined version
wire [256-1:0] gqa_q_in_final = gqa_q_in_r_pipelined;
wire gqa_valid_in_final = gqa_input_valid_delayed;
`

**Impact:** Reduces critical path on head replication muxes, 1-2 cycles

---

## PHASE 3: WEEK 3 SCALING (Target: 80 cycles, -38% overall)

### Task 3.1: Systolic Array 8×8 (Optional - Area Tradeoff)
**File:** \tl/compute/systolic_array.v\, line 19

If area budget allows:
`erilog
parameter ARRAY_SIZE = 8;  // Was: 4
// Synthesis should expand PE grid from 16 → 64 PEs
`

**Impact:** 7-cycle reduction (single-pass 8×8 matmul)  
**Cost:** ~4× area, ~4× power

**Alternative (lower cost):** Implement weight prefetching for 4×4 array
`erilog
// While computing pass 1, prefetch pass 2's weights
// Effective throughput improvement without area increase
`

### Task 3.2: KV Cache Double-Buffering (2 days)
**File:** Create \tl/memory/kv_cache_pingpong.v\

`erilog
// Implement dual KV cache RAMs (A and B)
// - Compute reads from cache_A
// - DMA/prefetch writes to cache_B
// - Swap on token boundary
// Impact: Eliminate read-after-write stall (2-3 cycles)
`

### Task 3.3: Layer Norm + Residuals (3-4 days, REQUIRED FOR CORRECTNESS)
**File:** Modify \tl/integration/optimized_transformer_layer.v\

Add missing stages:
1. LayerNorm before attention (uses inv_sqrt_lut_256)
2. Residual add after attention
3. LayerNorm before FFN
4. Residual add after FFN

**Important:** May add 6-8 cycles but required for real models!

---

## VERIFICATION & MEASUREMENT

### After Each Phase:
`ash
# Run cosim_report and check:
cd model
python cosim_runner.py --config config_optimized.yaml --output phase_X_report.txt

# Compare:
# Phase 0 baseline:    130 cycles
# Phase 1 target:      115 cycles (difference: -15)
# Phase 2 target:       95 cycles (difference: -20 cumulative)
# Phase 3 target:       80 cycles (difference: -15 cumulative)
`

### Zero-Skip Validation:
`ash
# Ensure sparsity benefit unchanged
grep "zero_skip_rate" phase_X_report.txt
# Expected: 28.7% ±2% (don't degrade below 26%)
`

### Per-Stage Cycle Breakdown:
`ash
# Check per-stage outputs (lines 46-52 in optimized_transformer_layer.v)
echo "rope_cycles: "
echo "gqa_cycles: "
echo "softmax_cycles: "
echo "gelu_cycles: "
echo "kv_quant_cycles: "
echo "compress_cycles: "
echo "total_cycles: "

# Verify overlap: total should be ~sum of max(stage latencies) not sum of all
# Before:  total = rope + gqa + softmax + gelu + kv + compress (130)
# After:   total = max(overlapped) (95-115)
`

---

## REGRESSION TEST SUITE

Create comprehensive tests to prevent regressions:

**tests/test_pipeline_throughput.py:**
`python
def test_phase1_pipeline_reduction():
    # Verify 15-cycle reduction from Phase 1
    cycles = run_cosim(config='phase1')
    assert cycles < 120, f"Expected <120 cycles, got {cycles}"

def test_phase2_memory_hiding():
    # Verify prefetch hides 8+ cycles
    cycles_seq_1 = run_cosim(config='phase2', seq_len=1)
    cycles_seq_5 = run_cosim(config='phase2', seq_len=5)
    assert (cycles_seq_5 * 5 - cycles_seq_1) < (95 * 5), \
           "Prefetch not hiding memory latency"

def test_zero_skip_preserved():
    # Ensure sparsity benefit maintained
    sparsity = measure_zero_skip_rate()
    assert sparsity > 0.26, f"Sparsity degraded to {sparsity}"
`

---

## CHECKPOINTS & MILESTONES

| Week | Task | Target Cycles | Effort | Success Criteria |
|------|------|--------------|--------|-----------------|
| 1.0 | Softmax exp LUT | 128-130 | 5 min | Exp values accurate ✓ |
| 1.1 | Pipeline FSM refactor | 115-120 | 3-5 days | -12-15 cycles ✓, no wait states ✓ |
| 1.2 | Accumulators | 114-118 | 1 hr | No saturation ✓ |
| 2.0 | KV quantization fusion | 110-115 | 2 days | -3-5 cycles ✓ |
| 2.1 | DMA prefetch wire | 105-110 | 1 day | Latency hiding visible ✓ |
| 2.2 | GQA pipeline | 103-108 | 1 day | Timing margin improved ✓ |
| 3.0 | Double-buffering (opt) | 100-105 | 2 days | -2-3 cycles ✓ |
| 3.1 | 8×8 systolic (opt) | 93-98 | 2 days | -7 cycles ✓ (if done) |
| 4.0 | Imprinting profiles | 8 cycles (models) | 3 days | 16× speedup for compatible ✓ |

