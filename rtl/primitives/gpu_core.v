// ============================================================================
// Module: gpu_core
// Description: Parameterized GPU compute core with scalable lanes.
//   LANES parameter controls how many parallel multiply-accumulate operations
//   happen per clock cycle. Uses Verilog generate blocks to auto-create
//   N parallel dequantizers, zero-checkers, and ALUs.
//
// Parameters:
//   LANES     = number of parallel compute lanes (4, 8, 16, 32, 64, 128)
//   MEM_DEPTH = weight memory depth per core
//
// Throughput: LANES results per clock cycle (after 4-cycle pipeline fill)
//   LANES=4:   4 products/cycle   (current)
//   LANES=32:  32 products/cycle  (8x improvement)
//   LANES=128: 128 products/cycle (32x improvement)
//
// Architecture (5-stage pipeline × N lanes):
//   Stage 1: FETCH      — Read N weights from memory (N-wide port)
//   Stage 2: DEQUANT    — N parallel dequantizers (INT4 → INT8)
//   Stage 3: ZERO_CHECK — N parallel zero detectors
//   Stage 4: ALU        — N parallel multipliers (skip if zero)
//   Stage 5: WRITEBACK  — Output N results + accumulate
// ============================================================================
module gpu_core #(
    parameter LANES     = 4,
    parameter MEM_DEPTH = 256,
    parameter ADDR_W    = 8         // log2(MEM_DEPTH)
)(
    input  wire                    clk,
    input  wire                    rst,

    // Configuration
    input  wire [3:0]              dq_scale,
    input  wire [3:0]              dq_offset,

    // Core ID (for multi-core identification)
    input  wire [3:0]              core_id,

    // Weight memory write
    input  wire                    mem_write_en,
    input  wire [7:0]              mem_write_val,
    input  wire [ADDR_W-1:0]       mem_write_idx,

    // Inference interface
    input  wire                    valid_in,
    input  wire [ADDR_W-1:0]       weight_base_addr,   // Read LANES weights starting here
    input  wire [7:0]              activation_in,       // Broadcast to all lanes

    // Output
    output reg                     valid_out,
    output reg  [LANES-1:0]        zero_skip_mask,      // Per-lane zero-skip flags
    output reg  [31:0]             accumulator,          // Running dot-product sum
    output reg  [16*LANES-1:0]     lane_results,         // All lane products packed

    // Status
    output wire [4:0]              pipe_active,
    output wire [31:0]             products_per_cycle    // For reporting
);

    // ========================================================================
    // Weight Memory (N-wide read)
    // ========================================================================
    reg [7:0] weight_mem [0:MEM_DEPTH-1];
    integer mi;

    always @(posedge clk) begin
        if (rst) begin
            for (mi = 0; mi < MEM_DEPTH; mi = mi + 1)
                weight_mem[mi] <= 8'd0;
        end else if (mem_write_en) begin
            weight_mem[mem_write_idx] <= mem_write_val;
        end
    end

    // N-wide combinational read
    wire [7:0] mem_read [0:LANES-1];
    genvar r;
    generate
        for (r = 0; r < LANES; r = r + 1) begin : mem_read_gen
            assign mem_read[r] = weight_mem[weight_base_addr + r];
        end
    endgenerate

    // ========================================================================
    // STAGE 1: FETCH — Capture N weights + activation
    // ========================================================================
    reg        s1_valid;
    reg [7:0]  s1_activation;
    reg [7:0]  s1_weight [0:LANES-1];

    integer s1i;
    always @(posedge clk) begin
        if (rst) begin
            s1_valid      <= 1'b0;
            s1_activation <= 8'd0;
            for (s1i = 0; s1i < LANES; s1i = s1i + 1)
                s1_weight[s1i] <= 8'd0;
        end else begin
            s1_valid      <= valid_in;
            s1_activation <= activation_in;
            for (s1i = 0; s1i < LANES; s1i = s1i + 1)
                s1_weight[s1i] <= mem_read[s1i];
        end
    end

    // ========================================================================
    // STAGE 2: DEQUANT — N parallel dequantizers
    // ========================================================================
    reg        s2_valid;
    reg [7:0]  s2_activation;
    reg [7:0]  s2_dq_weight [0:LANES-1];

    // N parallel dequantizers (combinational)
    wire [7:0] dq_out [0:LANES-1];
    genvar d;
    generate
        for (d = 0; d < LANES; d = d + 1) begin : dequant_gen
            wire [3:0] int4_val = s1_weight[d][3:0];
            assign dq_out[d] = (int4_val * dq_scale) + {4'd0, dq_offset};
        end
    endgenerate

    integer s2i;
    always @(posedge clk) begin
        if (rst) begin
            s2_valid      <= 1'b0;
            s2_activation <= 8'd0;
            for (s2i = 0; s2i < LANES; s2i = s2i + 1)
                s2_dq_weight[s2i] <= 8'd0;
        end else begin
            s2_valid      <= s1_valid;
            s2_activation <= s1_activation;
            for (s2i = 0; s2i < LANES; s2i = s2i + 1)
                s2_dq_weight[s2i] <= dq_out[s2i];
        end
    end

    // ========================================================================
    // STAGE 3: ZERO_CHECK — N parallel zero detectors
    // ========================================================================
    reg        s3_valid;
    reg [7:0]  s3_activation;
    reg [7:0]  s3_weight [0:LANES-1];
    reg [LANES-1:0] s3_zero_mask;

    // N parallel zero detectors (combinational)
    wire [LANES-1:0] is_zero;
    genvar z;
    generate
        for (z = 0; z < LANES; z = z + 1) begin : zero_det_gen
            assign is_zero[z] = (s2_dq_weight[z] == 8'd0) || (s2_activation == 8'd0);
        end
    endgenerate

    integer s3i;
    always @(posedge clk) begin
        if (rst) begin
            s3_valid      <= 1'b0;
            s3_activation <= 8'd0;
            s3_zero_mask  <= {LANES{1'b0}};
            for (s3i = 0; s3i < LANES; s3i = s3i + 1)
                s3_weight[s3i] <= 8'd0;
        end else begin
            s3_valid      <= s2_valid;
            s3_activation <= s2_activation;
            s3_zero_mask  <= is_zero;
            for (s3i = 0; s3i < LANES; s3i = s3i + 1)
                s3_weight[s3i] <= s2_dq_weight[s3i];
        end
    end

    // ========================================================================
    // STAGE 4: ALU — N parallel multipliers (with zero-skip)
    // ========================================================================
    reg        s4_valid;
    reg [LANES-1:0] s4_zero_mask;
    reg [15:0] s4_product [0:LANES-1];

    integer s4i;
    always @(posedge clk) begin
        if (rst) begin
            s4_valid     <= 1'b0;
            s4_zero_mask <= {LANES{1'b0}};
            for (s4i = 0; s4i < LANES; s4i = s4i + 1)
                s4_product[s4i] <= 16'd0;
        end else begin
            s4_valid     <= s3_valid;
            s4_zero_mask <= s3_zero_mask;
            for (s4i = 0; s4i < LANES; s4i = s4i + 1) begin
                if (s3_zero_mask[s4i])
                    s4_product[s4i] <= 16'd0;       // Zero-skip!
                else
                    s4_product[s4i] <= s3_activation * s3_weight[s4i];
            end
        end
    end

    // ========================================================================
    // STAGE 5: WRITEBACK — Output N results + accumulate
    // ========================================================================

    // Reduction tree: sum all LANES products
    // Using a sequential loop (synthesizable)
    reg [31:0] lane_sum;
    integer si;
    always @(*) begin
        lane_sum = 32'd0;
        for (si = 0; si < LANES; si = si + 1)
            lane_sum = lane_sum + {16'd0, s4_product[si]};
    end

    integer wi;
    always @(posedge clk) begin
        if (rst) begin
            valid_out      <= 1'b0;
            zero_skip_mask <= {LANES{1'b0}};
            accumulator    <= 32'd0;
            lane_results   <= {16*LANES{1'b0}};
        end else begin
            valid_out      <= s4_valid;
            zero_skip_mask <= s4_zero_mask;
            if (s4_valid) begin
                // Pack all lane results
                for (wi = 0; wi < LANES; wi = wi + 1)
                    lane_results[wi*16 +: 16] <= s4_product[wi];
                // Accumulate sum of all lanes
                accumulator <= accumulator + lane_sum;
            end
        end
    end

    // Pipeline activity
    assign pipe_active = {s4_valid, s3_valid, s2_valid, s1_valid, valid_in};
    assign products_per_cycle = LANES;

endmodule
