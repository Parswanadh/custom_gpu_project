// ============================================================================
// Module: gpu_top_integrated
// Description: Fully integrated GPU with ALL optimizations:
//   1. 5-stage pipeline (1 result per cycle per lane)
//   2. 4-wide memory read (4 weights per cycle)
//   3. 4 parallel compute lanes (4 multiplies per cycle)
//   4. Zero-skip on all 4 lanes
//
// Throughput: 4 results per clock cycle (after 4-cycle fill)
//   vs pipeline-only: 1 result/cycle = 4x improvement
//   vs original FSM:  1 result/7 cycles = ~28x improvement
//
// Architecture:
//   Stage 1 (FETCH):      Read 4 weights from wide memory controller
//   Stage 2 (DEQUANT):    Dequantize all 4 weights in parallel
//   Stage 3 (ZERO_CHECK): Check all 4 lanes for zeros
//   Stage 4 (ALU):        4 parallel multiplies (or skip if zero)
//   Stage 5 (WRITEBACK):  Output 4 results + accumulate
// ============================================================================
module gpu_top_integrated #(
    parameter MEM_DEPTH = 64,       // Weight memory depth
    parameter LANES     = 4         // Number of parallel compute lanes
)(
    input  wire        clk,
    input  wire        rst,

    // Configuration
    input  wire [3:0]  dq_scale,        // Dequantizer scale
    input  wire [3:0]  dq_offset,       // Dequantizer offset

    // Weight memory write interface
    input  wire        mem_write_en,
    input  wire [7:0]  mem_write_val,
    input  wire [5:0]  mem_write_idx,

    // Inference interface — accepts new input EVERY CYCLE
    input  wire        valid_in,
    input  wire [5:0]  weight_base_addr, // Read 4 weights starting here
    input  wire [7:0]  activation_in,    // Activation (broadcast to all 4 lanes)

    // Output — 4 results per cycle (after pipeline fill)
    output reg  [63:0] result_out,       // 4x 16-bit results packed
    output reg         valid_out,
    output reg  [3:0]  zero_skip_mask,   // Which lanes were zero-skipped
    output reg  [31:0] accumulator,      // Running dot-product sum

    // Pipeline status
    output wire [4:0]  pipe_active
);

    // ========================================================================
    // Weight Memory (4-wide read)
    // ========================================================================
    reg [7:0] weight_mem [0:MEM_DEPTH-1];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < MEM_DEPTH; i = i + 1)
                weight_mem[i] <= 8'd0;
        end else if (mem_write_en) begin
            weight_mem[mem_write_idx] <= mem_write_val;
        end
    end

    // 4-wide read: read 4 consecutive weights in one cycle
    wire [7:0] mem_read_0 = weight_mem[weight_base_addr];
    wire [7:0] mem_read_1 = weight_mem[weight_base_addr + 1];
    wire [7:0] mem_read_2 = weight_mem[weight_base_addr + 2];
    wire [7:0] mem_read_3 = weight_mem[weight_base_addr + 3];

    // ========================================================================
    // STAGE 1: FETCH — Capture 4 weights + activation
    // ========================================================================
    reg        s1_valid;
    reg [7:0]  s1_activation;
    reg [7:0]  s1_weight [0:3];

    always @(posedge clk) begin
        if (rst) begin
            s1_valid      <= 1'b0;
            s1_activation <= 8'd0;
            s1_weight[0]  <= 8'd0;
            s1_weight[1]  <= 8'd0;
            s1_weight[2]  <= 8'd0;
            s1_weight[3]  <= 8'd0;
        end else begin
            s1_valid      <= valid_in;
            s1_activation <= activation_in;
            s1_weight[0]  <= mem_read_0;
            s1_weight[1]  <= mem_read_1;
            s1_weight[2]  <= mem_read_2;
            s1_weight[3]  <= mem_read_3;
        end
    end

    // ========================================================================
    // STAGE 2: DEQUANT — Dequantize all 4 weights in parallel
    // ========================================================================
    reg        s2_valid;
    reg [7:0]  s2_activation;
    reg [7:0]  s2_dq_weight [0:3];

    // 4 parallel inline dequantizers
    wire [3:0] int4_val_0 = s1_weight[0][3:0];
    wire [3:0] int4_val_1 = s1_weight[1][3:0];
    wire [3:0] int4_val_2 = s1_weight[2][3:0];
    wire [3:0] int4_val_3 = s1_weight[3][3:0];

    wire [7:0] dq_out_0 = (int4_val_0 * dq_scale) + {4'd0, dq_offset};
    wire [7:0] dq_out_1 = (int4_val_1 * dq_scale) + {4'd0, dq_offset};
    wire [7:0] dq_out_2 = (int4_val_2 * dq_scale) + {4'd0, dq_offset};
    wire [7:0] dq_out_3 = (int4_val_3 * dq_scale) + {4'd0, dq_offset};

    always @(posedge clk) begin
        if (rst) begin
            s2_valid         <= 1'b0;
            s2_activation    <= 8'd0;
            s2_dq_weight[0]  <= 8'd0;
            s2_dq_weight[1]  <= 8'd0;
            s2_dq_weight[2]  <= 8'd0;
            s2_dq_weight[3]  <= 8'd0;
        end else begin
            s2_valid         <= s1_valid;
            s2_activation    <= s1_activation;
            s2_dq_weight[0]  <= dq_out_0;
            s2_dq_weight[1]  <= dq_out_1;
            s2_dq_weight[2]  <= dq_out_2;
            s2_dq_weight[3]  <= dq_out_3;
        end
    end

    // ========================================================================
    // STAGE 3: ZERO_CHECK — Check all 4 lanes for zeros
    // ========================================================================
    reg        s3_valid;
    reg [7:0]  s3_activation;
    reg [7:0]  s3_weight [0:3];
    reg [3:0]  s3_zero_mask;

    wire is_zero_0 = (s2_dq_weight[0] == 8'd0) || (s2_activation == 8'd0);
    wire is_zero_1 = (s2_dq_weight[1] == 8'd0) || (s2_activation == 8'd0);
    wire is_zero_2 = (s2_dq_weight[2] == 8'd0) || (s2_activation == 8'd0);
    wire is_zero_3 = (s2_dq_weight[3] == 8'd0) || (s2_activation == 8'd0);

    always @(posedge clk) begin
        if (rst) begin
            s3_valid      <= 1'b0;
            s3_activation <= 8'd0;
            s3_weight[0]  <= 8'd0;
            s3_weight[1]  <= 8'd0;
            s3_weight[2]  <= 8'd0;
            s3_weight[3]  <= 8'd0;
            s3_zero_mask  <= 4'd0;
        end else begin
            s3_valid      <= s2_valid;
            s3_activation <= s2_activation;
            s3_weight[0]  <= s2_dq_weight[0];
            s3_weight[1]  <= s2_dq_weight[1];
            s3_weight[2]  <= s2_dq_weight[2];
            s3_weight[3]  <= s2_dq_weight[3];
            s3_zero_mask  <= {is_zero_3, is_zero_2, is_zero_1, is_zero_0};
        end
    end

    // ========================================================================
    // STAGE 4: ALU — 4 parallel multiplies (or skip)
    // ========================================================================
    reg        s4_valid;
    reg [3:0]  s4_zero_mask;
    reg [15:0] s4_product [0:3];

    always @(posedge clk) begin
        if (rst) begin
            s4_valid       <= 1'b0;
            s4_zero_mask   <= 4'd0;
            s4_product[0]  <= 16'd0;
            s4_product[1]  <= 16'd0;
            s4_product[2]  <= 16'd0;
            s4_product[3]  <= 16'd0;
        end else begin
            s4_valid     <= s3_valid;
            s4_zero_mask <= s3_zero_mask;

            // Lane 0
            if (s3_zero_mask[0])
                s4_product[0] <= 16'd0;
            else
                s4_product[0] <= s3_activation * s3_weight[0];

            // Lane 1
            if (s3_zero_mask[1])
                s4_product[1] <= 16'd0;
            else
                s4_product[1] <= s3_activation * s3_weight[1];

            // Lane 2
            if (s3_zero_mask[2])
                s4_product[2] <= 16'd0;
            else
                s4_product[2] <= s3_activation * s3_weight[2];

            // Lane 3
            if (s3_zero_mask[3])
                s4_product[3] <= 16'd0;
            else
                s4_product[3] <= s3_activation * s3_weight[3];
        end
    end

    // ========================================================================
    // STAGE 5: WRITEBACK — Output 4 results + accumulate
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            result_out     <= 64'd0;
            valid_out      <= 1'b0;
            zero_skip_mask <= 4'd0;
            accumulator    <= 32'd0;
        end else begin
            valid_out      <= s4_valid;
            zero_skip_mask <= s4_zero_mask;
            if (s4_valid) begin
                // Pack 4x 16-bit results into 64-bit output
                result_out <= {s4_product[3], s4_product[2],
                               s4_product[1], s4_product[0]};
                // Accumulate all 4 products (for dot-product mode)
                accumulator <= accumulator +
                    {16'd0, s4_product[0]} + {16'd0, s4_product[1]} +
                    {16'd0, s4_product[2]} + {16'd0, s4_product[3]};
            end
        end
    end

    // Pipeline activity indicator
    assign pipe_active = {s4_valid, s3_valid, s2_valid, s1_valid, valid_in};

endmodule
