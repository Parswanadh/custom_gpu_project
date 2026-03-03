// ============================================================================
// Module: gpu_top_integrated
// Description: Fully integrated GPU with ALL optimizations.
//   FIXES APPLIED:
//     - Issue #1:  Signed arithmetic throughout
//     - Issue #4:  Pipeline stall support via ready/valid handshaking
//     - Issue #16: Parity on weight memory
// ============================================================================
module gpu_top_integrated #(
    parameter MEM_DEPTH = 64,
    parameter LANES     = 4
)(
    input  wire        clk,
    input  wire        rst,
    // Configuration
    input  wire [3:0]  dq_scale,
    input  wire [3:0]  dq_offset,
    // Weight memory write
    input  wire        mem_write_en,
    input  wire signed [7:0]  mem_write_val,
    input  wire [5:0]  mem_write_idx,
    // Inference interface
    input  wire        valid_in,
    input  wire [5:0]  weight_base_addr,
    input  wire signed [7:0]  activation_in,
    // Flow control
    input  wire        downstream_ready,
    output wire        ready,
    // Output
    output reg  signed [63:0] result_out,
    output reg         valid_out,
    output reg  [3:0]  zero_skip_mask,
    output reg  signed [31:0] accumulator,
    input  wire        acc_clear,
    // Status
    output wire [4:0]  pipe_active,
    output reg         parity_error
);

    wire pipe_stall = valid_out & ~downstream_ready;
    assign ready = ~pipe_stall;

    // Weight Memory with parity
    reg signed [7:0] weight_mem [0:MEM_DEPTH-1];
    reg              weight_par [0:MEM_DEPTH-1];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < MEM_DEPTH; i = i + 1) begin
                weight_mem[i] <= 8'sd0;
                weight_par[i] <= 1'b0;
            end
        end else if (mem_write_en) begin
            weight_mem[mem_write_idx] <= mem_write_val;
            weight_par[mem_write_idx] <= ^mem_write_val;
        end
    end

    wire signed [7:0] mr0 = weight_mem[weight_base_addr];
    wire signed [7:0] mr1 = weight_mem[weight_base_addr + 1];
    wire signed [7:0] mr2 = weight_mem[weight_base_addr + 2];
    wire signed [7:0] mr3 = weight_mem[weight_base_addr + 3];

    // Parity check
    wire par_ok0 = (^weight_mem[weight_base_addr]   == weight_par[weight_base_addr]);
    wire par_ok1 = (^weight_mem[weight_base_addr+1] == weight_par[weight_base_addr+1]);
    wire par_ok2 = (^weight_mem[weight_base_addr+2] == weight_par[weight_base_addr+2]);
    wire par_ok3 = (^weight_mem[weight_base_addr+3] == weight_par[weight_base_addr+3]);

    always @(posedge clk) begin
        if (rst) parity_error <= 1'b0;
        else if (s1_valid)
            parity_error <= ~(par_ok0 & par_ok1 & par_ok2 & par_ok3);
    end

    // STAGE 1: FETCH
    reg               s1_valid;
    reg signed [7:0]  s1_activation;
    reg signed [7:0]  s1_weight [0:3];

    always @(posedge clk) begin
        if (rst) begin
            s1_valid      <= 1'b0;
            s1_activation <= 8'sd0;
            s1_weight[0] <= 8'sd0; s1_weight[1] <= 8'sd0;
            s1_weight[2] <= 8'sd0; s1_weight[3] <= 8'sd0;
        end else if (!pipe_stall) begin
            s1_valid      <= valid_in;
            s1_activation <= activation_in;
            s1_weight[0] <= mr0; s1_weight[1] <= mr1;
            s1_weight[2] <= mr2; s1_weight[3] <= mr3;
        end
    end

    // STAGE 2: DEQUANT (signed)
    reg               s2_valid;
    reg signed [7:0]  s2_activation;
    reg signed [7:0]  s2_dq_weight [0:3];

    wire signed [15:0] sc0 = s1_weight[0] * $signed({1'b0, dq_scale});
    wire signed [15:0] sc1 = s1_weight[1] * $signed({1'b0, dq_scale});
    wire signed [15:0] sc2 = s1_weight[2] * $signed({1'b0, dq_scale});
    wire signed [15:0] sc3 = s1_weight[3] * $signed({1'b0, dq_scale});
    wire signed [7:0] dq0 = sc0[11:4] + $signed({4'b0, dq_offset});
    wire signed [7:0] dq1 = sc1[11:4] + $signed({4'b0, dq_offset});
    wire signed [7:0] dq2 = sc2[11:4] + $signed({4'b0, dq_offset});
    wire signed [7:0] dq3 = sc3[11:4] + $signed({4'b0, dq_offset});

    always @(posedge clk) begin
        if (rst) begin
            s2_valid <= 1'b0; s2_activation <= 8'sd0;
            s2_dq_weight[0] <= 8'sd0; s2_dq_weight[1] <= 8'sd0;
            s2_dq_weight[2] <= 8'sd0; s2_dq_weight[3] <= 8'sd0;
        end else if (!pipe_stall) begin
            s2_valid <= s1_valid; s2_activation <= s1_activation;
            s2_dq_weight[0] <= dq0; s2_dq_weight[1] <= dq1;
            s2_dq_weight[2] <= dq2; s2_dq_weight[3] <= dq3;
        end
    end

    // STAGE 3: ZERO_CHECK (signed)
    reg               s3_valid;
    reg signed [7:0]  s3_activation;
    reg signed [7:0]  s3_weight [0:3];
    reg [3:0]         s3_zero_mask;

    wire iz0 = (s2_dq_weight[0] == 8'sd0) || (s2_activation == 8'sd0);
    wire iz1 = (s2_dq_weight[1] == 8'sd0) || (s2_activation == 8'sd0);
    wire iz2 = (s2_dq_weight[2] == 8'sd0) || (s2_activation == 8'sd0);
    wire iz3 = (s2_dq_weight[3] == 8'sd0) || (s2_activation == 8'sd0);

    always @(posedge clk) begin
        if (rst) begin
            s3_valid <= 1'b0; s3_activation <= 8'sd0; s3_zero_mask <= 4'd0;
            s3_weight[0] <= 8'sd0; s3_weight[1] <= 8'sd0;
            s3_weight[2] <= 8'sd0; s3_weight[3] <= 8'sd0;
        end else if (!pipe_stall) begin
            s3_valid <= s2_valid; s3_activation <= s2_activation;
            s3_weight[0] <= s2_dq_weight[0]; s3_weight[1] <= s2_dq_weight[1];
            s3_weight[2] <= s2_dq_weight[2]; s3_weight[3] <= s2_dq_weight[3];
            s3_zero_mask <= {iz3, iz2, iz1, iz0};
        end
    end

    // STAGE 4: ALU (signed multiply)
    reg               s4_valid;
    reg [3:0]         s4_zero_mask;
    reg signed [15:0] s4_product [0:3];

    always @(posedge clk) begin
        if (rst) begin
            s4_valid <= 1'b0; s4_zero_mask <= 4'd0;
            s4_product[0] <= 16'sd0; s4_product[1] <= 16'sd0;
            s4_product[2] <= 16'sd0; s4_product[3] <= 16'sd0;
        end else if (!pipe_stall) begin
            s4_valid     <= s3_valid;
            s4_zero_mask <= s3_zero_mask;
            s4_product[0] <= s3_zero_mask[0] ? 16'sd0 : s3_activation * s3_weight[0];
            s4_product[1] <= s3_zero_mask[1] ? 16'sd0 : s3_activation * s3_weight[1];
            s4_product[2] <= s3_zero_mask[2] ? 16'sd0 : s3_activation * s3_weight[2];
            s4_product[3] <= s3_zero_mask[3] ? 16'sd0 : s3_activation * s3_weight[3];
        end
    end

    // STAGE 5: WRITEBACK
    always @(posedge clk) begin
        if (rst) begin
            result_out     <= 64'sd0;
            valid_out      <= 1'b0;
            zero_skip_mask <= 4'd0;
            accumulator    <= 32'sd0;
        end else if (acc_clear) begin
            accumulator <= 32'sd0;
        end else if (!pipe_stall) begin
            valid_out      <= s4_valid;
            zero_skip_mask <= s4_zero_mask;
            if (s4_valid) begin
                result_out <= {s4_product[3], s4_product[2], s4_product[1], s4_product[0]};
                accumulator <= accumulator +
                    {{16{s4_product[0][15]}}, s4_product[0]} +
                    {{16{s4_product[1][15]}}, s4_product[1]} +
                    {{16{s4_product[2][15]}}, s4_product[2]} +
                    {{16{s4_product[3][15]}}, s4_product[3]};
            end
        end
    end

    assign pipe_active = {s4_valid, s3_valid, s2_valid, s1_valid, valid_in};

endmodule
