// ============================================================================
// Module: gpu_top_pipelined
// Description: Deeply pipelined GPU top-level with signed Q8.8 support.
//   5-stage pipeline: FETCH → DEQUANT → ZERO_CHECK → ALU → WRITEBACK
//   Throughput: 1 result per clock cycle (after 4-cycle fill latency)
//
//   FIXES APPLIED:
//     - Issue #1:  Signed arithmetic throughout
//     - Issue #4:  Ready/valid handshaking with pipeline stall support
// ============================================================================
module gpu_top_pipelined (
    input  wire        clk,
    input  wire        rst,
    // Configuration
    input  wire [1:0]  mode,
    input  wire [3:0]  dq_scale,
    input  wire [3:0]  dq_offset,
    // Sparse memory write interface
    input  wire        mem_write_en,
    input  wire signed [7:0]  mem_write_val,
    input  wire [3:0]  mem_write_idx,
    // Inference interface
    input  wire        valid_in,
    input  wire [3:0]  weight_addr,
    input  wire signed [7:0]  activation_in,
    // Pipeline flow control (Issue #4)
    input  wire        downstream_ready,
    output wire        ready,
    // Output
    output reg  [63:0] result_out,
    output reg         valid_out,
    output reg         zero_skipped,
    // Pipeline status
    output wire [4:0]  pipe_active
);

    // Pipeline stall logic
    wire pipe_stall = valid_out & ~downstream_ready;
    assign ready = ~pipe_stall;

    // ========================================================================
    // Weight Memory (single-cycle read, signed)
    // ========================================================================
    reg signed [7:0] weight_mem [0:15];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 16; i = i + 1)
                weight_mem[i] <= 8'sd0;
        end else if (mem_write_en) begin
            weight_mem[mem_write_idx] <= mem_write_val;
        end
    end

    // ========================================================================
    // STAGE 1: FETCH
    // ========================================================================
    reg               s1_valid;
    reg signed [7:0]  s1_activation;
    reg [3:0]         s1_weight_addr;

    always @(posedge clk) begin
        if (rst) begin
            s1_valid      <= 1'b0;
            s1_activation <= 8'sd0;
            s1_weight_addr <= 4'd0;
        end else if (!pipe_stall) begin
            s1_valid      <= valid_in;
            s1_activation <= activation_in;
            s1_weight_addr <= weight_addr;
        end
    end

    // ========================================================================
    // STAGE 2: DEQUANT (signed)
    // ========================================================================
    wire signed [7:0] fetched_weight = weight_mem[s1_weight_addr];
    wire signed [15:0] scaled_weight = fetched_weight * $signed({1'b0, dq_scale});
    wire signed [7:0] dequant_result = scaled_weight[11:4] + $signed({4'b0, dq_offset});

    reg               s2_valid;
    reg signed [7:0]  s2_activation;
    reg signed [7:0]  s2_mem_data;

    always @(posedge clk) begin
        if (rst) begin
            s2_valid      <= 1'b0;
            s2_activation <= 8'sd0;
            s2_mem_data   <= 8'sd0;
        end else if (!pipe_stall) begin
            s2_valid      <= s1_valid;
            s2_activation <= s1_activation;
            s2_mem_data   <= dequant_result;
        end
    end

    // ========================================================================
    // STAGE 3: ZERO_CHECK (signed compare)
    // ========================================================================
    reg               s3_valid;
    reg signed [7:0]  s3_activation;
    reg signed [7:0]  s3_dequant_weight;

    always @(posedge clk) begin
        if (rst) begin
            s3_valid          <= 1'b0;
            s3_activation     <= 8'sd0;
            s3_dequant_weight <= 8'sd0;
        end else if (!pipe_stall) begin
            s3_valid          <= s2_valid;
            s3_activation     <= s2_activation;
            s3_dequant_weight <= s2_mem_data;
        end
    end

    wire s3_is_zero = (s3_dequant_weight == 8'sd0) || (s3_activation == 8'sd0);

    // ========================================================================
    // STAGE 4: ALU (signed multiply)
    // ========================================================================
    reg               s4_valid;
    reg               s4_zero_skip;
    reg signed [15:0] s4_alu_a;
    reg signed [15:0] s4_alu_b;

    always @(posedge clk) begin
        if (rst) begin
            s4_valid     <= 1'b0;
            s4_zero_skip <= 1'b0;
            s4_alu_a     <= 16'sd0;
            s4_alu_b     <= 16'sd0;
        end else if (!pipe_stall) begin
            s4_valid     <= s3_valid;
            s4_zero_skip <= s3_is_zero;
            if (s3_is_zero) begin
                s4_alu_a <= 16'sd0;
                s4_alu_b <= 16'sd0;
            end else begin
                s4_alu_a <= {{8{s3_dequant_weight[7]}}, s3_dequant_weight};
                s4_alu_b <= {{8{s3_activation[7]}}, s3_activation};
            end
        end
    end

    // ========================================================================
    // STAGE 5: WRITEBACK (signed multiply results)
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            result_out   <= 64'sd0;
            valid_out    <= 1'b0;
            zero_skipped <= 1'b0;
        end else if (!pipe_stall) begin
            valid_out    <= s4_valid;
            zero_skipped <= s4_zero_skip;
            if (s4_valid) begin
                if (s4_zero_skip) begin
                    result_out <= 64'sd0;
                end else begin
                    case (mode)
                        2'b00: begin
                            result_out <= {
                                {8{1'b0}}, s4_alu_a[15:12] * s4_alu_b[15:12],
                                {8{1'b0}}, s4_alu_a[11:8]  * s4_alu_b[11:8],
                                {8{1'b0}}, s4_alu_a[7:4]   * s4_alu_b[7:4],
                                {8{1'b0}}, s4_alu_a[3:0]   * s4_alu_b[3:0]
                            };
                        end
                        2'b01: begin
                            result_out <= {
                                32'sd0,
                                s4_alu_a[15:8] * s4_alu_b[15:8],
                                s4_alu_a[7:0]  * s4_alu_b[7:0]
                            };
                        end
                        2'b10: begin
                            result_out <= {{32{1'b0}}, s4_alu_a * s4_alu_b};
                        end
                        default: result_out <= 64'sd0;
                    endcase
                end
            end
        end
    end

    assign pipe_active = {s4_valid, s3_valid, s2_valid, s1_valid, valid_in};

endmodule
