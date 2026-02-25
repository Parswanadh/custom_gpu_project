// ============================================================================
// Module: gpu_top_pipelined
// Description: Deeply pipelined GPU top-level.
//   5-stage pipeline: FETCH → DEQUANT → ZERO_CHECK → ALU → WRITEBACK
//   Throughput: 1 result per clock cycle (after 4-cycle fill latency)
//   vs. original FSM: 1 result per 5 clock cycles
//
// Key difference from gpu_top:
//   - All 5 stages operate simultaneously on different data
//   - Inter-stage pipeline registers isolate each stage
//   - Valid bits propagate through pipeline
//   - Achieves 5x throughput improvement
// ============================================================================
module gpu_top_pipelined (
    input  wire        clk,
    input  wire        rst,
    // Configuration
    input  wire [1:0]  mode,           // ALU precision mode
    input  wire [3:0]  dq_scale,       // Dequantizer scale factor
    input  wire [3:0]  dq_offset,      // Dequantizer zero-point offset
    // Sparse memory write interface (for loading weights)
    input  wire        mem_write_en,
    input  wire [7:0]  mem_write_val,
    input  wire [3:0]  mem_write_idx,
    // Inference interface — can accept new input EVERY CYCLE
    input  wire        valid_in,       // New input available
    input  wire [3:0]  weight_addr,    // Which weight to fetch
    input  wire [7:0]  activation_in,  // Incoming activation value
    // Output — produces result EVERY CYCLE (after pipeline fill)
    output reg  [63:0] result_out,     // Final computation result
    output reg         valid_out,      // Result is ready
    output reg         zero_skipped,   // 1 if zero was detected
    // Pipeline status
    output wire [4:0]  pipe_active     // Which pipeline stages are active
);

    // ========================================================================
    // Stage 0 → 1 Pipeline Registers: FETCH stage
    // ========================================================================
    reg        s1_valid;
    reg [7:0]  s1_activation;
    reg [3:0]  s1_weight_addr;

    // ========================================================================
    // Stage 1 → 2 Pipeline Registers: DEQUANT stage
    // ========================================================================
    reg        s2_valid;
    reg [7:0]  s2_activation;
    reg [7:0]  s2_mem_data;

    // ========================================================================
    // Stage 2 → 3 Pipeline Registers: ZERO_CHECK stage
    // ========================================================================
    reg        s3_valid;
    reg [7:0]  s3_activation;
    reg [7:0]  s3_dequant_weight;

    // ========================================================================
    // Stage 3 → 4 Pipeline Registers: ALU stage
    // ========================================================================
    reg        s4_valid;
    reg        s4_zero_skip;
    reg [15:0] s4_alu_a;
    reg [15:0] s4_alu_b;

    // ========================================================================
    // Sparse Memory Controller (shared, single-cycle read)
    // ========================================================================
    reg [7:0] weight_mem [0:15];
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 16; i = i + 1)
                weight_mem[i] <= 8'd0;
        end else if (mem_write_en) begin
            weight_mem[mem_write_idx] <= mem_write_val;
        end
    end

    // ========================================================================
    // STAGE 1: FETCH — Read weight from memory
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            s1_valid      <= 1'b0;
            s1_activation <= 8'd0;
            s1_weight_addr <= 4'd0;
        end else begin
            s1_valid      <= valid_in;
            s1_activation <= activation_in;
            s1_weight_addr <= weight_addr;
        end
    end

    // ========================================================================
    // STAGE 2: DEQUANT — Read memory output, pass to dequantizer
    // ========================================================================
    wire [7:0] fetched_weight = weight_mem[s1_weight_addr];

    // Inline dequantizer: INT4 → INT8
    wire [3:0] int4_val = fetched_weight[3:0];
    wire [7:0] dequant_result = (int4_val * dq_scale) + {4'd0, dq_offset};

    always @(posedge clk) begin
        if (rst) begin
            s2_valid      <= 1'b0;
            s2_activation <= 8'd0;
            s2_mem_data   <= 8'd0;
        end else begin
            s2_valid      <= s1_valid;
            s2_activation <= s1_activation;
            s2_mem_data   <= dequant_result;
        end
    end

    // ========================================================================
    // STAGE 3: ZERO_CHECK — Detect zeros, decide skip
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            s3_valid          <= 1'b0;
            s3_activation     <= 8'd0;
            s3_dequant_weight <= 8'd0;
        end else begin
            s3_valid          <= s2_valid;
            s3_activation     <= s2_activation;
            s3_dequant_weight <= s2_mem_data;
        end
    end

    wire s3_is_zero = (s3_dequant_weight == 8'd0) || (s3_activation == 8'd0);

    // ========================================================================
    // STAGE 4: ALU — Compute or skip
    // ========================================================================
    always @(posedge clk) begin
        if (rst) begin
            s4_valid     <= 1'b0;
            s4_zero_skip <= 1'b0;
            s4_alu_a     <= 16'd0;
            s4_alu_b     <= 16'd0;
        end else begin
            s4_valid     <= s3_valid;
            s4_zero_skip <= s3_is_zero;
            if (s3_is_zero) begin
                s4_alu_a <= 16'd0;
                s4_alu_b <= 16'd0;
            end else begin
                s4_alu_a <= {8'd0, s3_dequant_weight};
                s4_alu_b <= {8'd0, s3_activation};
            end
        end
    end

    // ========================================================================
    // STAGE 5: WRITEBACK — Compute result and output
    // ========================================================================
    // Variable precision multiply (inline for pipeline efficiency)
    always @(posedge clk) begin
        if (rst) begin
            result_out   <= 64'd0;
            valid_out    <= 1'b0;
            zero_skipped <= 1'b0;
        end else begin
            valid_out    <= s4_valid;
            zero_skipped <= s4_zero_skip;
            if (s4_valid) begin
                if (s4_zero_skip) begin
                    result_out <= 64'd0;
                end else begin
                    case (mode)
                        2'b00: begin
                            // 4x parallel 4-bit multiplies
                            result_out <= {
                                8'd0, s4_alu_a[15:12] * s4_alu_b[15:12],
                                8'd0, s4_alu_a[11:8]  * s4_alu_b[11:8],
                                8'd0, s4_alu_a[7:4]   * s4_alu_b[7:4],
                                8'd0, s4_alu_a[3:0]   * s4_alu_b[3:0]
                            };
                        end
                        2'b01: begin
                            // 2x parallel 8-bit multiplies
                            result_out <= {
                                32'd0,
                                s4_alu_a[15:8] * s4_alu_b[15:8],
                                s4_alu_a[7:0]  * s4_alu_b[7:0]
                            };
                        end
                        2'b10: begin
                            // 1x 16-bit multiply
                            result_out <= {32'd0, s4_alu_a * s4_alu_b};
                        end
                        default: result_out <= 64'd0;
                    endcase
                end
            end
        end
    end

    // Pipeline activity indicator
    assign pipe_active = {s4_valid, s3_valid, s2_valid, s1_valid, valid_in};

endmodule
