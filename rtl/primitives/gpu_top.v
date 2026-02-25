// ============================================================================
// Module: gpu_top
// Description: Top-level pipeline connecting all 4 primitive modules:
//   sparse_memory_ctrl → fused_dequantizer → zero_detect_mult → variable_precision_alu
//
// Data flow:
//   1. weight_addr → sparse_memory_ctrl → fetches weight (INT4-width stored as 8-bit)
//   2. weight → fused_dequantizer → converts INT4 to INT8 (dequantized)
//   3. dequantized_weight + activation_in → zero_detect_mult → check zeros
//   4. If not zero → variable_precision_alu → compute result
//   5. result → result_out
// ============================================================================
module gpu_top (
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
    // Inference interface
    input  wire        start,          // Start inference for one weight
    input  wire [3:0]  weight_addr,    // Which weight to fetch
    input  wire [7:0]  activation_in,  // Incoming activation value
    // Output
    output wire [63:0] result_out,     // Final computation result
    output wire        valid_out,      // Result is ready
    output wire        zero_skipped    // 1 if zero was detected and multiply was skipped
);

    // ---- Internal wires ----

    // Sparse memory → dequantizer
    wire [7:0] mem_read_data;
    wire       mem_valid;

    // Dequantizer → zero detector
    wire [7:0] dequant_out;
    wire       dequant_valid;

    // Zero detector → ALU
    wire [15:0] zd_result;
    wire        zd_skipped;
    wire        zd_valid;

    // ---- State machine for pipeline sequencing ----
    reg [2:0] state;
    localparam IDLE       = 3'd0;
    localparam FETCH      = 3'd1;
    localparam DEQUANT    = 3'd2;
    localparam ZERO_CHECK = 3'd3;
    localparam ALU_EXEC   = 3'd4;
    localparam DONE       = 3'd5;

    reg        mem_read_en;
    reg [3:0]  mem_read_idx;
    reg        dq_valid_in;
    reg [3:0]  dq_int4_in;
    reg        zd_valid_in;
    reg [7:0]  zd_a, zd_b;
    reg        alu_valid_in;
    reg [15:0] alu_a, alu_b;
    reg        result_valid_reg;
    reg        zero_skip_reg;
    reg [63:0] result_reg;

    // Stored activation for pipeline
    reg [7:0]  act_stored;

    // ---- Module Instantiations ----

    // Module 1: Sparse Memory Controller
    sparse_memory_ctrl #(
        .MAX_VALUES(16),
        .DATA_WIDTH(8),
        .INDEX_WIDTH(4)
    ) u_sparse_mem (
        .clk(clk),
        .rst(rst),
        .write_en(mem_write_en),
        .write_val(mem_write_val),
        .write_idx(mem_write_idx),
        .read_en(mem_read_en),
        .read_idx(mem_read_idx),
        .read_data(mem_read_data),
        .valid_out(mem_valid),
        .num_stored()
    );

    // Module 2: Fused Dequantizer
    fused_dequantizer u_dequant (
        .clk(clk),
        .rst(rst),
        .valid_in(dq_valid_in),
        .int4_in(dq_int4_in),
        .scale(dq_scale),
        .offset(dq_offset),
        .int8_out(dequant_out),
        .valid_out(dequant_valid)
    );

    // Module 3: Zero Detect Multiplier
    zero_detect_mult u_zero_det (
        .clk(clk),
        .rst(rst),
        .valid_in(zd_valid_in),
        .a(zd_a),
        .b(zd_b),
        .result(zd_result),
        .skipped(zd_skipped),
        .valid_out(zd_valid)
    );

    // Module 4: Variable Precision ALU
    variable_precision_alu u_alu (
        .clk(clk),
        .rst(rst),
        .valid_in(alu_valid_in),
        .a(alu_a),
        .b(alu_b),
        .mode(mode),
        .result(result_out),
        .valid_out(valid_out)
    );

    assign zero_skipped = zero_skip_reg;

    // ---- Pipeline State Machine ----
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= IDLE;
            mem_read_en    <= 1'b0;
            mem_read_idx   <= 4'd0;
            dq_valid_in    <= 1'b0;
            dq_int4_in     <= 4'd0;
            zd_valid_in    <= 1'b0;
            zd_a           <= 8'd0;
            zd_b           <= 8'd0;
            alu_valid_in   <= 1'b0;
            alu_a          <= 16'd0;
            alu_b          <= 16'd0;
            act_stored     <= 8'd0;
            zero_skip_reg  <= 1'b0;
        end else begin
            // Default: deassert one-shot signals
            mem_read_en  <= 1'b0;
            dq_valid_in  <= 1'b0;
            zd_valid_in  <= 1'b0;
            alu_valid_in <= 1'b0;

            case (state)
                IDLE: begin
                    if (start) begin
                        mem_read_en  <= 1'b1;
                        mem_read_idx <= weight_addr;
                        act_stored   <= activation_in;
                        zero_skip_reg <= 1'b0;
                        state <= FETCH;
                    end
                end

                FETCH: begin
                    if (mem_valid) begin
                        // Feed raw weight (lower 4 bits) to dequantizer
                        dq_int4_in  <= mem_read_data[3:0];
                        dq_valid_in <= 1'b1;
                        state <= DEQUANT;
                    end
                end

                DEQUANT: begin
                    if (dequant_valid) begin
                        // Feed dequantized weight + activation to zero detector
                        zd_a        <= dequant_out;
                        zd_b        <= act_stored;
                        zd_valid_in <= 1'b1;
                        state <= ZERO_CHECK;
                    end
                end

                ZERO_CHECK: begin
                    if (zd_valid) begin
                        if (zd_skipped) begin
                            // Zero detected — skip ALU, go to DONE
                            zero_skip_reg <= 1'b1;
                            // Drive ALU with zeros to produce 0 output
                            alu_a        <= 16'd0;
                            alu_b        <= 16'd0;
                            alu_valid_in <= 1'b1;
                            state <= DONE;
                        end else begin
                            // Non-zero: feed to ALU
                            alu_a        <= {8'd0, dequant_out};
                            alu_b        <= {8'd0, act_stored};
                            alu_valid_in <= 1'b1;
                            state <= DONE;
                        end
                    end
                end

                DONE: begin
                    if (valid_out) begin
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
