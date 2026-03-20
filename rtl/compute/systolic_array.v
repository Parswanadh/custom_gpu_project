// ============================================================================
// Module: systolic_array
// Description: NxN systolic array of registered Processing Elements (PEs)
//   for matrix multiplication. Weight-stationary dataflow.
//
//   FIXES APPLIED:
//     - Issue #9: Proper PE mesh with registered inter-PE communication
//       (replaces combinational for-loop that computed entire matrix in 1 cycle)
//
//   Architecture: Weight-stationary, output-stationary accumulation
//     - Weights are preloaded into each PE
//     - Activations flow left-to-right with skewed (staggered) input
//     - Partial sums flow top-to-bottom and accumulate
//     - Result available after ARRAY_SIZE + ARRAY_SIZE - 1 cycles
//
// Parameters: ARRAY_SIZE, DATA_WIDTH, ACC_WIDTH
// ============================================================================
module systolic_array #(
    parameter ARRAY_SIZE  = 4,
    parameter DATA_WIDTH  = 16,
    parameter ACC_WIDTH   = 32
)(
    input  wire                                clk,
    input  wire                                rst,
    // Weight loading
    input  wire                                load_weight,
    input  wire [$clog2(ARRAY_SIZE)-1:0]       weight_row,
    input  wire [$clog2(ARRAY_SIZE)-1:0]       weight_col,
    input  wire signed [DATA_WIDTH-1:0]        weight_data,
    // Activation input (one element per row, skew-fed)
    input  wire                                valid_in,
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]    act_in,
    // Clear accumulators for new computation
    input  wire                                clear_acc,
    // Precision mode: 0=Q8.8 (16-bit weights), 1=Q4 (4-bit quantized weights)
    input  wire [1:0]                          precision_mode,
    input  wire [7:0]                          q4_block_scale,
    input  wire [3:0]                          q4_block_zero,
    // Result output
    output wire [ARRAY_SIZE*ACC_WIDTH-1:0]     result_out,
    output reg                                 valid_out
);

    // ========================================================================
    // Processing Element (PE) — the core building block
    //   Each PE:
    //   - Holds one weight (loaded once)
    //   - Receives activation from the left, passes it right
    //   - Receives partial sum from above, adds its own product, passes down
    //   - Zero-skip: skips MAC when either operand is zero
    // ========================================================================

    // Inter-PE wires
    // Activations flow left → right: act_wire[row][col]
    // Partial sums flow top → down: psum_wire[row][col]
    wire signed [DATA_WIDTH-1:0] act_wire  [0:ARRAY_SIZE-1][0:ARRAY_SIZE];
    wire signed [ACC_WIDTH-1:0]  psum_wire [0:ARRAY_SIZE+1-1][0:ARRAY_SIZE-1];
    wire                         act_valid_wire [0:ARRAY_SIZE-1][0:ARRAY_SIZE];

    // Weight storage
    reg signed [DATA_WIDTH-1:0] weights [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    integer wi, wj;
    always @(posedge clk) begin
        if (rst) begin
            for (wi = 0; wi < ARRAY_SIZE; wi = wi + 1)
                for (wj = 0; wj < ARRAY_SIZE; wj = wj + 1)
                    weights[wi][wj] <= {DATA_WIDTH{1'b0}};
        end else if (load_weight) begin
            weights[weight_row][weight_col] <= weight_data;
        end
    end

    // ========================================================================
    // Input Skew Registers — stagger activations for proper systolic timing
    //   Row 0: no delay, Row 1: 1-cycle delay, Row k: k-cycle delay
    // ========================================================================
    reg signed [DATA_WIDTH-1:0] skew_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg                         skew_valid [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    genvar sk;
    generate
        for (sk = 0; sk < ARRAY_SIZE; sk = sk + 1) begin : skew_gen
            integer sj;
            always @(posedge clk) begin
                if (rst) begin
                    for (sj = 0; sj < ARRAY_SIZE; sj = sj + 1) begin
                        skew_reg[sk][sj] <= {DATA_WIDTH{1'b0}};
                        skew_valid[sk][sj] <= 1'b0;
                    end
                end else begin
                    // First stage: capture input
                    skew_reg[sk][0] <= $signed(act_in[sk*DATA_WIDTH +: DATA_WIDTH]);
                    skew_valid[sk][0] <= valid_in;
                    // Shift register chain for skewing
                    for (sj = 1; sj < ARRAY_SIZE; sj = sj + 1) begin
                        skew_reg[sk][sj] <= skew_reg[sk][sj-1];
                        skew_valid[sk][sj] <= skew_valid[sk][sj-1];
                    end
                end
            end
        end
    endgenerate

    // Connect skewed inputs to first column of PE array
    genvar si;
    generate
        for (si = 0; si < ARRAY_SIZE; si = si + 1) begin : input_connect
            // Row si gets data from skew_reg[si][si] (si cycles of delay)
            assign act_wire[si][0] = (si < ARRAY_SIZE) ? skew_reg[si][si] : {DATA_WIDTH{1'b0}};
            assign act_valid_wire[si][0] = (si < ARRAY_SIZE) ? skew_valid[si][si] : 1'b0;
        end
    endgenerate

    // Top row partial sums start at zero
    generate
        for (si = 0; si < ARRAY_SIZE; si = si + 1) begin : psum_top
            assign psum_wire[0][si] = {ACC_WIDTH{1'b0}};
        end
    endgenerate

    // ========================================================================
    // PE Array — ARRAY_SIZE × ARRAY_SIZE registered processing elements
    // ========================================================================
    genvar pr, pc;
    generate
        for (pr = 0; pr < ARRAY_SIZE; pr = pr + 1) begin : pe_row
            for (pc = 0; pc < ARRAY_SIZE; pc = pc + 1) begin : pe_col

                // PE local registers
                reg signed [DATA_WIDTH-1:0] pe_act_reg;
                reg signed [ACC_WIDTH-1:0]  pe_psum_reg;
                reg                         pe_valid_reg;

                // Q4 dequant: extract INT4 weight, apply (w - zero) * scale
                wire signed [3:0] q4_weight_raw = weights[pr][pc][3:0];
                wire signed [5:0] q4_shifted = $signed({{2{q4_weight_raw[3]}}, q4_weight_raw}) - $signed({2'b00, q4_block_zero});
                wire signed [13:0] q4_product = q4_shifted * $signed({1'b0, q4_block_scale});
                // Sign-extend to DATA_WIDTH (q4_product max magnitude = 5865 with signed INT4 + zero-point shift)
                wire signed [DATA_WIDTH-1:0] q4_dequant_weight = {{(DATA_WIDTH-14){q4_product[13]}}, q4_product};

                // Combinational: product + zero-skip
                wire signed [DATA_WIDTH-1:0] pe_weight = (precision_mode == 2'd1) ? q4_dequant_weight : weights[pr][pc];
                wire pe_is_zero = (act_wire[pr][pc] == {DATA_WIDTH{1'b0}}) ||
                                  (pe_weight == {DATA_WIDTH{1'b0}});
                wire signed [2*DATA_WIDTH-1:0] pe_product = act_wire[pr][pc] * pe_weight;
                wire signed [ACC_WIDTH-1:0] pe_mac_result = psum_wire[pr][pc] +
                    (pe_is_zero ? {ACC_WIDTH{1'b0}} :
                     {{(ACC_WIDTH-2*DATA_WIDTH){pe_product[2*DATA_WIDTH-1]}}, pe_product});

                always @(posedge clk) begin
                    if (rst || clear_acc) begin
                        pe_act_reg  <= {DATA_WIDTH{1'b0}};
                        pe_psum_reg <= {ACC_WIDTH{1'b0}};
                        pe_valid_reg <= 1'b0;
                    end else begin
                        pe_act_reg  <= act_wire[pr][pc];
                        pe_valid_reg <= act_valid_wire[pr][pc];
                        if (act_valid_wire[pr][pc])
                            pe_psum_reg <= pe_mac_result;
                    end
                end

                // Pass activation to the right
                assign act_wire[pr][pc+1] = pe_act_reg;
                assign act_valid_wire[pr][pc+1] = pe_valid_reg;

                // Pass partial sum downward
                assign psum_wire[pr+1][pc] = pe_psum_reg;

            end
        end
    endgenerate

    // ========================================================================
    // Output: bottom row of partial sums = final results
    // ========================================================================
    generate
        for (si = 0; si < ARRAY_SIZE; si = si + 1) begin : pack_out
            assign result_out[si*ACC_WIDTH +: ACC_WIDTH] = psum_wire[ARRAY_SIZE][si];
        end
    endgenerate

    // Valid output: shift pipeline supports consecutive valid_in pulses.
    reg [2*ARRAY_SIZE-1:0] valid_pipe;
    always @(posedge clk) begin
        if (rst || clear_acc) begin
            valid_out  <= 1'b0;
            valid_pipe <= {2*ARRAY_SIZE{1'b0}};
        end else begin
            valid_pipe[0] <= valid_in;
            valid_pipe[2*ARRAY_SIZE-1:1] <= valid_pipe[2*ARRAY_SIZE-2:0];
            valid_out <= valid_pipe[2*ARRAY_SIZE-1];
        end
    end


endmodule
