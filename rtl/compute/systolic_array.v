// ============================================================================
// Module: systolic_array
// Description: NxN systolic array of MAC units for matrix multiplication.
//   Weight-stationary dataflow: weights are preloaded, activations flow through.
//   Computes C = A × B where A is fed row-by-row and B (weights) is preloaded.
//   Output: one row of result per cycle once pipeline is filled.
// Parameters: ARRAY_SIZE, DATA_WIDTH, ACC_WIDTH
// ============================================================================
module systolic_array #(
    parameter ARRAY_SIZE  = 4,    // NxN array dimension
    parameter DATA_WIDTH  = 16,   // Width of operands
    parameter ACC_WIDTH   = 32    // Width of accumulators
)(
    input  wire                                clk,
    input  wire                                rst,
    // Weight loading
    input  wire                                load_weight,
    input  wire [$clog2(ARRAY_SIZE)-1:0]       weight_row,
    input  wire [$clog2(ARRAY_SIZE)-1:0]       weight_col,
    input  wire [DATA_WIDTH-1:0]               weight_data,
    // Activation input (one element per cycle per row)
    input  wire                                valid_in,
    input  wire [ARRAY_SIZE*DATA_WIDTH-1:0]    act_in,      // One activation per row
    // Result output
    output wire [ARRAY_SIZE*ACC_WIDTH-1:0]     result_out,  // One result per column
    output reg                                 valid_out
);

    // Weight storage
    reg [DATA_WIDTH-1:0] weights [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Accumulator array
    reg [ACC_WIDTH-1:0]  accum [0:ARRAY_SIZE-1];

    // Pipeline shift register for activations (skew input)
    reg [DATA_WIDTH-1:0] act_pipe [0:ARRAY_SIZE-1];

    // Internal signals
    integer i, j;
    reg [2*DATA_WIDTH-1:0] product;

    // Cycle counter for pipeline fill
    reg [$clog2(ARRAY_SIZE)+1:0] cycle_count;
    reg computing;

    // Weight loading
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < ARRAY_SIZE; i = i + 1)
                for (j = 0; j < ARRAY_SIZE; j = j + 1)
                    weights[i][j] <= {DATA_WIDTH{1'b0}};
        end else if (load_weight) begin
            weights[weight_row][weight_col] <= weight_data;
        end
    end

    // Main computation: matrix-vector multiply (one column of weights × activation vector)
    // For simplicity, we do the full dot product in one cycle per output element
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < ARRAY_SIZE; i = i + 1)
                accum[i] <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
            computing <= 1'b0;
        end else if (valid_in) begin
            // For each output column j: result[j] = sum_i(act[i] * weight[i][j])
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin
                accum[j] = {ACC_WIDTH{1'b0}};
                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                    product = act_in[i*DATA_WIDTH +: DATA_WIDTH] * weights[i][j];
                    // Zero-skip optimization
                    if (act_in[i*DATA_WIDTH +: DATA_WIDTH] != {DATA_WIDTH{1'b0}} &&
                        weights[i][j] != {DATA_WIDTH{1'b0}}) begin
                        accum[j] = accum[j] + product[ACC_WIDTH-1:0];
                    end
                end
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

    // Pack accumulator outputs
    genvar g;
    generate
        for (g = 0; g < ARRAY_SIZE; g = g + 1) begin : pack_out
            assign result_out[g*ACC_WIDTH +: ACC_WIDTH] = accum[g];
        end
    endgenerate

endmodule
