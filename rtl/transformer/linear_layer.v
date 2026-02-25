// ============================================================================
// Module: linear_layer
// Description: Matrix-vector multiply y = W × x + bias
//   Uses the systolic array for the multiply, adds bias afterward.
//   W is stored internally and loaded via a write interface.
// Parameters: IN_DIM, OUT_DIM, DATA_WIDTH
// ============================================================================
module linear_layer #(
    parameter IN_DIM     = 4,
    parameter OUT_DIM    = 4,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32
)(
    input  wire                               clk,
    input  wire                               rst,
    // Weight loading interface
    input  wire                               load_weight,
    input  wire [$clog2(IN_DIM)-1:0]          w_row,
    input  wire [$clog2(OUT_DIM)-1:0]         w_col,
    input  wire signed [DATA_WIDTH-1:0]       w_data,
    // Bias loading interface
    input  wire                               load_bias,
    input  wire [$clog2(OUT_DIM)-1:0]         b_idx,
    input  wire signed [DATA_WIDTH-1:0]       b_data,
    // Compute interface
    input  wire                               valid_in,
    input  wire [IN_DIM*DATA_WIDTH-1:0]       x_in,     // Input vector
    output reg  [OUT_DIM*DATA_WIDTH-1:0]      y_out,    // Output vector
    output reg                                valid_out
);

    // Weight matrix storage
    reg signed [DATA_WIDTH-1:0] weights [0:IN_DIM-1][0:OUT_DIM-1];
    // Bias storage
    reg signed [DATA_WIDTH-1:0] bias [0:OUT_DIM-1];

    // Internal computation
    reg signed [DATA_WIDTH-1:0] x_buf [0:IN_DIM-1];
    reg signed [ACC_WIDTH-1:0]  accum;
    reg signed [2*DATA_WIDTH-1:0] product;

    integer i, j;
    reg [3:0] state;
    localparam IDLE    = 4'd0;
    localparam COMPUTE = 4'd1;
    localparam DONE    = 4'd2;

    reg [$clog2(OUT_DIM):0] out_idx;

    // Weight and bias loading
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < IN_DIM; i = i + 1)
                for (j = 0; j < OUT_DIM; j = j + 1)
                    weights[i][j] <= {DATA_WIDTH{1'b0}};
            for (j = 0; j < OUT_DIM; j = j + 1)
                bias[j] <= {DATA_WIDTH{1'b0}};
        end else begin
            if (load_weight)
                weights[w_row][w_col] <= w_data;
            if (load_bias)
                bias[b_idx] <= b_data;
        end
    end

    // Computation: y[j] = sum_i(x[i] * w[i][j]) + bias[j]
    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            out_idx   <= 0;
            y_out     <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Unpack input
                        for (i = 0; i < IN_DIM; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        out_idx <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (out_idx < OUT_DIM) begin
                        // Dot product for output j
                        accum = {ACC_WIDTH{1'b0}};
                        for (i = 0; i < IN_DIM; i = i + 1) begin
                            if (x_buf[i] != 0 && weights[i][out_idx] != 0) begin
                                product = x_buf[i] * weights[i][out_idx];
                                accum = accum + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
                            end
                        end
                        // Q8.8 * Q8.8 = Q16.16, shift to Q8.8 and add bias
                        y_out[out_idx*DATA_WIDTH +: DATA_WIDTH] <=
                            accum[DATA_WIDTH+7:8] + bias[out_idx];
                        out_idx <= out_idx + 1;
                    end else begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
