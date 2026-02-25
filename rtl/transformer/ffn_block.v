// ============================================================================
// Module: ffn_block
// Description: Feed-forward network block for transformer layers.
//   Computes: FFN(x) = Linear2(GELU(Linear1(x)))
//   Linear1: EMBED_DIM → FFN_DIM (expansion)
//   Linear2: FFN_DIM → EMBED_DIM (projection back)
// Parameters: EMBED_DIM, FFN_DIM, DATA_WIDTH
// ============================================================================
module ffn_block #(
    parameter EMBED_DIM  = 4,
    parameter FFN_DIM    = 8,     // Typically 4× EMBED_DIM
    parameter DATA_WIDTH = 16
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    x_in,
    // Flattened weight matrices
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]  w1_flat,  // EMBED→FFN
    input  wire [FFN_DIM*DATA_WIDTH-1:0]             b1_flat,  // Bias 1
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]   w2_flat,  // FFN→EMBED
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            b2_flat,  // Bias 2
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    y_out,
    output reg                                valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] hidden [0:FFN_DIM-1];    // After Linear1
    reg signed [DATA_WIDTH-1:0] activated [0:FFN_DIM-1]; // After GELU

    // Weight access
    reg signed [DATA_WIDTH-1:0] w1 [0:EMBED_DIM-1][0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] b1 [0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] w2 [0:FFN_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] b2 [0:EMBED_DIM-1];

    reg signed [2*DATA_WIDTH-1:0] accum;
    reg signed [2*DATA_WIDTH-1:0] product;

    // GELU constants (Q8.8)
    localparam signed [DATA_WIDTH-1:0] NEG_THREE = -16'sd768;
    localparam signed [DATA_WIDTH-1:0] POS_THREE =  16'sd768;
    localparam signed [DATA_WIDTH-1:0] HALF      =  16'sd128;
    localparam signed [DATA_WIDTH-1:0] SLOPE     =  16'sd43;

    reg signed [2*DATA_WIDTH-1:0] slope_x;
    reg signed [DATA_WIDTH-1:0]   sig_approx;
    reg signed [2*DATA_WIDTH-1:0] gelu_prod;

    integer i, j;
    reg [3:0] state;
    localparam IDLE    = 4'd0;
    localparam LINEAR1 = 4'd1;
    localparam GELU    = 4'd2;
    localparam LINEAR2 = 4'd3;
    localparam DONE    = 4'd4;

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            y_out     <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            x[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        // Unpack weights
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            for (j = 0; j < FFN_DIM; j = j + 1)
                                w1[i][j] <= w1_flat[(i*FFN_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                        for (j = 0; j < FFN_DIM; j = j + 1)
                            b1[j] <= b1_flat[j*DATA_WIDTH +: DATA_WIDTH];
                        for (i = 0; i < FFN_DIM; i = i + 1)
                            for (j = 0; j < EMBED_DIM; j = j + 1)
                                w2[i][j] <= w2_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                        for (j = 0; j < EMBED_DIM; j = j + 1)
                            b2[j] <= b2_flat[j*DATA_WIDTH +: DATA_WIDTH];
                        state <= LINEAR1;
                    end
                end

                LINEAR1: begin
                    // hidden[j] = sum_i(x[i] * w1[i][j]) + b1[j]
                    for (j = 0; j < FFN_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * w1[i][j];
                            accum = accum + product;
                        end
                        hidden[j] = accum[DATA_WIDTH+7:8] + b1[j];
                    end
                    state <= GELU;
                end

                GELU: begin
                    // Apply GELU activation to each hidden element
                    for (j = 0; j < FFN_DIM; j = j + 1) begin
                        if (hidden[j] < NEG_THREE)
                            activated[j] = 0;
                        else if (hidden[j] > POS_THREE)
                            activated[j] = hidden[j];
                        else begin
                            slope_x = SLOPE * hidden[j];
                            sig_approx = HALF + slope_x[DATA_WIDTH+7:8];
                            gelu_prod = hidden[j] * sig_approx;
                            activated[j] = gelu_prod[DATA_WIDTH+7:8];
                        end
                    end
                    state <= LINEAR2;
                end

                LINEAR2: begin
                    // y[j] = sum_i(activated[i] * w2[i][j]) + b2[j]
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < FFN_DIM; i = i + 1) begin
                            product = activated[i] * w2[i][j];
                            accum = accum + product;
                        end
                        y_out[j*DATA_WIDTH +: DATA_WIDTH] <= accum[DATA_WIDTH+7:8] + b2[j];
                    end
                    state <= DONE;
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
