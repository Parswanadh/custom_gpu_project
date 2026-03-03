// ============================================================================
// Module: ffn_block
// Description: Feed-forward network block for transformer layers.
//   Computes: FFN(x) = Linear2(GELU(Linear1(x)))
//   Linear1: EMBED_DIM → FFN_DIM (expansion)
//   Linear2: FFN_DIM → EMBED_DIM (projection back)
//
//   FIXES APPLIED:
//     - Issue #7:  Weights stored in SRAM, loaded via write interface
//     - Issue #13: GELU via 256-entry LUT (replaces inline 3-piece approx)
//
// Parameters: EMBED_DIM, FFN_DIM, DATA_WIDTH
// ============================================================================
module ffn_block #(
    parameter EMBED_DIM  = 4,
    parameter FFN_DIM    = 8,
    parameter DATA_WIDTH = 16
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    x_in,

    // Weight SRAM loading interface (Issue #7)
    input  wire                               weight_load_en,
    input  wire                               weight_layer_sel,   // 0=W1/b1, 1=W2/b2
    input  wire                               weight_is_bias,     // 0=weight, 1=bias
    input  wire [$clog2(FFN_DIM>EMBED_DIM?FFN_DIM:EMBED_DIM)-1:0] weight_row,
    input  wire [$clog2(FFN_DIM>EMBED_DIM?FFN_DIM:EMBED_DIM)-1:0] weight_col,
    input  wire signed [DATA_WIDTH-1:0]       weight_data,

    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    y_out,
    output reg                                valid_out,
    output reg  [31:0]                        zero_skip_count
);

    // Weight SRAM (Issue #7)
    reg signed [DATA_WIDTH-1:0] w1 [0:EMBED_DIM-1][0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] b1 [0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] w2 [0:FFN_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] b2 [0:EMBED_DIM-1];

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] hidden [0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] activated [0:FFN_DIM-1];

    reg signed [2*DATA_WIDTH-1:0] accum;
    reg signed [2*DATA_WIDTH-1:0] product;

    integer i, j;
    reg [3:0] state;
    localparam IDLE    = 4'd0;
    localparam LINEAR1 = 4'd1;
    localparam GELU_ST = 4'd2;
    localparam LINEAR2 = 4'd3;
    localparam DONE    = 4'd4;

    // Issue #13: GELU LUT instance (combinational)
    reg signed [DATA_WIDTH-1:0] gelu_input;
    wire signed [DATA_WIDTH-1:0] gelu_output;
    gelu_lut_256 u_gelu (.x_in(gelu_input), .gelu_out(gelu_output));

    reg [$clog2(FFN_DIM):0] gelu_idx;

    // Weight loading (Issue #7)
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < EMBED_DIM; i = i + 1)
                for (j = 0; j < FFN_DIM; j = j + 1)
                    w1[i][j] <= {DATA_WIDTH{1'b0}};
            for (j = 0; j < FFN_DIM; j = j + 1)
                b1[j] <= {DATA_WIDTH{1'b0}};
            for (i = 0; i < FFN_DIM; i = i + 1)
                for (j = 0; j < EMBED_DIM; j = j + 1)
                    w2[i][j] <= {DATA_WIDTH{1'b0}};
            for (j = 0; j < EMBED_DIM; j = j + 1)
                b2[j] <= {DATA_WIDTH{1'b0}};
        end else if (weight_load_en) begin
            if (!weight_layer_sel) begin
                if (weight_is_bias)
                    b1[weight_col] <= weight_data;
                else
                    w1[weight_row][weight_col] <= weight_data;
            end else begin
                if (weight_is_bias)
                    b2[weight_col] <= weight_data;
                else
                    w2[weight_row][weight_col] <= weight_data;
            end
        end
    end

    // Computation FSM
    always @(posedge clk) begin
        if (rst) begin
            state           <= IDLE;
            valid_out       <= 1'b0;
            y_out           <= 0;
            zero_skip_count <= 0;
            gelu_idx        <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            x[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        state <= LINEAR1;
                    end
                end

                LINEAR1: begin
                    for (j = 0; j < FFN_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            if (x[i] != 0 && w1[i][j] != 0) begin
                                product = x[i] * w1[i][j];
                                accum = accum + product;
                            end else begin
                                zero_skip_count <= zero_skip_count + 1;
                            end
                        end
                        hidden[j] = accum[DATA_WIDTH+7:8] + b1[j];
                    end
                    gelu_idx <= 0;
                    state <= GELU_ST;
                end

                GELU_ST: begin
                    // Issue #13: Apply GELU via LUT, one element per cycle
                    if (gelu_idx < FFN_DIM) begin
                        gelu_input <= hidden[gelu_idx];
                        activated[gelu_idx] <= gelu_output;
                        gelu_idx <= gelu_idx + 1;
                    end else begin
                        state <= LINEAR2;
                    end
                end

                LINEAR2: begin
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < FFN_DIM; i = i + 1) begin
                            if (activated[i] != 0 && w2[i][j] != 0) begin
                                product = activated[i] * w2[i][j];
                                accum = accum + product;
                            end else begin
                                zero_skip_count <= zero_skip_count + 1;
                            end
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
