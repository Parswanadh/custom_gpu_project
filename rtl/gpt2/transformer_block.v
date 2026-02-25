// ============================================================================
// Module: transformer_block
// Description: Full transformer decoder block:
//   x → LayerNorm → Attention → Add(residual) → LayerNorm → FFN → Add(residual)
//   Combines layer_norm, attention_unit, and ffn_block.
// Parameters: EMBED_DIM, NUM_HEADS, FFN_DIM, DATA_WIDTH
// ============================================================================
module transformer_block #(
    parameter EMBED_DIM  = 4,
    parameter NUM_HEADS  = 2,
    parameter HEAD_DIM   = 2,
    parameter FFN_DIM    = 8,
    parameter DATA_WIDTH = 16
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    x_in,
    // LayerNorm params (2 sets: pre-attention and pre-FFN)
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln1_gamma, ln1_beta,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln2_gamma, ln2_beta,
    // Attention weights (flattened)
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat, wk_flat, wv_flat, wo_flat,
    // FFN weights (flattened)
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]   ffn_w1_flat,
    input  wire [FFN_DIM*DATA_WIDTH-1:0]              ffn_b1_flat,
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]    ffn_w2_flat,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            ffn_b2_flat,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    y_out,
    output reg                                valid_out
);

    // Internal wires
    reg  [EMBED_DIM*DATA_WIDTH-1:0] residual1, residual2;
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln1_out, attn_out, ln2_out, ffn_out;
    wire                            ln1_valid, attn_valid, ln2_valid, ffn_valid;

    // Enable signals for sub-modules
    reg ln1_en, attn_en, ln2_en, ffn_en;

    // State machine
    reg [3:0] state;
    localparam IDLE     = 4'd0;
    localparam LN1      = 4'd1;
    localparam ATTN     = 4'd2;
    localparam ADD1     = 4'd3;
    localparam LN2      = 4'd4;
    localparam FFN      = 4'd5;
    localparam ADD2     = 4'd6;
    localparam DONE     = 4'd7;

    integer i;

    // Layer Norm 1
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln1 (
        .clk(clk), .rst(rst), .valid_in(ln1_en),
        .x_in(residual1), .gamma_in(ln1_gamma), .beta_in(ln1_beta),
        .y_out(ln1_out), .valid_out(ln1_valid)
    );

    // Attention
    attention_unit #(.EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
                     .HEAD_DIM(HEAD_DIM), .DATA_WIDTH(DATA_WIDTH)) u_attn (
        .clk(clk), .rst(rst), .valid_in(attn_en),
        .x_in(ln1_out),
        .wq_flat(wq_flat), .wk_flat(wk_flat), .wv_flat(wv_flat), .wo_flat(wo_flat),
        .y_out(attn_out), .valid_out(attn_valid)
    );

    // Layer Norm 2
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln2 (
        .clk(clk), .rst(rst), .valid_in(ln2_en),
        .x_in(residual2), .gamma_in(ln2_gamma), .beta_in(ln2_beta),
        .y_out(ln2_out), .valid_out(ln2_valid)
    );

    // FFN
    ffn_block #(.EMBED_DIM(EMBED_DIM), .FFN_DIM(FFN_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ffn (
        .clk(clk), .rst(rst), .valid_in(ffn_en),
        .x_in(ln2_out),
        .w1_flat(ffn_w1_flat), .b1_flat(ffn_b1_flat),
        .w2_flat(ffn_w2_flat), .b2_flat(ffn_b2_flat),
        .y_out(ffn_out), .valid_out(ffn_valid)
    );

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            y_out     <= 0;
            residual1 <= 0;
            residual2 <= 0;
            ln1_en <= 0; attn_en <= 0; ln2_en <= 0; ffn_en <= 0;
        end else begin
            // Default: deassert all enables
            ln1_en <= 0; attn_en <= 0; ln2_en <= 0; ffn_en <= 0;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        residual1 <= x_in;
                        ln1_en <= 1'b1;
                        state <= LN1;
                    end
                end

                LN1: begin
                    if (ln1_valid) begin
                        attn_en <= 1'b1;
                        state <= ATTN;
                    end
                end

                ATTN: begin
                    if (attn_valid) begin
                        // Residual add: residual2 = x + attention_out
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            residual2[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual1[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(attn_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        state <= ADD1;
                    end
                end

                ADD1: begin
                    ln2_en <= 1'b1;
                    state <= LN2;
                end

                LN2: begin
                    if (ln2_valid) begin
                        ffn_en <= 1'b1;
                        state <= FFN;
                    end
                end

                FFN: begin
                    if (ffn_valid) begin
                        // Residual add: y = residual2 + ffn_out
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual2[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(ffn_out[i*DATA_WIDTH +: DATA_WIDTH]);
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
