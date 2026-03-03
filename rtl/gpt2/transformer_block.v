// ============================================================================
// Module: transformer_block
// Description: Full transformer decoder block:
//   x → LayerNorm → Attention → Add(residual) → LayerNorm → FFN → Add(residual)
//
//   FIXES APPLIED:
//     - Issue #7:  Weights stored in sub-module SRAMs (load interface)
//     - Issue #8:  Per-layer weight loading (weights loaded before compute)
//
// Parameters: EMBED_DIM, NUM_HEADS, FFN_DIM, DATA_WIDTH, MAX_SEQ_LEN
// ============================================================================
module transformer_block #(
    parameter EMBED_DIM   = 4,
    parameter NUM_HEADS   = 2,
    parameter HEAD_DIM    = 2,
    parameter FFN_DIM     = 8,
    parameter MAX_SEQ_LEN = 32,
    parameter DATA_WIDTH  = 16
)(
    input  wire                               clk,
    input  wire                               rst,
    input  wire                               valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     seq_pos,

    // LayerNorm params
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln1_gamma, ln1_beta,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln2_gamma, ln2_beta,

    // Attention weight load interface (Issue #7/#8)
    input  wire                               attn_weight_load_en,
    input  wire [1:0]                         attn_weight_matrix_sel,
    input  wire [$clog2(EMBED_DIM)-1:0]       attn_weight_row,
    input  wire [$clog2(EMBED_DIM)-1:0]       attn_weight_col,
    input  wire signed [DATA_WIDTH-1:0]       attn_weight_data,

    // FFN weight load interface (Issue #7/#8)
    input  wire                               ffn_weight_load_en,
    input  wire                               ffn_weight_layer_sel,
    input  wire                               ffn_weight_is_bias,
    input  wire [$clog2(FFN_DIM>EMBED_DIM?FFN_DIM:EMBED_DIM)-1:0] ffn_weight_row,
    input  wire [$clog2(FFN_DIM>EMBED_DIM?FFN_DIM:EMBED_DIM)-1:0] ffn_weight_col,
    input  wire signed [DATA_WIDTH-1:0]       ffn_weight_data,

    // Attention mask
    input  wire                               causal_mask_en,

    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    y_out,
    output reg                                valid_out,
    output reg  [31:0]                        block_zero_skips
);

    // Internal wires
    reg  [EMBED_DIM*DATA_WIDTH-1:0] residual1, residual2;
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln1_out, attn_out, ln2_out, ffn_out;
    wire                            ln1_valid, attn_valid, ln2_valid, ffn_valid;
    wire [31:0]                     attn_zero_skips, ffn_zero_skips;

    reg ln1_en, attn_en, ln2_en, ffn_en;

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

    // Attention (Issue #7: SRAM weight interface)
    attention_unit #(
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_attn (
        .clk(clk), .rst(rst), .valid_in(attn_en),
        .x_in(ln1_out), .seq_pos(seq_pos),
        .weight_load_en(attn_weight_load_en),
        .weight_matrix_sel(attn_weight_matrix_sel),
        .weight_row(attn_weight_row),
        .weight_col(attn_weight_col),
        .weight_data(attn_weight_data),
        .causal_mask_en(causal_mask_en),
        .y_out(attn_out), .valid_out(attn_valid),
        .zero_skip_count(attn_zero_skips)
    );

    // Layer Norm 2
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln2 (
        .clk(clk), .rst(rst), .valid_in(ln2_en),
        .x_in(residual2), .gamma_in(ln2_gamma), .beta_in(ln2_beta),
        .y_out(ln2_out), .valid_out(ln2_valid)
    );

    // FFN (Issue #7: SRAM weight interface)
    ffn_block #(
        .EMBED_DIM(EMBED_DIM), .FFN_DIM(FFN_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_ffn (
        .clk(clk), .rst(rst), .valid_in(ffn_en),
        .x_in(ln2_out),
        .weight_load_en(ffn_weight_load_en),
        .weight_layer_sel(ffn_weight_layer_sel),
        .weight_is_bias(ffn_weight_is_bias),
        .weight_row(ffn_weight_row),
        .weight_col(ffn_weight_col),
        .weight_data(ffn_weight_data),
        .y_out(ffn_out), .valid_out(ffn_valid),
        .zero_skip_count(ffn_zero_skips)
    );

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            y_out     <= 0;
            residual1 <= 0;
            residual2 <= 0;
            ln1_en <= 0; attn_en <= 0; ln2_en <= 0; ffn_en <= 0;
            block_zero_skips <= 0;
        end else begin
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
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            residual2[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual1[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(attn_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        block_zero_skips <= attn_zero_skips;
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
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual2[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(ffn_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        block_zero_skips <= block_zero_skips + ffn_zero_skips;
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
