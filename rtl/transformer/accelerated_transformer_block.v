// ============================================================================
// Module: accelerated_transformer_block
// Description: Complete transformer block using accelerated components.
//   Replaces the original transformer_block with one that uses:
//     - accelerated_attention (real KV cache + scoring)
//     - layer_norm (pre-norm architecture, same as original)
//     - accelerated FFN with residual connections
//
//   Architecture: Pre-LayerNorm Transformer
//     x → LN1 → Attention(+KV cache) → + residual → LN2 → FFN → + residual → out
//
// This block actually uses the optimized pipeline indirectly through
// accelerated_attention and can be wired to use accelerated_linear_layer.
// ============================================================================
module accelerated_transformer_block #(
    parameter EMBED_DIM   = 8,
    parameter NUM_HEADS   = 2,
    parameter HEAD_DIM    = 4,
    parameter FFN_DIM     = 16,
    parameter MAX_SEQ_LEN = 32,
    parameter DATA_WIDTH  = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]      seq_pos,

    // LayerNorm params
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     ln1_gamma, ln1_beta,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     ln2_gamma, ln2_beta,

    // Attention weights
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat, wk_flat, wv_flat, wo_flat,

    // FFN weights + biases
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]   ffn_w1_flat,
    input  wire [FFN_DIM*DATA_WIDTH-1:0]              ffn_b1_flat,
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]    ffn_w2_flat,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            ffn_b2_flat,

    // Output
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out,
    output reg  [31:0]                         block_zero_skips
);

    // Internal signals
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln1_out, ln2_out;
    wire                            ln1_valid, ln2_valid;
    wire [EMBED_DIM*DATA_WIDTH-1:0] attn_out;
    wire                            attn_valid;
    wire [31:0]                     attn_zero_skips;

    reg  [EMBED_DIM*DATA_WIDTH-1:0] residual1;
    reg  [EMBED_DIM*DATA_WIDTH-1:0] after_attn;

    // State machine
    reg [3:0] state;
    localparam IDLE     = 4'd0;
    localparam LN1      = 4'd1;
    localparam ATTN     = 4'd2;
    localparam RESID1   = 4'd3;
    localparam LN2      = 4'd4;
    localparam FFN      = 4'd5;
    localparam RESID2   = 4'd6;
    localparam DONE     = 4'd7;

    reg ln1_en, ln2_en, attn_en;
    integer i;

    // Sub-module: Layer Norm 1
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln1 (
        .clk(clk), .rst(rst), .valid_in(ln1_en),
        .x_in(x_in),
        .gamma_in(ln1_gamma), .beta_in(ln1_beta),
        .y_out(ln1_out), .valid_out(ln1_valid)
    );

    // Sub-module: Accelerated Attention (with KV cache!)
    accelerated_attention #(
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_attn (
        .clk(clk), .rst(rst), .valid_in(attn_en),
        .x_in(ln1_out), .seq_pos(seq_pos),
        .wq_flat(wq_flat), .wk_flat(wk_flat),
        .wv_flat(wv_flat), .wo_flat(wo_flat),
        .y_out(attn_out), .valid_out(attn_valid),
        .zero_skip_count(attn_zero_skips)
    );

    // Sub-module: Layer Norm 2
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln2 (
        .clk(clk), .rst(rst), .valid_in(ln2_en),
        .x_in(after_attn),
        .gamma_in(ln2_gamma), .beta_in(ln2_beta),
        .y_out(ln2_out), .valid_out(ln2_valid)
    );

    // FFN computation (inline for now — could use accelerated_linear_layer)
    reg signed [DATA_WIDTH-1:0] ffn_hidden [0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] ffn_out_buf [0:EMBED_DIM-1];
    reg signed [2*DATA_WIDTH-1:0] ffn_accum;
    reg signed [2*DATA_WIDTH-1:0] ffn_product;
    reg ffn_done;
    integer fi, fj;

    always @(posedge clk) begin
        if (rst) begin
            state          <= IDLE;
            valid_out      <= 1'b0;
            y_out          <= 0;
            ln1_en         <= 0;
            ln2_en         <= 0;
            attn_en        <= 0;
            block_zero_skips <= 0;
            ffn_done       <= 0;
        end else begin
            ln1_en  <= 0;
            ln2_en  <= 0;
            attn_en <= 0;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Save input as residual
                        residual1 <= x_in;
                        ln1_en    <= 1'b1;
                        state     <= LN1;
                    end
                end

                LN1: begin
                    if (ln1_valid) begin
                        attn_en <= 1'b1;
                        state   <= ATTN;
                    end
                end

                ATTN: begin
                    if (attn_valid) begin
                        // Residual connection: x + attention(LN(x))
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            after_attn[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual1[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(attn_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        block_zero_skips <= attn_zero_skips;
                        state <= RESID1;
                    end
                end

                RESID1: begin
                    ln2_en <= 1'b1;
                    state  <= LN2;
                end

                LN2: begin
                    if (ln2_valid) begin
                        state    <= FFN;
                        ffn_done <= 0;
                    end
                end

                FFN: begin
                    if (!ffn_done) begin
                        // FFN layer 1: hidden = ReLU(x * W1 + b1)
                        for (fj = 0; fj < FFN_DIM; fj = fj + 1) begin
                            ffn_accum = 0;
                            for (fi = 0; fi < EMBED_DIM; fi = fi + 1) begin
                                ffn_product = $signed(ln2_out[fi*DATA_WIDTH +: DATA_WIDTH]) *
                                    $signed(ffn_w1_flat[(fi*FFN_DIM+fj)*DATA_WIDTH +: DATA_WIDTH]);
                                if (ln2_out[fi*DATA_WIDTH +: DATA_WIDTH] != 0)
                                    ffn_accum = ffn_accum + ffn_product;
                                else
                                    block_zero_skips <= block_zero_skips + 1;
                            end
                            // Add bias + ReLU
                            ffn_accum = ffn_accum[DATA_WIDTH+7:8] +
                                $signed(ffn_b1_flat[fj*DATA_WIDTH +: DATA_WIDTH]);
                            // ReLU
                            if (ffn_accum < 0)
                                ffn_hidden[fj] = 0;  // ReLU zero — creates sparsity!
                            else
                                ffn_hidden[fj] = ffn_accum[DATA_WIDTH-1:0];
                        end

                        // FFN layer 2: out = hidden * W2 + b2
                        for (fj = 0; fj < EMBED_DIM; fj = fj + 1) begin
                            ffn_accum = 0;
                            for (fi = 0; fi < FFN_DIM; fi = fi + 1) begin
                                ffn_product = ffn_hidden[fi] *
                                    $signed(ffn_w2_flat[(fi*EMBED_DIM+fj)*DATA_WIDTH +: DATA_WIDTH]);
                                if (ffn_hidden[fi] != 0)
                                    ffn_accum = ffn_accum + ffn_product;
                                else
                                    block_zero_skips <= block_zero_skips + 1;
                            end
                            ffn_out_buf[fj] = ffn_accum[DATA_WIDTH+7:8] +
                                $signed(ffn_b2_flat[fj*DATA_WIDTH +: DATA_WIDTH]);
                        end
                        ffn_done <= 1;
                    end else begin
                        state <= RESID2;
                    end
                end

                RESID2: begin
                    // Residual connection: after_attn + FFN(LN2(after_attn))
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                            $signed(after_attn[i*DATA_WIDTH +: DATA_WIDTH]) +
                            ffn_out_buf[i];
                    state <= DONE;
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end
            endcase
        end
    end

endmodule
