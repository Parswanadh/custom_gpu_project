// ============================================================================
// Module: accelerated_gpt2_engine
// Description: Complete GPT-2 inference engine using ACCELERATED components.
//   token_in → Embedding → N × AcceleratedTransformerBlock → LayerNorm → Argmax
//
//   Key improvements over original gpt2_engine:
//     - Uses accelerated_attention with KV cache (real multi-token attention)
//     - Uses accelerated_transformer_block (connected to pipeline, zero-skip)
//     - Supports seq_pos for autoregressive generation
//     - Tracks zero-skip count across all layers
//
// Parameters: VOCAB_SIZE, MAX_SEQ_LEN, EMBED_DIM, NUM_LAYERS, etc.
// ============================================================================
module accelerated_gpt2_engine #(
    parameter VOCAB_SIZE   = 16,
    parameter MAX_SEQ_LEN  = 8,
    parameter EMBED_DIM    = 4,
    parameter NUM_HEADS    = 2,
    parameter HEAD_DIM     = 2,
    parameter FFN_DIM      = 8,
    parameter NUM_LAYERS   = 2,
    parameter DATA_WIDTH   = 16
)(
    input  wire                               clk,
    input  wire                               rst,

    // Embedding loading
    input  wire                               load_token_emb,
    input  wire [$clog2(VOCAB_SIZE)-1:0]      load_token_idx,
    input  wire [$clog2(EMBED_DIM)-1:0]       load_dim_idx,
    input  wire signed [DATA_WIDTH-1:0]       load_emb_data,
    input  wire                               load_pos_emb,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     load_pos_idx,

    // Transformer weights (shared across layers for simplicity)
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln1_gamma, ln1_beta,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln2_gamma, ln2_beta,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat, wk_flat, wv_flat, wo_flat,
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]   ffn_w1_flat,
    input  wire [FFN_DIM*DATA_WIDTH-1:0]              ffn_b1_flat,
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]    ffn_w2_flat,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            ffn_b2_flat,

    // Final layer norm
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    ln_final_gamma, ln_final_beta,

    // Inference interface
    input  wire                               valid_in,
    input  wire [$clog2(VOCAB_SIZE)-1:0]      token_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     position_in,
    output reg  [$clog2(VOCAB_SIZE)-1:0]      token_out,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    logits_out,
    output reg                                valid_out,

    // Performance counters (NEW — not in original)
    output reg  [31:0]                        total_zero_skips,
    output reg  [31:0]                        total_cycles
);

    // Internal signals
    wire [EMBED_DIM*DATA_WIDTH-1:0] emb_out;
    wire                            emb_valid;

    reg  [EMBED_DIM*DATA_WIDTH-1:0] current_hidden;
    wire [EMBED_DIM*DATA_WIDTH-1:0] block_out;
    wire                            block_valid;
    wire [31:0]                     block_zero_skips;
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln_final_out;
    wire                            ln_final_valid;

    reg                             block_en, ln_final_en;
    reg [$clog2(MAX_SEQ_LEN)-1:0]   seq_position;

    // State machine
    reg [3:0] state;
    reg [$clog2(NUM_LAYERS):0] layer_idx;

    localparam IDLE       = 4'd0;
    localparam EMBEDDING  = 4'd1;
    localparam TRANSFORMER= 4'd2;
    localparam FINAL_LN   = 4'd3;
    localparam LOGITS     = 4'd4;
    localparam OUTPUT     = 4'd5;

    integer i;

    // Embedding lookup (reused from original)
    embedding_lookup #(
        .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .EMBED_DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)
    ) u_emb (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_data(load_emb_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .valid_in(valid_in && state == IDLE),
        .token_id(token_in), .position(position_in),
        .emb_out(emb_out), .valid_out(emb_valid)
    );

    // ACCELERATED Transformer block (replaces old transformer_block!)
    accelerated_transformer_block #(
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN), .DATA_WIDTH(DATA_WIDTH)
    ) u_block (
        .clk(clk), .rst(rst), .valid_in(block_en),
        .x_in(current_hidden), .seq_pos(seq_position),
        .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
        .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
        .wq_flat(wq_flat), .wk_flat(wk_flat),
        .wv_flat(wv_flat), .wo_flat(wo_flat),
        .ffn_w1_flat(ffn_w1_flat), .ffn_b1_flat(ffn_b1_flat),
        .ffn_w2_flat(ffn_w2_flat), .ffn_b2_flat(ffn_b2_flat),
        .y_out(block_out), .valid_out(block_valid),
        .block_zero_skips(block_zero_skips)
    );

    // Final layer norm
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln_final (
        .clk(clk), .rst(rst), .valid_in(ln_final_en),
        .x_in(current_hidden),
        .gamma_in(ln_final_gamma), .beta_in(ln_final_beta),
        .y_out(ln_final_out), .valid_out(ln_final_valid)
    );

    always @(posedge clk) begin
        if (rst) begin
            state          <= IDLE;
            valid_out      <= 1'b0;
            token_out      <= 0;
            logits_out     <= 0;
            layer_idx      <= 0;
            current_hidden <= 0;
            block_en       <= 0;
            ln_final_en    <= 0;
            seq_position   <= 0;
            total_zero_skips <= 0;
            total_cycles   <= 0;
        end else begin
            block_en    <= 0;
            ln_final_en <= 0;
            total_cycles <= total_cycles + 1;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        seq_position <= position_in;
                        state <= EMBEDDING;
                    end
                end

                EMBEDDING: begin
                    if (emb_valid) begin
                        current_hidden <= emb_out;
                        layer_idx <= 0;
                        state <= TRANSFORMER;
                        block_en <= 1'b1;
                    end
                end

                TRANSFORMER: begin
                    if (block_valid) begin
                        current_hidden <= block_out;
                        total_zero_skips <= total_zero_skips + block_zero_skips;
                        layer_idx <= layer_idx + 1;
                        if (layer_idx + 1 < NUM_LAYERS) begin
                            block_en <= 1'b1;
                        end else begin
                            ln_final_en <= 1'b1;
                            state <= FINAL_LN;
                        end
                    end
                end

                FINAL_LN: begin
                    if (ln_final_valid) begin
                        logits_out <= ln_final_out;
                        state <= LOGITS;
                    end
                end

                LOGITS: begin
                    // Argmax over logits
                    begin : argmax_block
                        reg signed [DATA_WIDTH-1:0] max_val;
                        reg [$clog2(VOCAB_SIZE)-1:0] max_idx;
                        max_val = $signed(logits_out[0 +: DATA_WIDTH]);
                        max_idx = 0;
                        for (i = 1; i < EMBED_DIM; i = i + 1) begin
                            if ($signed(logits_out[i*DATA_WIDTH +: DATA_WIDTH]) > max_val) begin
                                max_val = $signed(logits_out[i*DATA_WIDTH +: DATA_WIDTH]);
                                max_idx = i;
                            end
                        end
                        token_out <= max_idx;
                    end
                    state <= OUTPUT;
                end

                OUTPUT: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
