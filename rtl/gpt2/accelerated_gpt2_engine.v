// ============================================================================
// Module: accelerated_gpt2_engine
// Description: Complete GPT-2 inference engine using ACCELERATED components.
//   token_in → Embedding → N × AcceleratedTransformerBlock → LayerNorm → Argmax
//
//   FIXES APPLIED:
//     - Issue #8:  Per-layer weight support via weight banks (indexed by layer_idx)
//     - Updated for new gpu_core interface (signed, acc_clear)
//     - Per-layer LN parameter banks
//
// Note: The accelerated_transformer_block still takes flat weight wires for
//   the attention matrices (wq/wk/wv/wo_flat). In a fully fixed design these
//   would be loaded via SRAM too. The LN parameters are now per-layer.
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

    // Per-layer LN parameter loading (Issue #8)
    input  wire                               load_ln_en,
    input  wire [$clog2(NUM_LAYERS):0]        load_layer_idx,
    input  wire                               load_ln_sel,       // 0=LN1, 1=LN2
    input  wire                               load_ln_is_gamma,
    input  wire [$clog2(EMBED_DIM)-1:0]       load_ln_dim,
    input  wire signed [DATA_WIDTH-1:0]       load_ln_data,

    // Transformer weights (per layer — host must load each layer's weights before running that layer)
    // In a production design, these would come from the weight SRAM via DMA
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat, wk_flat, wv_flat, wo_flat,
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]   ffn_w1_flat,
    input  wire [FFN_DIM*DATA_WIDTH-1:0]              ffn_b1_flat,
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]    ffn_w2_flat,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            ffn_b2_flat,

    // Inference interface
    input  wire                               valid_in,
    input  wire [$clog2(VOCAB_SIZE)-1:0]      token_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     position_in,
    output reg  [$clog2(VOCAB_SIZE)-1:0]      token_out,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    logits_out, // Debug slice: vocab logits [0:EMBED_DIM-1]
    output reg                                valid_out,

    // Performance counters
    output reg  [31:0]                        total_zero_skips,
    output reg  [31:0]                        total_cycles
);

    // Per-layer LN parameter banks (Issue #8)
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln1_gamma_bank [0:NUM_LAYERS-1];
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln1_beta_bank  [0:NUM_LAYERS-1];
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln2_gamma_bank [0:NUM_LAYERS-1];
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln2_beta_bank  [0:NUM_LAYERS-1];
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln_final_gamma;
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln_final_beta;
    localparam integer Q_FRAC_BITS   = 8;
    localparam integer DOT_ACC_WIDTH = (2*DATA_WIDTH) + ((EMBED_DIM > 1) ? $clog2(EMBED_DIM) : 1);

    // Output projection uses tied token embeddings as the LM head.
    reg signed [DATA_WIDTH-1:0] lm_head_token_bank [0:VOCAB_SIZE-1][0:EMBED_DIM-1];

    integer li;
    always @(posedge clk) begin
        if (rst) begin
            for (li = 0; li < NUM_LAYERS; li = li + 1) begin
                ln1_gamma_bank[li] <= 0;
                ln1_beta_bank[li]  <= 0;
                ln2_gamma_bank[li] <= 0;
                ln2_beta_bank[li]  <= 0;
            end
            ln_final_gamma <= 0;
            ln_final_beta  <= 0;
        end else if (load_ln_en) begin
            if (load_layer_idx < NUM_LAYERS) begin
                if (!load_ln_sel) begin
                    if (load_ln_is_gamma)
                        ln1_gamma_bank[load_layer_idx][load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
                    else
                        ln1_beta_bank[load_layer_idx][load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
                end else begin
                    if (load_ln_is_gamma)
                        ln2_gamma_bank[load_layer_idx][load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
                    else
                        ln2_beta_bank[load_layer_idx][load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
                end
            end else begin
                if (load_ln_is_gamma)
                    ln_final_gamma[load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
                else
                    ln_final_beta[load_ln_dim*DATA_WIDTH +: DATA_WIDTH] <= load_ln_data;
            end
        end
    end

    // Keep a local copy of token embeddings for vocab-logit projection.
    integer lv, ld;
    always @(posedge clk) begin
        if (rst) begin
            for (lv = 0; lv < VOCAB_SIZE; lv = lv + 1)
                for (ld = 0; ld < EMBED_DIM; ld = ld + 1)
                    lm_head_token_bank[lv][ld] <= 0;
        end else if (load_token_emb) begin
            lm_head_token_bank[load_token_idx][load_dim_idx] <= load_emb_data;
        end
    end

    // Current layer params
    reg [$clog2(NUM_LAYERS):0] layer_idx;
    wire [EMBED_DIM*DATA_WIDTH-1:0] cur_ln1_gamma = ln1_gamma_bank[layer_idx];
    wire [EMBED_DIM*DATA_WIDTH-1:0] cur_ln1_beta  = ln1_beta_bank[layer_idx];
    wire [EMBED_DIM*DATA_WIDTH-1:0] cur_ln2_gamma = ln2_gamma_bank[layer_idx];
    wire [EMBED_DIM*DATA_WIDTH-1:0] cur_ln2_beta  = ln2_beta_bank[layer_idx];

    // Internal signals
    wire [EMBED_DIM*DATA_WIDTH-1:0] emb_out;
    wire                            emb_valid;

    reg  [EMBED_DIM*DATA_WIDTH-1:0] current_hidden;
    reg  [EMBED_DIM*DATA_WIDTH-1:0] final_hidden;
    wire [EMBED_DIM*DATA_WIDTH-1:0] block_out;
    wire                            block_valid;
    wire [31:0]                     block_zero_skips;
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln_final_out;
    wire                            ln_final_valid;

    reg                             block_en, ln_final_en;
    reg                             block_valid_d, block_active;
    wire                            block_done_pulse = block_valid & ~block_valid_d;
    reg [$clog2(MAX_SEQ_LEN)-1:0]   seq_position;

    reg [3:0] state;
    localparam IDLE       = 4'd0;
    localparam EMBEDDING  = 4'd1;
    localparam TRANSFORMER= 4'd2;
    localparam FINAL_LN   = 4'd3;
    localparam LOGITS     = 4'd4;
    localparam OUTPUT     = 4'd5;

    integer i;

    // Embedding lookup
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

    // ACCELERATED Transformer block — now with per-layer LN params
    accelerated_transformer_block #(
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),
        .MAX_SEQ_LEN(MAX_SEQ_LEN), .DATA_WIDTH(DATA_WIDTH)
    ) u_block (
        .clk(clk), .rst(rst), .valid_in(block_en),
        .x_in(current_hidden), .seq_pos(seq_position),
        .ln1_gamma(cur_ln1_gamma), .ln1_beta(cur_ln1_beta),
        .ln2_gamma(cur_ln2_gamma), .ln2_beta(cur_ln2_beta),
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
            final_hidden   <= 0;
            block_en       <= 0;
            ln_final_en    <= 0;
            block_valid_d  <= 0;
            block_active   <= 0;
            seq_position   <= 0;
            total_zero_skips <= 0;
            total_cycles   <= 0;
        end else begin
            block_en    <= 0;
            ln_final_en <= 0;
            block_valid_d <= block_valid;
            if (state != IDLE) begin
                total_cycles <= total_cycles + 1;
            end

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    block_active <= 1'b0;
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
                        block_active <= 1'b1;
                    end
                end

                TRANSFORMER: begin
                    if (block_done_pulse && block_active) begin
                        current_hidden <= block_out;
                        total_zero_skips <= total_zero_skips + block_zero_skips;
                        layer_idx <= layer_idx + 1;
                        if (layer_idx + 1 < NUM_LAYERS) begin
                            block_en <= 1'b1;
                            block_active <= 1'b1;
                        end else begin
                            block_active <= 1'b0;
                            ln_final_en <= 1'b1;
                            state <= FINAL_LN;
                        end
                    end
                end

                FINAL_LN: begin
                    if (ln_final_valid) begin
                        final_hidden <= ln_final_out;
                        state <= LOGITS;
                    end
                end

                LOGITS: begin
                    begin : vocab_argmax_block
                        integer vocab_i, dim_i;
                        reg signed [DOT_ACC_WIDTH-1:0] dot_acc;
                        reg signed [DATA_WIDTH-1:0] vocab_logit;
                        reg [$clog2(VOCAB_SIZE)-1:0] max_idx;
                        reg signed [DATA_WIDTH-1:0] max_val;
                        reg [EMBED_DIM*DATA_WIDTH-1:0] logits_debug;

                        logits_debug = 0;
                        dot_acc = 0;
                        for (dim_i = 0; dim_i < EMBED_DIM; dim_i = dim_i + 1)
                            dot_acc = dot_acc +
                                ($signed(final_hidden[dim_i*DATA_WIDTH +: DATA_WIDTH]) *
                                 $signed(lm_head_token_bank[0][dim_i]));
                        vocab_logit = $signed(dot_acc >>> Q_FRAC_BITS);
                        max_val = vocab_logit;
                        max_idx = 0;

                        if (EMBED_DIM > 0)
                            logits_debug[0 +: DATA_WIDTH] = vocab_logit;

                        for (vocab_i = 1; vocab_i < VOCAB_SIZE; vocab_i = vocab_i + 1) begin
                            dot_acc = 0;
                            for (dim_i = 0; dim_i < EMBED_DIM; dim_i = dim_i + 1)
                                dot_acc = dot_acc +
                                    ($signed(final_hidden[dim_i*DATA_WIDTH +: DATA_WIDTH]) *
                                     $signed(lm_head_token_bank[vocab_i][dim_i]));
                            vocab_logit = $signed(dot_acc >>> Q_FRAC_BITS);

                            if (vocab_i < EMBED_DIM)
                                logits_debug[vocab_i*DATA_WIDTH +: DATA_WIDTH] = vocab_logit;

                            if (vocab_logit > max_val) begin
                                max_val = vocab_logit;
                                max_idx = vocab_i[$clog2(VOCAB_SIZE)-1:0];
                            end
                        end

                        logits_out <= logits_debug;
                        token_out  <= max_idx;
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
