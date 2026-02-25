// ============================================================================
// Module: embedding_lookup
// Description: Token embedding + positional embedding lookup.
//   Stores embedding tables in internal RAM, outputs combined embedding.
//   embedding_out = token_embedding[token_id] + position_embedding[position]
// Parameters: VOCAB_SIZE, MAX_SEQ_LEN, EMBED_DIM, DATA_WIDTH
// ============================================================================
module embedding_lookup #(
    parameter VOCAB_SIZE   = 16,    // Vocabulary size (small for simulation)
    parameter MAX_SEQ_LEN  = 8,     // Max sequence length
    parameter EMBED_DIM    = 4,     // Embedding dimension
    parameter DATA_WIDTH   = 16     // Q8.8 fixed-point
)(
    input  wire                               clk,
    input  wire                               rst,
    // Embedding table loading
    input  wire                               load_token_emb,
    input  wire [$clog2(VOCAB_SIZE)-1:0]      load_token_idx,
    input  wire [$clog2(EMBED_DIM)-1:0]       load_dim_idx,
    input  wire signed [DATA_WIDTH-1:0]       load_data,
    input  wire                               load_pos_emb,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     load_pos_idx,
    // Inference
    input  wire                               valid_in,
    input  wire [$clog2(VOCAB_SIZE)-1:0]      token_id,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]     position,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    emb_out,    // Combined embedding
    output reg                                valid_out
);

    // Embedding tables
    reg signed [DATA_WIDTH-1:0] token_table [0:VOCAB_SIZE-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] pos_table   [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];

    integer i;

    // Table loading
    always @(posedge clk) begin
        if (rst) begin
            // Initialize tables to zero
            for (i = 0; i < VOCAB_SIZE * EMBED_DIM; i = i + 1)
                token_table[i / EMBED_DIM][i % EMBED_DIM] <= 0;
            for (i = 0; i < MAX_SEQ_LEN * EMBED_DIM; i = i + 1)
                pos_table[i / EMBED_DIM][i % EMBED_DIM] <= 0;
        end else begin
            if (load_token_emb)
                token_table[load_token_idx][load_dim_idx] <= load_data;
            if (load_pos_emb)
                pos_table[load_pos_idx][load_dim_idx] <= load_data;
        end
    end

    // Embedding lookup: 1-cycle latency
    always @(posedge clk) begin
        if (rst) begin
            emb_out   <= 0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            for (i = 0; i < EMBED_DIM; i = i + 1)
                emb_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                    token_table[token_id][i] + pos_table[position][i];
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
