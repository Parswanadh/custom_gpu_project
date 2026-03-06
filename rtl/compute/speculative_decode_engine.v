`timescale 1ns / 1ps

// ============================================================================
// Module: speculative_decode_engine
// Description: Hardware Speculative Decoding Engine.
//   Implements the "SpecInfer" / "Medusa" concept in silicon:
//
//   1. A lightweight draft predictor (n-gram cache) instantly guesses the
//      next K tokens based on the previous 2 tokens (bigram).
//   2. The main engine verifies all K drafts in a single batch.
//   3. If a draft matches, we accept it (free tokens).
//      If it mismatches, we discard it and compute normally.
//
//   This breaks the sequential autoregressive bottleneck —
//   instead of generating 1 token per inference pass, we generate up to
//   K+1 tokens per pass when predictions are correct.
//
//   Architecture:
//     ┌──────────────────────────────────┐
//     │      N-gram Cache (64 entries)    │
//     │  [prev_token] → [draft_0..K-1]   │
//     └──────────────────────────────────┘
//              │ Draft sequence
//     ┌────────▼─────────────────────────┐
//     │     Verification Logic           │
//     │  Compare draft[i] vs actual[i]    │
//     │  Accept matching prefix           │
//     └──────────────────────────────────┘
//
//   Parameters: VOCAB_BITS, DRAFT_LEN, CACHE_DEPTH
// ============================================================================
module speculative_decode_engine #(
    parameter VOCAB_BITS  = 8,     // log2(vocab_size), e.g. 8 for 256 tokens
    parameter DRAFT_LEN   = 3,     // Number of speculative draft tokens
    parameter CACHE_DEPTH = 64,    // Number of n-gram cache entries
    parameter CACHE_ADDR  = 6      // $clog2(CACHE_DEPTH)
)(
    input  wire                     clk,
    input  wire                     rst,
    
    // Draft prediction interface
    input  wire                     predict_valid,      // Request draft prediction
    input  wire [VOCAB_BITS-1:0]    prev_token,         // Previous token (bigram key)
    output reg  [DRAFT_LEN*VOCAB_BITS-1:0] draft_tokens, // K predicted tokens
    output reg                      draft_valid,        // Draft ready
    
    // Cache programming interface (host fills cache with n-gram statistics)
    input  wire                     cache_write_en,
    input  wire [CACHE_ADDR-1:0]    cache_write_addr,
    input  wire [DRAFT_LEN*VOCAB_BITS-1:0] cache_write_data, // K follow-up tokens
    
    // Verification interface
    input  wire                     verify_valid,       // Start verification
    input  wire [DRAFT_LEN*VOCAB_BITS-1:0] actual_tokens, // Actual engine outputs
    output reg  [3:0]               accepted_count,     // How many draft tokens matched
    output reg                      verify_done,
    output reg                      all_accepted,       // All K drafts correct
    
    // Statistics
    output reg  [31:0]              total_predictions,
    output reg  [31:0]              total_accepted,     // Total accepted draft tokens
    output reg  [31:0]              total_rejected      // Total rejected draft tokens
);

    // N-gram cache: indexed by hash of prev_token
    reg [DRAFT_LEN*VOCAB_BITS-1:0] ngram_cache [0:CACHE_DEPTH-1];
    reg                            cache_valid [0:CACHE_DEPTH-1];
    
    // Hash function: simple modular hash for cache lookup
    wire [CACHE_ADDR-1:0] cache_addr = prev_token[CACHE_ADDR-1:0];
    
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            draft_valid       <= 1'b0;
            verify_done       <= 1'b0;
            all_accepted      <= 1'b0;
            accepted_count    <= 4'd0;
            draft_tokens      <= 0;
            total_predictions <= 32'd0;
            total_accepted    <= 32'd0;
            total_rejected    <= 32'd0;
            for (i = 0; i < CACHE_DEPTH; i = i + 1) begin
                ngram_cache[i] <= 0;
                cache_valid[i] <= 1'b0;
            end
        end else begin
            draft_valid <= 1'b0;
            verify_done <= 1'b0;
            
            // ---- Cache Programming ----
            if (cache_write_en) begin
                ngram_cache[cache_write_addr] <= cache_write_data;
                cache_valid[cache_write_addr] <= 1'b1;
            end
            
            // ---- Draft Prediction ----
            // Look up prev_token in cache, output K predicted tokens
            if (predict_valid) begin
                if (cache_valid[cache_addr]) begin
                    draft_tokens <= ngram_cache[cache_addr];
                    draft_valid  <= 1'b1;
                end else begin
                    // Cache miss — output zeros (main engine will compute normally)
                    draft_tokens <= 0;
                    draft_valid  <= 1'b1;
                end
                total_predictions <= total_predictions + 1;
            end
            
            // ---- Verification ----
            // Compare each draft token against the actual engine output
            // Accept the longest matching prefix
            if (verify_valid) begin
                accepted_count <= 4'd0;
                all_accepted   <= 1'b1;
                
                // Sequential prefix match (accept until first mismatch)
                begin : verify_block
                    integer vi;
                    reg mismatch_found;
                    reg [3:0] match_count;
                    mismatch_found = 1'b0;
                    match_count = 4'd0;
                    
                    for (vi = 0; vi < DRAFT_LEN; vi = vi + 1) begin
                        if (!mismatch_found) begin
                            if (draft_tokens[vi*VOCAB_BITS +: VOCAB_BITS] == 
                                actual_tokens[vi*VOCAB_BITS +: VOCAB_BITS]) begin
                                match_count = match_count + 1;
                            end else begin
                                mismatch_found = 1'b1;
                            end
                        end
                    end
                    
                    accepted_count <= match_count;
                    all_accepted   <= (match_count == DRAFT_LEN);
                    total_accepted <= total_accepted + {28'd0, match_count};
                    total_rejected <= total_rejected + (DRAFT_LEN - {28'd0, match_count});
                end
                
                verify_done <= 1'b1;
            end
        end
    end

endmodule
