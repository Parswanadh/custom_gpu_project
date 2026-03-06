`timescale 1ns / 1ps

// ============================================================================
// Module: grouped_query_attention
// Description: Grouped Query Attention (GQA) Hardware Unit.
//
//   PAPER: "GQA: Training Generalized Multi-Query Transformer Models" 
//          (Ainslie et al., Google, 2023) + ISOCC 2025 FPGA paper
//
//   RATIONALE: Standard Multi-Head Attention (MHA) uses separate K,V per head.
//   GQA shares K,V across groups of query heads:
//     - MHA: 8 Q heads × 8 K heads × 8 V heads = 24 head memories
//     - GQA: 8 Q heads × 2 K heads × 2 V heads = 12 head memories
//     Result: 4× KV cache reduction.
//
//   WHY THIS MATTERS FOR BITBYBIT:
//   - Llama 2 70B uses GQA (8 KV heads for 64 Q heads = 8× savings)
//   - Mistral 7B uses GQA (8 KV heads for 32 Q heads = 4× savings)
//   - Our PagedAttention MMU benefits directly from reduced KV size
//   - Makes BitbyBit architecture-compatible with ALL modern LLMs
//
//   This module computes attention for one token position across G groups.
//   Each group has (NUM_Q_HEADS/NUM_KV_HEADS) query heads sharing one K,V.
//
// Parameters: EMBED_DIM, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN
// ============================================================================
module grouped_query_attention #(
    parameter EMBED_DIM    = 8,
    parameter NUM_Q_HEADS  = 4,     // Total query heads
    parameter NUM_KV_HEADS = 2,     // KV heads (shared across groups)
    parameter HEAD_DIM     = 4,     // Dimension per head (EMBED_DIM / NUM_Q_HEADS)
    parameter MAX_SEQ_LEN  = 16,
    parameter DATA_WIDTH   = 16
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     valid_in,
    
    // Q vectors: one per query head
    input  wire [NUM_Q_HEADS*HEAD_DIM*DATA_WIDTH-1:0] q_heads,
    
    // K,V vectors: one per KV head (shared across query groups)
    input  wire [NUM_KV_HEADS*HEAD_DIM*DATA_WIDTH-1:0] k_heads,
    input  wire [NUM_KV_HEADS*HEAD_DIM*DATA_WIDTH-1:0] v_heads,
    
    // Score for the current token (simplified: just dot-product Q·K)
    output reg  [NUM_Q_HEADS*DATA_WIDTH-1:0] attention_scores,
    output reg                               valid_out,
    
    // Statistics
    output reg  [15:0] kv_memory_saved   // How many KV entries saved vs MHA
);

    // Group mapping: query head h maps to KV head (h * NUM_KV_HEADS / NUM_Q_HEADS)
    // For 4 Q heads, 2 KV heads: Q[0,1] → KV[0], Q[2,3] → KV[1]
    
    localparam HEADS_PER_GROUP = NUM_Q_HEADS / NUM_KV_HEADS;  // Queries sharing each KV

    integer qh, kv_idx, d;
    reg signed [2*DATA_WIDTH-1:0] dot_product;
    reg signed [DATA_WIDTH-1:0] q_val, k_val;

    always @(posedge clk) begin
        if (rst) begin
            valid_out        <= 1'b0;
            attention_scores <= 0;
            kv_memory_saved  <= 0;
        end else begin
            valid_out <= 1'b0;
            
            if (valid_in) begin
                // For each query head, compute attention with its SHARED KV head
                for (qh = 0; qh < NUM_Q_HEADS; qh = qh + 1) begin
                    // Which KV head does this query belong to?
                    kv_idx = qh / HEADS_PER_GROUP;
                    
                    // Compute dot product: Q[qh] · K[kv_idx]
                    dot_product = 0;
                    for (d = 0; d < HEAD_DIM; d = d + 1) begin
                        q_val = $signed(q_heads[(qh*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH]);
                        k_val = $signed(k_heads[(kv_idx*HEAD_DIM + d)*DATA_WIDTH +: DATA_WIDTH]);
                        dot_product = dot_product + q_val * k_val;
                    end
                    
                    // Scale by 1/sqrt(HEAD_DIM) ≈ >> 1 for HEAD_DIM=4
                    attention_scores[qh*DATA_WIDTH +: DATA_WIDTH] <= (dot_product >>> 8) >>> 1;
                end
                
                // KV memory saved: (NUM_Q_HEADS - NUM_KV_HEADS) × HEAD_DIM × MAX_SEQ_LEN entries
                kv_memory_saved <= (NUM_Q_HEADS - NUM_KV_HEADS) * HEAD_DIM * MAX_SEQ_LEN;
                valid_out <= 1'b1;
            end
        end
    end

endmodule
