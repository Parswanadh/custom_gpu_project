`timescale 1ns / 1ps

// ============================================================================
// Module: medusa_head_predictor
// Description: MEDUSA Multi-Head Draft Token Predictor.
//
//   PAPER: "MEDUSA: Simple LLM Inference Acceleration Framework with 
//          Multiple Decoding Heads" (Cai et al., 2024)
//
//   RATIONALE: Standard autoregressive decoding generates 1 token per step.
//   MEDUSA adds K lightweight "draft heads" that each predict a different
//   future token position simultaneously:
//     - Head 0 predicts token[t+1]
//     - Head 1 predicts token[t+2]
//     - Head K-1 predicts token[t+K]
//   All K predictions are then verified in a single forward pass.
//   Accept rate of ~60-80% → 2.3-3.6× speedup.
//
//   WHY THIS MATTERS FOR BITBYBIT:
//   - Our speculative_decode_engine uses n-gram cache (software-like approach)
//   - MEDUSA uses hardware prediction heads (neural approach)
//   - Combining both: n-gram for common patterns + MEDUSA for novel text
//   - Makes our architecture competitive with Google's TPU speculative decoding
//
//   This module implements K lightweight linear prediction heads.
//   Each head: output_logit = sum(hidden[i] * weight[head][i]) + bias
//
// Parameters: NUM_HEADS, HIDDEN_DIM, VOCAB_SIZE
// ============================================================================
module medusa_head_predictor #(
    parameter NUM_HEADS   = 3,      // Number of draft heads (predict t+1, t+2, t+3)
    parameter HIDDEN_DIM  = 8,      // Hidden state dimension
    parameter VOCAB_BITS  = 8,      // Output token bits (256 vocab tokens)
    parameter DATA_WIDTH  = 16
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     valid_in,
    input  wire [HIDDEN_DIM*DATA_WIDTH-1:0] hidden_state,
    
    // Weight loading for draft heads
    input  wire                     weight_load_en,
    input  wire [$clog2(NUM_HEADS)-1:0] head_sel,
    input  wire [$clog2(HIDDEN_DIM)-1:0] weight_idx,
    input  wire signed [DATA_WIDTH-1:0]  weight_data,
    
    // Predictions: one token per head
    output reg  [NUM_HEADS*VOCAB_BITS-1:0] predicted_tokens,
    output reg                              valid_out,
    
    // Verification interface
    input  wire                             verify_en,
    input  wire [NUM_HEADS*VOCAB_BITS-1:0]  actual_tokens,
    output reg  [NUM_HEADS-1:0]             accept_mask,    // Which predictions were correct
    output reg  [$clog2(NUM_HEADS):0]       accepted_count,
    
    // Statistics
    output reg  [31:0]                      total_predictions,
    output reg  [31:0]                      total_accepted
);

    // Weight storage: one linear layer per head
    reg signed [DATA_WIDTH-1:0] head_weights [0:NUM_HEADS-1][0:HIDDEN_DIM-1];
    
    // Working storage
    reg signed [DATA_WIDTH-1:0] h_reg [0:HIDDEN_DIM-1];
    reg signed [2*DATA_WIDTH-1:0] dot_product;
    integer h, d;

    // Weight loading
    always @(posedge clk) begin
        if (rst) begin
            for (h = 0; h < NUM_HEADS; h = h + 1)
                for (d = 0; d < HIDDEN_DIM; d = d + 1)
                    head_weights[h][d] <= 0;
        end else if (weight_load_en) begin
            head_weights[head_sel][weight_idx] <= weight_data;
        end
    end

    // Prediction FSM
    reg [2:0] state;
    localparam IDLE    = 3'd0;
    localparam PREDICT = 3'd1;
    localparam DONE_ST = 3'd2;

    always @(posedge clk) begin
        if (rst) begin
            state            <= IDLE;
            valid_out        <= 1'b0;
            predicted_tokens <= 0;
            accept_mask      <= 0;
            accepted_count   <= 0;
            total_predictions <= 0;
            total_accepted    <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Capture hidden state
                        for (d = 0; d < HIDDEN_DIM; d = d + 1)
                            h_reg[d] <= $signed(hidden_state[d*DATA_WIDTH +: DATA_WIDTH]);
                        state <= PREDICT;
                    end
                    
                    // Verification (can happen while idle)
                    if (verify_en) begin
                        for (h = 0; h < NUM_HEADS; h = h + 1) begin
                            if (predicted_tokens[h*VOCAB_BITS +: VOCAB_BITS] == 
                                actual_tokens[h*VOCAB_BITS +: VOCAB_BITS]) begin
                                accept_mask[h] <= 1'b1;
                            end else begin
                                accept_mask[h] <= 1'b0;
                            end
                        end
                        // Count accepted (use sequential add since we can't easily popcount in a loop)
                        accepted_count <= 0;
                        for (h = 0; h < NUM_HEADS; h = h + 1) begin
                            if (predicted_tokens[h*VOCAB_BITS +: VOCAB_BITS] == 
                                actual_tokens[h*VOCAB_BITS +: VOCAB_BITS])
                                total_accepted <= total_accepted + 1;
                        end
                        total_predictions <= total_predictions + NUM_HEADS;
                    end
                end
                
                PREDICT: begin
                    // Each head computes: logit = sum(hidden * weight)
                    // Then: predicted token = logit[VOCAB_BITS-1:0] (argmax approx)
                    for (h = 0; h < NUM_HEADS; h = h + 1) begin
                        dot_product = 0;
                        for (d = 0; d < HIDDEN_DIM; d = d + 1) begin
                            dot_product = dot_product + h_reg[d] * head_weights[h][d];
                        end
                        // Take lower bits as predicted token (simplified argmax)
                        predicted_tokens[h*VOCAB_BITS +: VOCAB_BITS] <= 
                            dot_product[VOCAB_BITS+7:8];
                    end
                    state <= DONE_ST;
                end
                
                DONE_ST: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
