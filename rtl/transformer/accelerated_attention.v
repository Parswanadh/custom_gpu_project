// ============================================================================
// Module: accelerated_attention
// Description: Multi-head self-attention with KV cache.
//   Unlike the original attention_unit which trivially sets output = V,
//   this module implements REAL attention:
//     1. Project: Q = x*Wq, K = x*Wk, V = x*Wv
//     2. Score:   scores = Q * K_cache^T / sqrt(d_k)
//     3. Softmax: probs = softmax(scores)
//     4. Output:  attn = probs * V_cache
//     5. Project: y = attn * Wo
//
//   KV Cache: stores K and V from all previous tokens so the model
//   can attend to the full sequence history.
//
// Parameters: EMBED_DIM, NUM_HEADS, HEAD_DIM, MAX_SEQ_LEN, DATA_WIDTH
// ============================================================================
module accelerated_attention #(
    parameter EMBED_DIM   = 8,
    parameter NUM_HEADS   = 2,
    parameter HEAD_DIM    = 4,       // EMBED_DIM / NUM_HEADS
    parameter MAX_SEQ_LEN = 32,
    parameter DATA_WIDTH  = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,

    // Current sequence position (for KV cache indexing)
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]      seq_pos,

    // Weight matrices (flattened)
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wk_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wv_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wo_flat,

    // Output
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out,
    output reg  [31:0]                         zero_skip_count
);

    // KV Cache: stores K and V for all past positions
    reg signed [DATA_WIDTH-1:0] k_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] q [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] k_new [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_new [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] attn_out [0:EMBED_DIM-1];

    // Attention scores and probabilities
    reg signed [2*DATA_WIDTH-1:0] scores [0:MAX_SEQ_LEN-1];
    reg [7:0]                     probs  [0:MAX_SEQ_LEN-1];   // Q0.8 format

    // Weight matrices (unpacked)
    reg signed [DATA_WIDTH-1:0] wq [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wk [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wv [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wo [0:EMBED_DIM-1][0:EMBED_DIM-1];

    // Working registers
    reg signed [2*DATA_WIDTH-1:0] accum;
    reg signed [2*DATA_WIDTH-1:0] product;
    reg signed [DATA_WIDTH-1:0]   max_score;
    reg [15:0] exp_sum;
    reg [7:0]  exp_val;
    integer i, j, t;

    // State machine
    reg [3:0] state;
    localparam IDLE       = 4'd0;
    localparam PROJ_QKV   = 4'd1;
    localparam STORE_KV   = 4'd2;
    localparam SCORE      = 4'd3;
    localparam SOFTMAX    = 4'd4;
    localparam WEIGHTED_V = 4'd5;
    localparam OUT_PROJ   = 4'd6;
    localparam DONE       = 4'd7;

    reg [$clog2(MAX_SEQ_LEN)-1:0] current_pos;

    // Exp approximation function for softmax
    function [7:0] exp_approx;
        input signed [DATA_WIDTH-1:0] val;
        reg signed [DATA_WIDTH+3:0] tmp;
        begin
            if (val >= 0)
                exp_approx = 8'd255;
            else if (val < -16'sd2048)
                exp_approx = 8'd1;
            else begin
                tmp = 255 + (val * 89) / 256;
                if (tmp < 1) exp_approx = 8'd1;
                else if (tmp > 255) exp_approx = 8'd255;
                else exp_approx = tmp[7:0];
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state           <= IDLE;
            valid_out       <= 1'b0;
            y_out           <= 0;
            zero_skip_count <= 0;
            current_pos     <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Unpack input and weights
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            x[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                                wq[i][j] <= wq_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wk[i][j] <= wk_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wv[i][j] <= wv_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wo[i][j] <= wo_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                            end
                        current_pos <= seq_pos;
                        state <= PROJ_QKV;
                    end
                end

                PROJ_QKV: begin
                    // Compute Q, K, V projections: mat-vec multiply
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        // Q[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wq[i][j];
                            if (x[i] != 0 && wq[i][j] != 0)
                                accum = accum + product;
                            else
                                zero_skip_count <= zero_skip_count + 1;
                        end
                        q[j] = accum[DATA_WIDTH+7:8];

                        // K[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wk[i][j];
                            accum = accum + product;
                        end
                        k_new[j] = accum[DATA_WIDTH+7:8];

                        // V[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wv[i][j];
                            accum = accum + product;
                        end
                        v_new[j] = accum[DATA_WIDTH+7:8];
                    end
                    state <= STORE_KV;
                end

                STORE_KV: begin
                    // Store new K, V into cache at current position
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        k_cache[current_pos][j] <= k_new[j];
                        v_cache[current_pos][j] <= v_new[j];
                    end
                    state <= SCORE;
                end

                SCORE: begin
                    // Compute attention scores: score[t] = Q · K_cache[t] / sqrt(d_k)
                    // For all positions 0..current_pos
                    max_score = -16'sd32768;
                    for (t = 0; t <= current_pos; t = t + 1) begin
                        accum = 0;
                        for (j = 0; j < EMBED_DIM; j = j + 1) begin
                            product = q[j] * k_cache[t][j];
                            accum = accum + product;
                        end
                        // Divide by sqrt(HEAD_DIM) ≈ shift right
                        scores[t] = accum[DATA_WIDTH+7:8] >>> 1;  // Approximate /sqrt(4)=/2
                        if (scores[t][DATA_WIDTH-1:0] > max_score)
                            max_score = scores[t][DATA_WIDTH-1:0];
                    end
                    // Zero out future positions
                    for (t = current_pos + 1; t < MAX_SEQ_LEN; t = t + 1)
                        scores[t] = 0;
                    state <= SOFTMAX;
                end

                SOFTMAX: begin
                    // Softmax: exp(score - max) / sum(exp(score - max))
                    exp_sum = 0;
                    for (t = 0; t <= current_pos; t = t + 1) begin
                        exp_val = exp_approx(scores[t][DATA_WIDTH-1:0] - max_score);
                        probs[t] = exp_val;
                        exp_sum = exp_sum + {8'd0, exp_val};
                    end
                    // Normalize
                    if (exp_sum > 0) begin
                        for (t = 0; t <= current_pos; t = t + 1) begin
                            probs[t] = ({8'b0, probs[t]} * 16'd256) / exp_sum;
                        end
                    end
                    state <= WEIGHTED_V;
                end

                WEIGHTED_V: begin
                    // attn_out = sum(probs[t] * V_cache[t]) for all t
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        accum = 0;
                        for (t = 0; t <= current_pos; t = t + 1) begin
                            product = probs[t] * v_cache[t][j];
                            accum = accum + product;
                        end
                        attn_out[j] = accum[15:8];  // Scale back from Q0.8 * Q8.8
                    end
                    state <= OUT_PROJ;
                end

                OUT_PROJ: begin
                    // Output projection: y = attn_out * Wo
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = attn_out[i] * wo[i][j];
                            accum = accum + product;
                        end
                        y_out[j*DATA_WIDTH +: DATA_WIDTH] <= accum[DATA_WIDTH+7:8];
                    end
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
