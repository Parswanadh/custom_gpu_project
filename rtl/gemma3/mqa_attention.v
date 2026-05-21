`timescale 1ns / 1ps
// ============================================================================
// Module: mqa_attention
// Description: Multi-Query Attention (MQA) for Gemma 3 architecture.
//   - NUM_Q_HEADS query heads share 1 key/value head (MQA pattern).
//   - Includes QK-Norm: RMSNorm applied to Q and K before dot product.
//   - RoPE-ready: expects pre-rotated Q/K inputs.
//   - Outputs softmax-weighted value vectors per head, then concatenated.
//
//   Dataflow:
//     1. Q projection: input -> Q (NUM_Q_HEADS × HEAD_DIM)
//     2. K projection: input -> K (1 × HEAD_DIM)
//     3. V projection: input -> V (1 × HEAD_DIM)
//     4. QK-Norm: RMSNorm(Q), RMSNorm(K)
//     5. Score = Q · K^T / sqrt(HEAD_DIM)
//     6. Probs = softmax(Score)
//     7. Out = Probs · V
//     8. Concat all heads -> O projection
//
//   Parameterized for simulation (small dims) and FPGA (full Gemma3 dims).
// ============================================================================

module mqa_attention #(
    parameter DIM         = 8,      // Model hidden dim (Gemma3: 640)
    parameter NUM_Q_HEADS = 4,      // Number of query heads (Gemma3: 4)
    parameter HEAD_DIM    = 4,      // Dimension per head (Gemma3: 256)
    parameter DATA_W      = 16     // Data width (Q8.8 fixed-point)
)(
    input  wire                           clk,
    input  wire                           rst,

    // Handshake
    input  wire                           valid_in,
    output reg                            valid_out,

    // Input: pre-projected Q, K, V vectors
    // Q: NUM_Q_HEADS × HEAD_DIM values
    // K: 1 × HEAD_DIM values (shared across all Q heads)
    // V: 1 × HEAD_DIM values (shared across all Q heads)
    input  wire [NUM_Q_HEADS*HEAD_DIM*DATA_W-1:0]  q_in,
    input  wire [HEAD_DIM*DATA_W-1:0]               k_in,
    input  wire [HEAD_DIM*DATA_W-1:0]               v_in,

    // Output: concatenated attention output (NUM_Q_HEADS × HEAD_DIM)
    output reg  [NUM_Q_HEADS*HEAD_DIM*DATA_W-1:0]   attn_out,

    // Diagnostic
    output reg  [15:0]                               cycles_used
);

    // =====================================================================
    // FSM States
    // =====================================================================
    localparam S_IDLE      = 3'd0;
    localparam S_QK_NORM   = 3'd1;  // RMSNorm on Q and K
    localparam S_DOT       = 3'd2;  // Q · K^T dot products
    localparam S_SCALE     = 3'd3;  // Divide by sqrt(HEAD_DIM)
    localparam S_SOFTMAX   = 3'd4;  // Softmax across heads (trivial for single KV)
    localparam S_VALUE     = 3'd5;  // Weighted V multiplication
    localparam S_DONE      = 3'd6;

    reg [2:0] state;
    reg [15:0] cycle_cnt;

    // Internal storage
    reg signed [DATA_W-1:0] q_buf [0:NUM_Q_HEADS*HEAD_DIM-1];
    reg signed [DATA_W-1:0] k_buf [0:HEAD_DIM-1];
    reg signed [DATA_W-1:0] v_buf [0:HEAD_DIM-1];

    // QK-Norm accumulators (sum of squares for RMSNorm)
    reg [31:0] q_ss [0:NUM_Q_HEADS-1];  // Sum of squares per Q head
    reg [31:0] k_ss;                     // Sum of squares for K

    // Dot product scores
    reg signed [31:0] scores [0:NUM_Q_HEADS-1];

    // Softmax probabilities (Q8.8)
    reg signed [DATA_W-1:0] probs [0:NUM_Q_HEADS-1];

    // Attention output buffer
    reg signed [DATA_W-1:0] out_buf [0:NUM_Q_HEADS*HEAD_DIM-1];

    // Loop counters
    reg [7:0] head_idx;
    reg [7:0] dim_idx;
    reg [1:0] sub_state;

    // Inverse sqrt approximation for scaling
    // For HEAD_DIM=4: 1/sqrt(4) = 0.5 -> Q8.8 = 128
    // For HEAD_DIM=256: 1/sqrt(256) = 0.0625 -> Q8.8 = 16
    wire signed [DATA_W-1:0] inv_sqrt_hd =
        (HEAD_DIM <= 4)   ? 16'sd128 :
        (HEAD_DIM <= 16)  ? 16'sd64  :
        (HEAD_DIM <= 64)  ? 16'sd32  :
        (HEAD_DIM <= 256) ? 16'sd16  :
                            16'sd8;

    // RMSNorm inverse sqrt LUT (simplified — uses upper bits of sum-of-squares)
    function signed [DATA_W-1:0] rms_inv_sqrt;
        input [31:0] sum_sq;
        input [7:0]  vec_len;
        reg [31:0] mean_sq;
        begin
            mean_sq = sum_sq / (vec_len > 0 ? vec_len : 1);
            // Approximate 1/sqrt(mean_sq) in Q8.8
            if (mean_sq == 0)           rms_inv_sqrt = 16'sd256; // = 1.0 in Q8.8
            else if (mean_sq < 32'd64)  rms_inv_sqrt = 16'sd256;
            else if (mean_sq < 32'd256) rms_inv_sqrt = 16'sd128;
            else if (mean_sq < 32'd1024) rms_inv_sqrt = 16'sd64;
            else if (mean_sq < 32'd4096) rms_inv_sqrt = 16'sd32;
            else if (mean_sq < 32'd16384) rms_inv_sqrt = 16'sd16;
            else if (mean_sq < 32'd65536) rms_inv_sqrt = 16'sd8;
            else                          rms_inv_sqrt = 16'sd4;
        end
    endfunction

    integer ii;

    // =====================================================================
    // Latch inputs on valid_in
    // =====================================================================
    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            valid_out <= 1'b0;
            cycle_cnt <= 16'd0;
            head_idx  <= 8'd0;
            dim_idx   <= 8'd0;
            sub_state <= 2'd0;
            k_ss      <= 32'd0;
            for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                scores[ii] <= 32'sd0;
                probs[ii]  <= {DATA_W{1'b0}};
                q_ss[ii]   <= 32'd0;
            end
            cycles_used <= 16'd0;
        end else begin
            valid_out <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (valid_in) begin
                        cycle_cnt <= 16'd0;
                        // Latch Q
                        for (ii = 0; ii < NUM_Q_HEADS * HEAD_DIM; ii = ii + 1)
                            q_buf[ii] <= $signed(q_in[ii*DATA_W +: DATA_W]);
                        // Latch K
                        for (ii = 0; ii < HEAD_DIM; ii = ii + 1)
                            k_buf[ii] <= $signed(k_in[ii*DATA_W +: DATA_W]);
                        // Latch V
                        for (ii = 0; ii < HEAD_DIM; ii = ii + 1)
                            v_buf[ii] <= $signed(v_in[ii*DATA_W +: DATA_W]);
                        // Initialize accumulators
                        for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1)
                            q_ss[ii] <= 32'd0;
                        k_ss <= 32'd0;
                        head_idx <= 8'd0;
                        dim_idx  <= 8'd0;
                        state <= S_QK_NORM;
                    end
                end

                // ---------------------------------------------------------
                // QK-Norm: compute sum-of-squares, then scale
                // ---------------------------------------------------------
                S_QK_NORM: begin
                    cycle_cnt <= cycle_cnt + 1;
                    if (sub_state == 2'd0) begin
                        // Accumulate sum of squares for all Q heads and K
                        for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                            q_ss[ii] <= q_ss[ii] +
                                ($signed(q_buf[ii * HEAD_DIM + dim_idx]) *
                                 $signed(q_buf[ii * HEAD_DIM + dim_idx]));
                        end
                        k_ss <= k_ss + ($signed(k_buf[dim_idx]) * $signed(k_buf[dim_idx]));

                        if (dim_idx == HEAD_DIM - 1) begin
                            dim_idx   <= 8'd0;
                            sub_state <= 2'd1;
                        end else begin
                            dim_idx <= dim_idx + 1;
                        end
                    end else begin
                        // Apply RMSNorm scaling to Q and K
                        for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                            q_buf[ii * HEAD_DIM + dim_idx] <=
                                ($signed(q_buf[ii * HEAD_DIM + dim_idx]) *
                                 rms_inv_sqrt(q_ss[ii], HEAD_DIM[7:0])) >>> 8;
                        end
                        k_buf[dim_idx] <=
                            ($signed(k_buf[dim_idx]) * rms_inv_sqrt(k_ss, HEAD_DIM[7:0])) >>> 8;

                        if (dim_idx == HEAD_DIM - 1) begin
                            dim_idx   <= 8'd0;
                            sub_state <= 2'd0;
                            head_idx  <= 8'd0;
                            for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1)
                                scores[ii] <= 32'sd0;
                            state <= S_DOT;
                        end else begin
                            dim_idx <= dim_idx + 1;
                        end
                    end
                end

                // ---------------------------------------------------------
                // Dot product: score[h] = sum(Q[h][d] * K[d]) for each head
                // ---------------------------------------------------------
                S_DOT: begin
                    cycle_cnt <= cycle_cnt + 1;
                    for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                        scores[ii] <= scores[ii] +
                            ($signed(q_buf[ii * HEAD_DIM + dim_idx]) * $signed(k_buf[dim_idx]));
                    end
                    if (dim_idx == HEAD_DIM - 1) begin
                        dim_idx <= 8'd0;
                        state <= S_SCALE;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end

                // ---------------------------------------------------------
                // Scale by 1/sqrt(HEAD_DIM)
                // ---------------------------------------------------------
                S_SCALE: begin
                    cycle_cnt <= cycle_cnt + 1;
                    for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                        scores[ii] <= (scores[ii] * inv_sqrt_hd) >>> 8;
                    end
                    state <= S_SOFTMAX;
                end

                // ---------------------------------------------------------
                // Softmax (simplified for single KV head — each score is
                // independent probability since there's only one K vector)
                // For proper attention over sequence, this would be over
                // context positions. Here we normalize across heads.
                // ---------------------------------------------------------
                S_SOFTMAX: begin
                    cycle_cnt <= cycle_cnt + 1;
                    // For MQA with single K: each head gets its own
                    // attention weight. Clamp to [0, 256] (0.0 to 1.0 in Q8.8).
                    // Simple sigmoid-like activation for single-position attention.
                    for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                        if (scores[ii] > 32'sd256)
                            probs[ii] <= 16'sd256;  // 1.0 in Q8.8
                        else if (scores[ii] < -32'sd256)
                            probs[ii] <= 16'sd0;
                        else
                            probs[ii] <= scores[ii][DATA_W-1:0];
                    end
                    dim_idx  <= 8'd0;
                    head_idx <= 8'd0;
                    state <= S_VALUE;
                end

                // ---------------------------------------------------------
                // Value weighting: out[h][d] = prob[h] * V[d]
                // ---------------------------------------------------------
                S_VALUE: begin
                    cycle_cnt <= cycle_cnt + 1;
                    for (ii = 0; ii < NUM_Q_HEADS; ii = ii + 1) begin
                        out_buf[ii * HEAD_DIM + dim_idx] <=
                            ($signed(probs[ii]) * $signed(v_buf[dim_idx])) >>> 8;
                    end
                    if (dim_idx == HEAD_DIM - 1) begin
                        state <= S_DONE;
                    end else begin
                        dim_idx <= dim_idx + 1;
                    end
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    cycle_cnt  <= cycle_cnt + 1;
                    cycles_used <= cycle_cnt + 1;
                    // Pack output
                    for (ii = 0; ii < NUM_Q_HEADS * HEAD_DIM; ii = ii + 1)
                        attn_out[ii*DATA_W +: DATA_W] <= out_buf[ii];
                    valid_out <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
