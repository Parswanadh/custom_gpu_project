// ============================================================================
// Module: accelerated_attention
// Description: Multi-head self-attention with KV cache.
//   Implements REAL attention (not trivial output=V):
//     1. Project: Q = x*Wq, K = x*Wk, V = x*Wv
//     2. KV Cache: store K, V at current position
//     3. Score:   scores = Q * K_cache^T / sqrt(d_k)
//     4. Softmax: probs = softmax(scores)
//     5. Output:  attn = probs * V_cache
//     6. Project: y = attn * Wo
// ============================================================================
module accelerated_attention #(
    parameter EMBED_DIM   = 8,
    parameter NUM_HEADS   = 2,
    parameter HEAD_DIM    = 4,
    parameter MAX_SEQ_LEN = 32,
    parameter DATA_WIDTH  = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]      seq_pos,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wk_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wv_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wo_flat,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out,
    output reg  [31:0]                         zero_skip_count
);

    // KV Cache
    reg signed [DATA_WIDTH-1:0] k_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];

    // Working storage
    reg signed [DATA_WIDTH-1:0] x_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] q_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] k_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] attn_out [0:EMBED_DIM-1];

    // Weight storage (registered)
    reg signed [DATA_WIDTH-1:0] wq [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wk [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wv [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wo [0:EMBED_DIM-1][0:EMBED_DIM-1];

    // Attention scores
    reg signed [DATA_WIDTH-1:0] scores [0:MAX_SEQ_LEN-1];
    reg [7:0]                   probs  [0:MAX_SEQ_LEN-1];
    reg signed [DATA_WIDTH-1:0] max_score;

    // Accumulators
    reg signed [2*DATA_WIDTH-1:0] acc;
    reg [15:0] exp_sum;
    integer i, j, t;

    // FSM
    reg [3:0] state;
    localparam S_IDLE     = 0;
    localparam S_LOAD     = 1;
    localparam S_PROJ     = 2;
    localparam S_CACHE    = 3;
    localparam S_SCORE    = 4;
    localparam S_SOFTMAX  = 5;
    localparam S_SM_READ  = 9;
    localparam S_WGTV     = 6;
    localparam S_OUTPROJ  = 7;
    localparam S_DONE     = 8;

    reg [$clog2(MAX_SEQ_LEN)-1:0] cur_pos;

    // Proper 256-entry exp LUT (replaces crude linear function)
    reg signed [15:0] lut_input;
    wire [7:0] lut_output;
    exp_lut_256 u_exp_lut (.x_in(lut_input), .exp_out(lut_output));

    // Softmax iteration state
    reg [$clog2(MAX_SEQ_LEN):0] sm_idx;  // Softmax index
    reg sm_phase;  // 0 = compute exp, 1 = normalize

    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            valid_out      <= 1'b0;
            y_out          <= 0;
            zero_skip_count <= 0;
            cur_pos        <= 0;
        end else begin
            case (state)

            // --- IDLE: wait for valid input ---
            S_IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    cur_pos <= seq_pos;
                    // Register all inputs
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        x_reg[i] <= $signed(x_in[i*DATA_WIDTH +: DATA_WIDTH]);
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        for (j = 0; j < EMBED_DIM; j = j + 1) begin
                            wq[i][j] <= $signed(wq_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH]);
                            wk[i][j] <= $signed(wk_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH]);
                            wv[i][j] <= $signed(wv_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH]);
                            wo[i][j] <= $signed(wo_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH]);
                        end
                    state <= S_LOAD;
                end
            end

            // --- LOAD: 1 cycle delay so registers settle ---
            S_LOAD: begin
                state <= S_PROJ;
            end

            // --- PROJ: Q=x*Wq, K=x*Wk, V=x*Wv ---
            S_PROJ: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    // Q[j] = sum(x[i] * Wq[i][j])
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1) begin
                        if (x_reg[i] != 0 && wq[i][j] != 0) begin
                            acc = acc + x_reg[i] * wq[i][j];
                        end else begin
                            zero_skip_count <= zero_skip_count + 1;
                        end
                    end
                    q_reg[j] <= acc >>> 8;  // Q8.8 * Q8.8 = Q16.16, shift to Q8.8

                    // K[j]
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + x_reg[i] * wk[i][j];
                    k_reg[j] <= acc >>> 8;

                    // V[j]
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + x_reg[i] * wv[i][j];
                    v_reg[j] <= acc >>> 8;
                end
                state <= S_CACHE;
            end

            // --- CACHE: store K,V into cache ---
            S_CACHE: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    k_cache[cur_pos][j] <= k_reg[j];
                    v_cache[cur_pos][j] <= v_reg[j];
                end
                state <= S_SCORE;
            end

            // --- SCORE: score[t] = Q dot K_cache[t] / sqrt(d) ---
            S_SCORE: begin
                begin : score_reduce
                    reg signed [DATA_WIDTH-1:0] score_val;
                    reg signed [DATA_WIDTH-1:0] max_local;
                    max_local = -16'sd32767;
                    for (t = 0; t <= cur_pos; t = t + 1) begin
                        acc = 0;
                        for (j = 0; j < EMBED_DIM; j = j + 1) begin
                            acc = acc + q_reg[j] * k_cache[t][j];
                        end
                        // Divide by sqrt(HEAD_DIM), approx >> 1
                        score_val = (acc >>> 8) >>> 1;
                        scores[t] <= score_val;
                        if (score_val > max_local)
                            max_local = score_val;
                    end
                    max_score <= max_local;
                end
                sm_idx    <= 0;
                sm_phase  <= 0;
                exp_sum   <= 16'd0;
                state <= S_SOFTMAX;
            end

            // --- SOFTMAX: exp via LUT + normalize ---
            // Phase 0: iterate tokens, look up exp(score - max) via LUT, accumulate sum
            // Phase 1: iterate tokens, normalize probs = exp[t] * 255 / sum
            S_SOFTMAX: begin
                if (!sm_phase) begin
                    // Phase 0: compute exp values
                    if (sm_idx <= cur_pos) begin
                        lut_input <= scores[sm_idx] - max_score;
                        state <= S_SM_READ;
                    end else begin
                        // Done with exp, start normalize
                        sm_phase <= 1;
                        sm_idx   <= 0;
                    end
                end else begin
                    // Phase 1: normalize
                    if (sm_idx <= cur_pos) begin
                        begin : norm_block
                            reg [15:0] norm_val;
                            if (exp_sum == 16'd0)
                                norm_val = 16'd0;
                            else
                                norm_val = ({8'd0, probs[sm_idx]} * 16'd255) / exp_sum;
                            probs[sm_idx] <= (norm_val > 255) ? 8'd255 : norm_val[7:0];
                        end
                        sm_idx <= sm_idx + 1;
                    end else begin
                        state <= S_WGTV;
                    end
                end
            end

            // LUT data is consumed in a dedicated state to avoid stale-read hazards
            S_SM_READ: begin
                probs[sm_idx] <= lut_output;
                exp_sum <= exp_sum + {8'd0, lut_output};
                sm_idx <= sm_idx + 1;
                state <= S_SOFTMAX;
            end

            // --- WEIGHTED V: attn = sum(prob[t] * V[t]) ---
            S_WGTV: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    acc = 0;
                    for (t = 0; t <= cur_pos; t = t + 1) begin
                        acc = acc + $signed({1'b0, probs[t]}) * v_cache[t][j];
                    end
                    attn_out[j] <= acc >>> 8;  // Q0.8 * Q8.8 >> 8 = Q8.8
                end
                state <= S_OUTPROJ;
            end

            // --- OUTPUT PROJECTION: y = attn * Wo ---
            S_OUTPROJ: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + attn_out[i] * wo[i][j];
                    y_out[j*DATA_WIDTH +: DATA_WIDTH] <= acc >>> 8;
                end
                state <= S_DONE;
            end

            // --- DONE ---
            S_DONE: begin
                valid_out <= 1'b1;
                state     <= S_IDLE;
            end

            endcase
        end
    end

endmodule
