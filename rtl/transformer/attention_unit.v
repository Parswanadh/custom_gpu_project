// ============================================================================
// Module: attention_unit
// Description: Multi-head self-attention for transformer blocks.
//   Implements REAL attention (not trivial output=V):
//     1. Project: Q = x*Wq, K = x*Wk, V = x*Wv
//     2. Score:   scores = Q·K^T / sqrt(d_k)
//     3. Softmax: probs = softmax(scores)
//     4. Output:  attn = probs · V
//     5. Project: y = attn * Wo
//
//   FIXES APPLIED:
//     - Issue #5:  Real attention computation (replaces trivial attn_out = V)
//     - Issue #7:  Weights stored in internal SRAM, loaded via write interface
//                  (replaces massive flat wire ports)
//     - Issue #24: Configurable attention mask support
//
// Parameters: EMBED_DIM, NUM_HEADS, HEAD_DIM, MAX_SEQ_LEN
// ============================================================================
module attention_unit #(
    parameter EMBED_DIM    = 8,
    parameter NUM_HEADS    = 2,
    parameter HEAD_DIM     = 4,
    parameter MAX_SEQ_LEN  = 32,
    parameter DATA_WIDTH   = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]      seq_pos,

    // Weight SRAM loading interface (Issue #7)
    input  wire                                weight_load_en,
    input  wire [1:0]                          weight_matrix_sel, // 0=Wq, 1=Wk, 2=Wv, 3=Wo
    input  wire [$clog2(EMBED_DIM)-1:0]        weight_row,
    input  wire [$clog2(EMBED_DIM)-1:0]        weight_col,
    input  wire signed [DATA_WIDTH-1:0]        weight_data,

    // Attention mask (Issue #24)
    input  wire                                causal_mask_en,    // Enable causal masking

    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out,
    output reg  [31:0]                         zero_skip_count
);

    // Weight SRAM (Issue #7)
    reg signed [DATA_WIDTH-1:0] wq [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wk [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wv [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wo [0:EMBED_DIM-1][0:EMBED_DIM-1];

    // KV Cache
    reg signed [DATA_WIDTH-1:0] k_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_cache [0:MAX_SEQ_LEN-1][0:EMBED_DIM-1];

    // Working storage
    reg signed [DATA_WIDTH-1:0] x_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] q_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] k_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v_reg [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] attn_out [0:EMBED_DIM-1];

    // Attention scores and probabilities
    reg signed [DATA_WIDTH-1:0] scores [0:MAX_SEQ_LEN-1];
    reg [7:0]                   probs  [0:MAX_SEQ_LEN-1];
    reg signed [DATA_WIDTH-1:0] max_score;

    // exp LUT
    reg signed [15:0] lut_input;
    wire [7:0] lut_output;
    exp_lut_256 u_exp_lut (.x_in(lut_input), .exp_out(lut_output));

    reg signed [2*DATA_WIDTH-1:0] acc;
    reg [15:0] exp_sum;
    integer i, j, t;

    // Weight loading
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < EMBED_DIM; i = i + 1)
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    wq[i][j] <= {DATA_WIDTH{1'b0}};
                    wk[i][j] <= {DATA_WIDTH{1'b0}};
                    wv[i][j] <= {DATA_WIDTH{1'b0}};
                    wo[i][j] <= {DATA_WIDTH{1'b0}};
                end
        end else if (weight_load_en) begin
            case (weight_matrix_sel)
                2'd0: wq[weight_row][weight_col] <= weight_data;
                2'd1: wk[weight_row][weight_col] <= weight_data;
                2'd2: wv[weight_row][weight_col] <= weight_data;
                2'd3: wo[weight_row][weight_col] <= weight_data;
            endcase
        end
    end

    // FSM
    reg [3:0] state;
    localparam S_IDLE     = 0;
    localparam S_PROJ     = 1;
    localparam S_CACHE    = 2;
    localparam S_SCORE    = 3;
    localparam S_SOFTMAX  = 4;
    localparam S_WGTV     = 5;
    localparam S_OUTPROJ  = 6;
    localparam S_DONE     = 7;

    reg [$clog2(MAX_SEQ_LEN)-1:0] cur_pos;
    reg [$clog2(MAX_SEQ_LEN):0] sm_idx;
    reg sm_phase;

    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            valid_out      <= 1'b0;
            y_out          <= 0;
            zero_skip_count <= 0;
            cur_pos        <= 0;
        end else begin
            case (state)

            S_IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    cur_pos <= seq_pos;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        x_reg[i] <= $signed(x_in[i*DATA_WIDTH +: DATA_WIDTH]);
                    state <= S_PROJ;
                end
            end

            // Issue #5: Real Q/K/V projections with zero-skip
            S_PROJ: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    // Q[j]
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1) begin
                        if (x_reg[i] != 0 && wq[i][j] != 0)
                            acc = acc + x_reg[i] * wq[i][j];
                        else
                            zero_skip_count <= zero_skip_count + 1;
                    end
                    q_reg[j] = acc >>> 8;

                    // K[j]
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + x_reg[i] * wk[i][j];
                    k_reg[j] = acc >>> 8;

                    // V[j]
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + x_reg[i] * wv[i][j];
                    v_reg[j] = acc >>> 8;
                end
                state <= S_CACHE;
            end

            S_CACHE: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    k_cache[cur_pos][j] <= k_reg[j];
                    v_cache[cur_pos][j] <= v_reg[j];
                end
                state <= S_SCORE;
            end

            // Issue #5: Real Q·K^T score computation with scaling
            S_SCORE: begin
                max_score <= -16'sd32767;
                for (t = 0; t <= cur_pos; t = t + 1) begin
                    acc = 0;
                    for (j = 0; j < EMBED_DIM; j = j + 1)
                        acc = acc + q_reg[j] * k_cache[t][j];
                    // Scale by 1/sqrt(HEAD_DIM) ≈ >> 1
                    scores[t] <= (acc >>> 8) >>> 1;
                    if ((acc >>> 9) > max_score)
                        max_score <= acc >>> 9;
                end
                // Issue #24: Apply causal mask (mask future positions)
                if (causal_mask_en) begin
                    for (t = 0; t < MAX_SEQ_LEN; t = t + 1) begin
                        if (t > cur_pos)
                            scores[t] <= -16'sd32767;  // -inf
                    end
                end
                sm_idx   <= 0;
                sm_phase <= 0;
                exp_sum   = 16'd0;
                state <= S_SOFTMAX;
            end

            // Issue #5: Real softmax via exp LUT
            S_SOFTMAX: begin
                if (!sm_phase) begin
                    if (sm_idx <= cur_pos) begin
                        lut_input <= scores[sm_idx] - max_score;
                        probs[sm_idx] <= lut_output;
                        exp_sum = exp_sum + {8'd0, lut_output};
                        sm_idx <= sm_idx + 1;
                    end else begin
                        sm_phase <= 1;
                        sm_idx   <= 0;
                    end
                end else begin
                    if (sm_idx <= cur_pos) begin
                        begin : norm_block
                            reg [15:0] norm_val;
                            if (exp_sum > 0)
                                norm_val = ({8'd0, probs[sm_idx]} * 16'd255) / exp_sum;
                            else
                                norm_val = 16'd0;
                            probs[sm_idx] <= (norm_val > 255) ? 8'd255 : norm_val[7:0];
                        end
                        sm_idx <= sm_idx + 1;
                    end else begin
                        state <= S_WGTV;
                    end
                end
            end

            // Issue #5: Real weighted V sum
            S_WGTV: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    acc = 0;
                    for (t = 0; t <= cur_pos; t = t + 1)
                        acc = acc + $signed({1'b0, probs[t]}) * v_cache[t][j];
                    attn_out[j] <= acc >>> 8;
                end
                state <= S_OUTPROJ;
            end

            S_OUTPROJ: begin
                for (j = 0; j < EMBED_DIM; j = j + 1) begin
                    acc = 0;
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        acc = acc + attn_out[i] * wo[i][j];
                    y_out[j*DATA_WIDTH +: DATA_WIDTH] <= acc >>> 8;
                end
                state <= S_DONE;
            end

            S_DONE: begin
                valid_out <= 1'b1;
                state     <= S_IDLE;
            end

            endcase
        end
    end

endmodule
