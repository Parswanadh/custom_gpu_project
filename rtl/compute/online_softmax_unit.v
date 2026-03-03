// ============================================================================
// Module: online_softmax_unit
// Description: Single-pass streaming softmax with fused value accumulation.
//
//   Standard softmax requires TWO passes:
//     Pass 1: Find max(scores), compute exp(s_i - max), accumulate sum
//     Pass 2: Normalize each exp by the sum
//   This requires storing ALL N scores in memory.
//
//   The ONLINE softmax processes ONE score at a time (streaming):
//     For each new score s_j arriving with value vector v_j:
//       1. m_new = max(m_old, s_j)
//       2. corr  = exp(m_old - m_new)          ← correction factor
//       3. e_j   = exp(s_j - m_new)            ← new exp term
//       4. denom = denom * corr + e_j           ← update denominator
//       5. acc[d] = acc[d] * corr + e_j * v[d]  ← fuse softmax × V
//     After all scores:  output[d] = acc[d] / denom
//
//   This fuses softmax + weighted-V, eliminating score/prob storage entirely.
//   Uses 2 exp_lut_256 lookups per score.
//
//   Reference: Milakov & Gimelshein, arXiv:1805.02867 (2018)
// ============================================================================
module online_softmax_unit #(
    parameter EMBED_DIM  = 8,
    parameter DATA_WIDTH = 16
)(
    input  wire                               clk,
    input  wire                               rst,

    // Streaming input: feed one (score, value_vector) per cycle
    input  wire                               score_valid,
    input  wire signed [DATA_WIDTH-1:0]       score_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]    value_in,

    // Control
    input  wire                               start,      // Pulse: reset for new softmax
    input  wire                               finalize,   // Pulse: trigger final division

    // Output
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]    result_out,
    output reg                                result_valid
);

    // ---- Running state ----
    reg signed [DATA_WIDTH-1:0]     running_max;
    reg        [2*DATA_WIDTH-1:0]   running_denom;  // Wider to avoid overflow
    reg signed [2*DATA_WIDTH-1:0]   running_acc [0:EMBED_DIM-1];

    // ---- Two exp_lut_256 instances ----
    // LUT A: computes exp(m_old - m_new) — correction factor
    // LUT B: computes exp(s_j - m_new)   — new score term
    // Both are COMBINATIONAL (outputs valid same cycle as inputs change)
    wire signed [15:0] lut_a_in;
    wire        [7:0]  lut_a_out;
    exp_lut_256 u_exp_corr (.x_in(lut_a_in), .exp_out(lut_a_out));

    wire signed [15:0] lut_b_in;
    wire        [7:0]  lut_b_out;
    exp_lut_256 u_exp_score (.x_in(lut_b_in), .exp_out(lut_b_out));

    // ---- Combinational LUT input computation ----
    // These are WIRES — they track the current inputs instantly
    wire new_max_needed = score_valid && ($signed(score_in) > $signed(running_max));

    // When new max: correction = exp(old_max - new_score), new_term = exp(0) = 255
    // When no new max: correction = exp(0) = 255 (no change), new_term = exp(score - max)
    assign lut_a_in = new_max_needed ? (running_max - score_in) : 16'sd0;
    assign lut_b_in = new_max_needed ? 16'sd0 : (score_in - running_max);

    // Unpacked value vector (combinational)
    wire signed [DATA_WIDTH-1:0] v_unpacked [0:EMBED_DIM-1];
    genvar gi;
    generate
        for (gi = 0; gi < EMBED_DIM; gi = gi + 1) begin : unpack_v
            assign v_unpacked[gi] = $signed(value_in[gi*DATA_WIDTH +: DATA_WIDTH]);
        end
    endgenerate

    // ---- Finalize state ----
    reg        finalizing;
    reg [7:0]  fin_dim;

    integer d;

    always @(posedge clk) begin
        if (rst || start) begin
            running_max   <= -16'sd32767;
            running_denom <= 0;
            for (d = 0; d < EMBED_DIM; d = d + 1)
                running_acc[d] <= 0;
            result_valid  <= 1'b0;
            finalizing    <= 1'b0;
            fin_dim       <= 0;
        end else begin

            result_valid <= 1'b0;

            // ====================================================
            // ACCUMULATE: On each valid score, update running state
            // LUT outputs are COMBINATIONAL — available this cycle
            // ====================================================
            if (score_valid) begin
                // Update running max
                if (new_max_needed)
                    running_max <= score_in;

                // lut_a_out = correction factor (exp(m_old - m_new))
                // lut_b_out = new exp term (exp(s_j - m_new))
                // Both are Q0.8 unsigned [0..255] where 255 = 1.0

                // Update denominator: denom = denom * corr/255 + exp_term
                begin : denom_blk
                    reg [2*DATA_WIDTH-1:0] corrected;
                    corrected = (running_denom * {24'd0, lut_a_out}) >> 8;
                    running_denom <= corrected + {24'd0, lut_b_out};
                end

                // Update accumulators: acc[d] = acc[d] * corr/255 + exp_term * v[d]
                for (d = 0; d < EMBED_DIM; d = d + 1) begin : acc_blk
                    reg signed [2*DATA_WIDTH-1:0] corrected_a;
                    reg signed [2*DATA_WIDTH-1:0] new_contrib;
                    // Multiply by correction (Q0.8), shift back
                    corrected_a = (running_acc[d] * $signed({1'b0, lut_a_out})) >>> 8;
                    // New contribution: exp_term × v[d] (Q0.8 × Q8.8 = Q8.16)
                    new_contrib = $signed({1'b0, lut_b_out}) * v_unpacked[d];
                    running_acc[d] <= corrected_a + new_contrib;
                end
            end

            // ====================================================
            // FINALIZE: Divide accumulators by denominator
            // Process one dimension per cycle
            // ====================================================
            if (finalize && !finalizing) begin
                finalizing <= 1'b1;
                fin_dim    <= 0;
            end

            if (finalizing) begin
                if (fin_dim < EMBED_DIM) begin
                    begin : div_blk
                        reg signed [2*DATA_WIDTH-1:0] num;
                        reg signed [DATA_WIDTH-1:0] quotient;
                        num = running_acc[fin_dim];
                        if (running_denom != 0)
                            // acc is in Q8.16 (from Q0.8 × Q8.8), denom is in Q0.8
                            // result = acc / denom → Q8.8
                            quotient = num / $signed({1'b0, running_denom[DATA_WIDTH-1:0]});
                        else
                            quotient = 0;
                        result_out[fin_dim*DATA_WIDTH +: DATA_WIDTH] <= quotient;
                    end
                    fin_dim <= fin_dim + 1;
                end else begin
                    result_valid <= 1'b1;
                    finalizing   <= 1'b0;
                end
            end

        end
    end

endmodule
