`timescale 1ns / 1ps

// ============================================================================
// Module: online_softmax
// Description: Single-pass streaming softmax using the Online Normalizer
//   Calculation algorithm (Milakov & Gimelshein, 2018).
//
//   Traditional softmax requires 2 passes:
//     Pass 1: Find max(x_i)
//     Pass 2: Compute exp(x_i - max) / sum(exp)
//
//   Online softmax does it in 1 pass by maintaining:
//     - running_max: largest value seen so far
//     - running_sum: sum of exp() values, corrected as max updates
//
//   When a new x_i arrives:
//     new_max = max(running_max, x_i)
//     correction = exp(old_max - new_max)   // fix old sum for new max
//     running_sum = running_sum * correction + exp(x_i - new_max)
//
//   This eliminates the FIND_MAX pass entirely, halving softmax latency.
//   Uses exp_lut_256 for hardware exp() approximation.
//   Uses recip_lut_256 for division-free normalization (synthesizable).
//
// Parameters: VECTOR_LEN, DATA_WIDTH
// ============================================================================
module online_softmax #(
    parameter VECTOR_LEN = 8,
    parameter DATA_WIDTH = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [VECTOR_LEN*DATA_WIDTH-1:0]    x_in,
    output reg  [VECTOR_LEN*8-1:0]             prob_out,
    output reg                                 valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf [0:VECTOR_LEN-1];
    reg signed [DATA_WIDTH-1:0] running_max;
    reg [15:0]                  running_sum;  // Accumulated exp sum
    reg [7:0]                   exp_val [0:VECTOR_LEN-1];

    integer i;
    reg [3:0] state;
    localparam IDLE         = 4'd0;
    localparam STREAM       = 4'd1;   // Single-pass: update max + sum
    localparam STREAM_READ  = 4'd2;   // Read exp LUT for current element
    localparam CORRECTION   = 4'd3;   // Apply correction factor
    localparam CORR_READ    = 4'd4;   // Read correction exp from LUT
    localparam FINAL_EXP    = 4'd5;   // Compute final exp(x_i - final_max)
    localparam FINAL_READ   = 4'd6;   // Read final exp from LUT
    localparam RECIP_SETUP  = 4'd7;   // Set up recip LUT input (1 cycle)
    localparam RECIP_READ   = 4'd8;   // Read recip LUT output (1 cycle)
    localparam NORMALIZE    = 4'd9;   // Final normalization

    reg [$clog2(VECTOR_LEN):0] idx;

    // Exp LUT instance
    reg signed [15:0] lut_input;
    wire [7:0] lut_output;
    exp_lut_256 u_exp_lut (.x_in(lut_input), .exp_out(lut_output));

    // Correction factor storage
    reg [7:0] correction_factor;

    // Reciprocal LUT for division-free normalization
    // Provides 65536 / x for 8-bit x (synthesizable combinational ROM)
    reg [7:0] recip_lut_input;
    wire [15:0] recip_lut_val;
    recip_lut_256 u_recip_lut (.x_in(recip_lut_input), .recip_out(recip_lut_val));
    reg [15:0] reciprocal;
    reg [3:0] recip_shift;  // How many extra bits to shift the result

    always @(posedge clk) begin
        if (rst) begin
            state       <= IDLE;
            valid_out   <= 1'b0;
            running_max <= -16'sd32767;
            running_sum <= 16'd0;
            idx         <= 0;
            prob_out    <= {(VECTOR_LEN*8){1'b0}};
            lut_input   <= 16'd0;
            recip_lut_input <= 8'd1;
            reciprocal  <= 16'd0;
            recip_shift <= 4'd0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Buffer all inputs
                        for (i = 0; i < VECTOR_LEN; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        running_max <= -16'sd32767;
                        running_sum <= 16'd0;
                        idx <= 0;
                        state <= STREAM;
                    end
                end

                // ============================================================
                // SINGLE-PASS STREAMING: Process one element per iteration
                // ============================================================
                STREAM: begin
                    if (idx < VECTOR_LEN) begin
                        if ($signed(x_buf[idx]) > $signed(running_max)) begin
                            // New max found — need correction
                            // Set LUT to compute exp(old_max - new_max)
                            lut_input <= running_max - x_buf[idx];
                            running_max <= x_buf[idx];
                            state <= CORR_READ;
                        end else begin
                            // No new max — just compute exp(x_i - running_max)
                            lut_input <= x_buf[idx] - running_max;
                            state <= STREAM_READ;
                        end
                    end else begin
                        // All elements processed in single pass!
                        // Now compute final probabilities
                        idx <= 0;
                        state <= FINAL_EXP;
                    end
                end

                // Read exp(x_i - running_max) — no correction needed
                STREAM_READ: begin
                    running_sum <= running_sum + {8'd0, lut_output};
                    idx <= idx + 1;
                    state <= STREAM;
                end

                // Read exp(old_max - new_max) for correction
                CORR_READ: begin
                    correction_factor <= lut_output;
                    state <= CORRECTION;
                end

                // Apply correction: running_sum = running_sum * correction / 256
                // Then compute exp(x_i - new_max) = exp(0) = 255
                CORRECTION: begin
                    // x_i IS the new max, so exp(x_i - max) = exp(0) = 255
                    running_sum <= ((running_sum * {8'd0, correction_factor}) >> 8) + 16'd255;
                    idx <= idx + 1;
                    state <= STREAM;
                end

                // ============================================================
                // FINAL PASS: Compute exp(x_i - final_max) for each element
                // ============================================================
                FINAL_EXP: begin
                    if (idx < VECTOR_LEN) begin
                        lut_input <= x_buf[idx] - running_max;
                        state <= FINAL_READ;
                    end else begin
                        // Done with final exp values — now set up reciprocal LUT
                        // We want: reciprocal ≈ 65536 / running_sum
                        // recip_lut gives 65536/x for 8-bit x
                        // If sum fits in 8 bits: reciprocal = recip_lut_val (direct)
                        // If sum > 255: shift sum right by S bits, get LUT output,
                        //               then shift LUT output right by S bits
                        if (running_sum == 16'd0) begin
                            recip_lut_input <= 8'd1;
                            recip_shift <= 4'd0;
                        end else if (running_sum <= 16'd255) begin
                            recip_lut_input <= running_sum[7:0];
                            recip_shift <= 4'd0;
                        end else if (running_sum <= 16'd511) begin
                            recip_lut_input <= running_sum[8:1];
                            recip_shift <= 4'd1;
                        end else if (running_sum <= 16'd1023) begin
                            recip_lut_input <= running_sum[9:2];
                            recip_shift <= 4'd2;
                        end else if (running_sum <= 16'd2047) begin
                            recip_lut_input <= running_sum[10:3];
                            recip_shift <= 4'd3;
                        end else if (running_sum <= 16'd4095) begin
                            recip_lut_input <= running_sum[11:4];
                            recip_shift <= 4'd4;
                        end else begin
                            recip_lut_input <= running_sum[15:8];
                            recip_shift <= 4'd8;
                        end
                        state <= RECIP_SETUP;
                    end
                end

                FINAL_READ: begin
                    exp_val[idx] <= lut_output;
                    idx <= idx + 1;
                    state <= FINAL_EXP;
                end

                // Wait 1 cycle for recip_lut_input to propagate through LUT
                RECIP_SETUP: begin
                    state <= RECIP_READ;
                end

                // Read the LUT output now that recip_lut_input has settled
                // reciprocal = recip_lut_val >> recip_shift ≈ 65536/running_sum
                RECIP_READ: begin
                    reciprocal <= recip_lut_val >> recip_shift;
                    idx <= 0;
                    state <= NORMALIZE;
                end

                // ============================================================
                // NORMALIZE: prob[i] = (exp_val[i] * reciprocal) >> 8
                // reciprocal ≈ 65536 / running_sum
                // so (exp * 65536/sum) >> 8 = exp * 256/sum = correct probability
                // ============================================================
                NORMALIZE: begin
                    begin : norm_all
                        integer ni;
                        reg [23:0] np;
                        for (ni = 0; ni < VECTOR_LEN; ni = ni + 1) begin
                            np = {16'd0, exp_val[ni]} * reciprocal;
                            // np ≈ exp * 65536/sum, shift >>8 to get exp*256/sum
                            prob_out[ni*8 +: 8] <= np[15:8];
                        end
                    end
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
