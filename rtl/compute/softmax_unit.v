// ============================================================================
// Module: softmax_unit
// Description: Softmax approximation for attention scores.
//   Uses max-subtract + LUT exp + normalization.
//   Operates on a vector of Q8.8 fixed-point values.
//   Steps:
//     1. Find max of input vector
//     2. Subtract max from each element (stability)
//     3. Look up exp() from LUT for each element
//     4. Sum all exp values
//     5. Divide each exp by sum (output probabilities in Q0.8 format)
// Parameters: VECTOR_LEN, DATA_WIDTH
// ============================================================================
module softmax_unit #(
    parameter VECTOR_LEN = 8,     // Length of input vector
    parameter DATA_WIDTH = 16     // Q8.8 fixed-point width
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [VECTOR_LEN*DATA_WIDTH-1:0]    x_in,       // Packed input vector
    output reg  [VECTOR_LEN*8-1:0]             prob_out,   // Output probabilities (Q0.8 per element)
    output reg                                 valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf [0:VECTOR_LEN-1];
    reg signed [DATA_WIDTH-1:0] max_val;
    reg signed [DATA_WIDTH-1:0] shifted;
    reg [7:0]                   exp_val [0:VECTOR_LEN-1];
    reg [15:0]                  exp_sum;

    // LUT for exp approximation: exp(x) for x in [-8, 0] mapped as Q8.8
    // Index i → exp(-(i/32)) scaled to [0,255]
    // We use a simple 2^(x/ln2) approximation: exp(x) ≈ max(0, 256 + x*89)
    // This is a linear approximation that's sufficient for attention scores

    integer i;
    reg [3:0] state;
    localparam IDLE     = 4'd0;
    localparam FIND_MAX = 4'd1;
    localparam COMPUTE  = 4'd2;
    localparam NORMALIZE= 4'd3;
    localparam OUTPUT   = 4'd4;

    reg [$clog2(VECTOR_LEN):0] idx;
    reg signed [DATA_WIDTH-1:0] temp;
    reg [15:0] div_result;

    // Simple exp approximation function
    // For x <= 0: exp(x) ≈ max(1, 256 + x) where x is in Q8.8 range [-8,0]
    // Scaled so exp(0) = 255, exp(-3) ≈ 13
    function [7:0] exp_approx;
        input signed [DATA_WIDTH-1:0] val;
        reg signed [DATA_WIDTH+3:0] tmp;
        begin
            if (val >= 0)
                exp_approx = 8'd255;
            else if (val < -16'sd1024) // less than -4.0
                exp_approx = 8'd1;
            else begin
                // Linear approx: 255 * (1 + x/4) for x in [-4, 0]
                tmp = 255 + (val * 64) / 256; // Scale: val is Q8.8
                if (tmp < 1) exp_approx = 8'd1;
                else if (tmp > 255) exp_approx = 8'd255;
                else exp_approx = tmp[7:0];
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            max_val   <= {DATA_WIDTH{1'b0}};
            exp_sum   <= 16'd0;
            idx       <= 0;
            prob_out  <= {(VECTOR_LEN*8){1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Unpack input vector
                        for (i = 0; i < VECTOR_LEN; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        max_val <= x_in[DATA_WIDTH-1:0]; // Initialize max with first element
                        idx <= 1;
                        state <= FIND_MAX;
                    end
                end

                FIND_MAX: begin
                    if (idx < VECTOR_LEN) begin
                        if ($signed(x_buf[idx]) > $signed(max_val))
                            max_val <= x_buf[idx];
                        idx <= idx + 1;
                    end else begin
                        idx <= 0;
                        exp_sum <= 16'd0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (idx < VECTOR_LEN) begin
                        shifted = x_buf[idx] - max_val;
                        exp_val[idx] <= exp_approx(shifted);
                        exp_sum <= exp_sum + {8'd0, exp_approx(shifted)};
                        idx <= idx + 1;
                    end else begin
                        idx <= 0;
                        state <= NORMALIZE;
                    end
                end

                NORMALIZE: begin
                    if (idx < VECTOR_LEN) begin
                        // Probability = exp_val[i] * 256 / exp_sum (Q0.8 output)
                        if (exp_sum > 0)
                            div_result = ({8'b0, exp_val[idx]} * 16'd256) / exp_sum;
                        else
                            div_result = 16'd0;
                        prob_out[idx*8 +: 8] <= div_result[7:0];
                        idx <= idx + 1;
                    end else begin
                        state <= OUTPUT;
                    end
                end

                OUTPUT: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
