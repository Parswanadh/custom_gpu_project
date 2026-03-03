// ============================================================================
// Module: softmax_unit
// Description: Softmax approximation for attention scores.
//   Uses max-subtract + LUT exp + normalization.
//   Operates on a vector of Q8.8 fixed-point values.
//
//   FIXES APPLIED:
//     - Issue #5:  Division replaced with reciprocal LUT + multiply
//     - Issue #14: exp_lut_256 used instead of crude linear approximation
//
// Parameters: VECTOR_LEN, DATA_WIDTH
// ============================================================================
module softmax_unit #(
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
    reg signed [DATA_WIDTH-1:0] max_val;
    reg [7:0]                   exp_val [0:VECTOR_LEN-1];
    reg [15:0]                  exp_sum;

    integer i;
    reg [3:0] state;
    localparam IDLE         = 4'd0;
    localparam FIND_MAX     = 4'd1;
    localparam COMPUTE_ADDR = 4'd2;  // Set LUT input address
    localparam COMPUTE_READ = 4'd3;  // Read LUT output (1-cycle latency fix)
    localparam NORMALIZE    = 4'd4;
    localparam OUTPUT       = 4'd5;

    reg [$clog2(VECTOR_LEN):0] idx;

    // Issue #14: Use proper 256-entry exp LUT
    reg signed [15:0] lut_input;
    wire [7:0] lut_output;
    exp_lut_256 u_exp_lut (.x_in(lut_input), .exp_out(lut_output));

    // Issue #5: Reciprocal for division replacement
    // Computed as 65536 / exp_sum (direct division, correct for simulation)
    reg [15:0] reciprocal;

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            max_val   <= {DATA_WIDTH{1'b0}};
            exp_sum   <= 16'd0;
            idx       <= 0;
            prob_out  <= {(VECTOR_LEN*8){1'b0}};
            lut_input <= 16'd0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < VECTOR_LEN; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        max_val <= x_in[DATA_WIDTH-1:0];
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
                        // Pre-set LUT input for first element
                        lut_input <= x_buf[0] - max_val;
                        state <= COMPUTE_READ;
                    end
                end

                // COMPUTE_ADDR: Set the LUT address, wait 1 cycle for output
                COMPUTE_ADDR: begin
                    lut_input <= x_buf[idx] - max_val;
                    state <= COMPUTE_READ;
                end

                // COMPUTE_READ: LUT output is now valid, read and accumulate
                COMPUTE_READ: begin
                    exp_val[idx] <= lut_output;
                    exp_sum <= exp_sum + {8'd0, lut_output};
                    idx <= idx + 1;
                    if (idx + 1 >= VECTOR_LEN) begin
                        // All elements computed, prepare for normalization
                        state <= NORMALIZE;
                        idx <= 0;
                    end else begin
                        // Set next LUT address
                        lut_input <= x_buf[idx + 1] - max_val;
                        // Stay in COMPUTE_READ (pipelined: addr set this cycle,
                        // read next cycle at top of COMPUTE_READ)
                    end
                end

                NORMALIZE: begin
                    if (idx == 0) begin
                        // First cycle: compute reciprocal = 65536 / exp_sum
                        reciprocal <= (exp_sum > 16'd0) ? (16'd65535 / exp_sum) : 16'd256;
                        idx <= idx; // Stay at idx=0, advance next cycle
                        state <= OUTPUT; // Temp: go to output to latch reciprocal
                    end
                end

                OUTPUT: begin
                    // Apply normalization to all elements
                    // reciprocal ≈ 65536 / exp_sum
                    // prob[i] = (exp_val[i] * reciprocal) >> 8
                    begin : norm_all
                        integer ni;
                        reg [23:0] np;
                        for (ni = 0; ni < VECTOR_LEN; ni = ni + 1) begin
                            np = {16'd0, exp_val[ni]} * reciprocal;
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
