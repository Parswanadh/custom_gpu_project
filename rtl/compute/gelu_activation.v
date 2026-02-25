// ============================================================================
// Module: gelu_activation
// Description: Piecewise-linear GELU approximation in Q8.8 fixed-point.
//   GELU(x) ≈ x * sigmoid(1.702 * x)
//   Simplified to 3 regions for hardware:
//     x < -3.0  → 0
//     x > 3.0   → x
//     else      → linear interpolation: x * (0.5 + 0.167*x)
//   All values in Q8.8 format (8 integer bits, 8 fractional bits, signed).
// ============================================================================
module gelu_activation #(
    parameter WIDTH = 16   // Q8.8 fixed-point width
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               valid_in,
    input  wire signed [WIDTH-1:0]  x_in,     // Q8.8 input
    output reg  signed [WIDTH-1:0]  y_out,    // Q8.8 output
    output reg                      valid_out
);

    // Q8.8 constants
    localparam signed [WIDTH-1:0] NEG_THREE = -16'sd768;   // -3.0 in Q8.8
    localparam signed [WIDTH-1:0] POS_THREE =  16'sd768;   //  3.0 in Q8.8
    localparam signed [WIDTH-1:0] HALF      =  16'sd128;   //  0.5 in Q8.8
    // 0.167 ≈ 43/256 in Q8.8
    localparam signed [WIDTH-1:0] SLOPE     =  16'sd43;    //  0.167 in Q8.8

    // Intermediate computations
    reg signed [2*WIDTH-1:0] slope_x;      // SLOPE * x
    reg signed [WIDTH-1:0]   sigmoid_approx; // 0.5 + 0.167*x
    reg signed [2*WIDTH-1:0] full_product;   // x * sigmoid_approx

    always @(posedge clk) begin
        if (rst) begin
            y_out     <= {WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (x_in < NEG_THREE) begin
                // Region 1: x < -3 → output 0
                y_out <= {WIDTH{1'b0}};
            end else if (x_in > POS_THREE) begin
                // Region 3: x > 3 → output x (GELU ≈ x for large x)
                y_out <= x_in;
            end else begin
                // Region 2: -3 ≤ x ≤ 3 → linear approximation
                // sigmoid_approx = 0.5 + 0.167 * x (in Q8.8)
                slope_x = SLOPE * x_in;  // Q8.8 * Q8.8 = Q16.16, shift back to Q8.8
                sigmoid_approx = HALF + slope_x[WIDTH+7:8];  // Take middle bits

                // y = x * sigmoid_approx (Q8.8 * Q8.8 → Q16.16 → Q8.8)
                full_product = x_in * sigmoid_approx;
                y_out <= full_product[WIDTH+7:8];  // Shift to get Q8.8
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
