// ============================================================================
// Module: gelu_activation
// Description: GELU activation using 256-entry LUT for high accuracy.
//   Replaces crude 3-piece linear approximation (Issue #13).
//   All values in Q8.8 format (8 integer bits, 8 fractional bits, signed).
//   Single-cycle pipelined output using combinational LUT.
// ============================================================================
module gelu_activation #(
    parameter WIDTH = 16
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               valid_in,
    input  wire signed [WIDTH-1:0]  x_in,
    output reg  signed [WIDTH-1:0]  y_out,
    output reg                      valid_out
);

    // Issue #13: Use 256-entry LUT for precise GELU
    wire signed [WIDTH-1:0] gelu_lut_result;

    gelu_lut_256 u_gelu_lut (
        .x_in(x_in),
        .gelu_out(gelu_lut_result)
    );

    always @(posedge clk) begin
        if (rst) begin
            y_out     <= {WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            y_out <= gelu_lut_result;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
