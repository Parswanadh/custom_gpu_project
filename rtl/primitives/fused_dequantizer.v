// ============================================================================
// Module: fused_dequantizer
// Description: Converts INT4 (4-bit quantized weight) to INT8 (8-bit usable value)
//   using the dequantization formula: int8_out = (int4_in - offset) * scale
//   All integer arithmetic. Scale represents 0 to 15 (used as multiplier).
//   Single-cycle pipelined output.
// ============================================================================
module fused_dequantizer (
    input  wire       clk,
    input  wire       rst,
    input  wire       valid_in,
    input  wire [3:0] int4_in,   // 4-bit quantized weight (0-15)
    input  wire [3:0] scale,     // 4-bit scale factor (0-15)
    input  wire [3:0] offset,    // 4-bit zero-point offset (0-15)
    output reg  [7:0] int8_out,  // 8-bit dequantized output
    output reg        valid_out
);

    // Internal wires for computation
    wire signed [4:0] shifted;   // (int4_in - offset), signed 5-bit
    wire signed [8:0] product;   // shifted * scale, signed 9-bit

    // Combinational computation
    assign shifted = {1'b0, int4_in} - {1'b0, offset};  // 5-bit signed subtract
    assign product = shifted * $signed({1'b0, scale});   // signed multiply

    // Registered output
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            int8_out  <= 8'd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // Clamp to 8-bit unsigned range [0, 255]
            if (product < 0)
                int8_out <= 8'd0;
            else if (product > 255)
                int8_out <= 8'd255;
            else
                int8_out <= product[7:0];
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
