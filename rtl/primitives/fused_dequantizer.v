// ============================================================================
// Module: fused_dequantizer
// Description: Converts INT4 (4-bit quantized weight) to signed INT8
//   using: int8_out = (int4_in - offset) * scale
//   FIXES: Synchronous reset (Issue #6), signed output (Issue #1)
// ============================================================================
module fused_dequantizer (
    input  wire       clk,
    input  wire       rst,
    input  wire       valid_in,
    input  wire [3:0] int4_in,
    input  wire [3:0] scale,
    input  wire [3:0] offset,
    output reg  signed [7:0] int8_out,
    output reg        valid_out
);

    wire signed [4:0] shifted;
    wire signed [8:0] product;

    assign shifted = $signed({1'b0, int4_in}) - $signed({1'b0, offset});
    assign product = shifted * $signed({1'b0, scale});

    always @(posedge clk) begin  // Synchronous reset
        if (rst) begin
            int8_out  <= 8'sd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // Clamp to signed 8-bit range [-128, 127]
            if (product < -128)
                int8_out <= -8'sd128;
            else if (product > 127)
                int8_out <= 8'sd127;
            else
                int8_out <= product[7:0];
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
