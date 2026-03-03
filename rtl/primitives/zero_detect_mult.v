// ============================================================================
// Module: zero_detect_mult
// Description: Multiplier with zero-detection bypass.
//   If either input is zero, output 0 (skip multiply).
//   Otherwise, output a * b.
//   Single clock cycle latency (registered output).
//   Supports signed Q8.8 arithmetic.
// ============================================================================
module zero_detect_mult (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,   // Input data valid
    input  wire signed [7:0]  a,   // 8-bit signed input operand A
    input  wire signed [7:0]  b,   // 8-bit signed input operand B
    output reg  signed [15:0] result,     // 16-bit signed multiplication result
    output reg         skipped,    // 1 = computation was skipped (zero detected)
    output reg         valid_out   // Output result valid
);

    always @(posedge clk) begin
        if (rst) begin
            result    <= 16'sd0;
            skipped   <= 1'b0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (a == 8'sd0 || b == 8'sd0) begin
                result    <= 16'sd0;
                skipped   <= 1'b1;
            end else begin
                result    <= a * b;
                skipped   <= 1'b0;
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
