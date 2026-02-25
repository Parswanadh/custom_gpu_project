// ============================================================================
// Module: zero_detect_mult
// Description: Multiplier with zero-detection bypass.
//   If either input is zero, output 0 (skip multiply).
//   Otherwise, output a * b.
//   Single clock cycle latency (registered output).
// ============================================================================
module zero_detect_mult (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,   // Input data valid
    input  wire [7:0]  a,          // 8-bit input operand A
    input  wire [7:0]  b,          // 8-bit input operand B
    output reg  [15:0] result,     // 16-bit multiplication result
    output reg         skipped,    // 1 = computation was skipped (zero detected)
    output reg         valid_out   // Output result valid
);

    always @(posedge clk) begin
        if (rst) begin
            result    <= 16'd0;
            skipped   <= 1'b0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (a == 8'd0 || b == 8'd0) begin
                result    <= 16'd0;
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
