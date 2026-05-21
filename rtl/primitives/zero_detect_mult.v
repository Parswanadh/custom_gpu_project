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
    input  wire        ready_in,   // Downstream ready signal
    input  wire signed [7:0]  a,   // 8-bit signed input operand A
    input  wire signed [7:0]  b,   // 8-bit signed input operand B
    output reg  signed [15:0] result,     // 16-bit signed multiplication result
    output reg         skipped,    // 1 = computation was skipped (zero detected)
    output reg         valid_out,  // Output result valid
    output wire        ready_out   // Upstream ready signal
);

    // Internal registers for output data
    reg [15:0] result_reg;
    reg        skipped_reg;
    reg        valid_out_reg;
    // ready_out is combinational - no internal register needed

    // Combinational ready_out: we are ready to accept new input if we don't have valid output holding or if downstream is ready to take our current output
    assign ready_out = ready_in || !valid_out_reg;

    always @(posedge clk) begin
        if (rst) begin
            result_reg    <= 16'sd0;
            skipped_reg   <= 1'b0;
            valid_out_reg <= 1'b0;
        end else begin
            // Check if we can accept new input: we can if we don't have valid output holding or if downstream is ready to take our current output
            if (ready_in || !valid_out_reg) begin
                if (valid_in) begin
                    if (a == 8'sd0 || b == 8'sd0) begin
                        result_reg    <= 16'sd0;
                        skipped_reg   <= 1'b1;
                    end else begin
                        result_reg    <= a * b;
                        skipped_reg   <= 1'b0;
                    end
                    valid_out_reg <= 1'b1;
                end else begin
                    // No valid input, so we invalidate the output.
                    valid_out_reg <= 1'b0;
                end
            end
            // Else: we hold the output registers (result_reg, skipped_reg, valid_out_reg) because downstream is not ready and we have valid output.
        end
    end

    // Output assignments
    always @(*) begin
        result    = result_reg;
        skipped   = skipped_reg;
        valid_out = valid_out_reg;
    end

endmodule
