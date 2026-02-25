// ============================================================================
// Module: mac_unit
// Description: Multiply-Accumulate unit — the building block of all neural
//   network hardware. Multiplies two inputs and accumulates the result.
//   Supports clear and integrates zero-skip from zero_detect_mult.
// Parameters: DATA_WIDTH, ACC_WIDTH
// ============================================================================
module mac_unit #(
    parameter DATA_WIDTH = 16,   // Width of input operands
    parameter ACC_WIDTH  = 32    // Width of accumulator
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    clear_acc,    // Clear accumulator
    input  wire                    valid_in,     // Input data valid
    input  wire [DATA_WIDTH-1:0]   a,            // Input operand A
    input  wire [DATA_WIDTH-1:0]   b,            // Input operand B
    output reg  [ACC_WIDTH-1:0]    acc_out,      // Accumulated result
    output reg                     valid_out     // Output valid (pulses each accumulation)
);

    wire [2*DATA_WIDTH-1:0] product;
    wire                    is_zero;

    // Zero detection: skip multiply if either operand is zero
    assign is_zero = (a == {DATA_WIDTH{1'b0}}) || (b == {DATA_WIDTH{1'b0}});
    assign product = a * b;

    always @(posedge clk) begin
        if (rst || clear_acc) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (!is_zero) begin
                acc_out <= acc_out + {{(ACC_WIDTH-2*DATA_WIDTH){1'b0}}, product};
            end
            // Even if zero-skipped, we signal valid (the accumulator stays unchanged)
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
