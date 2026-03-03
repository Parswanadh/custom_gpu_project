// ============================================================================
// Module: mac_unit
// Description: Multiply-Accumulate unit — the building block of all neural
//   network hardware. Supports signed Q8.8 arithmetic.
//   FIXES: Signed arithmetic (#1), zero detection uses signed comparison
// ============================================================================
module mac_unit #(
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          clear_acc,
    input  wire                          valid_in,
    input  wire signed [DATA_WIDTH-1:0]  a,
    input  wire signed [DATA_WIDTH-1:0]  b,
    output reg  signed [ACC_WIDTH-1:0]   acc_out,
    output reg                           valid_out
);

    wire signed [2*DATA_WIDTH-1:0] product;
    wire is_zero;

    assign is_zero = (a == {DATA_WIDTH{1'b0}}) || (b == {DATA_WIDTH{1'b0}});
    assign product = a * b;

    always @(posedge clk) begin
        if (rst || clear_acc) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (!is_zero) begin
                // Sign-extend product to ACC_WIDTH before adding
                acc_out <= acc_out + {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
