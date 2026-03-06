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
    wire signed [ACC_WIDTH-1:0] sum_result;
    wire overflow_pos, overflow_neg;

    assign is_zero = (a == {DATA_WIDTH{1'b0}}) || (b == {DATA_WIDTH{1'b0}});
    assign product = a * b;
    
    // Sign-extended product for accumulation
    wire signed [ACC_WIDTH-1:0] product_ext = {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
    assign sum_result = acc_out + product_ext;
    
    // Overflow detection: if signs of operands match but result sign differs
    assign overflow_pos = !acc_out[ACC_WIDTH-1] && !product_ext[ACC_WIDTH-1] && sum_result[ACC_WIDTH-1];
    assign overflow_neg = acc_out[ACC_WIDTH-1] && product_ext[ACC_WIDTH-1] && !sum_result[ACC_WIDTH-1];

    always @(posedge clk) begin
        if (rst || clear_acc) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (valid_in) begin
            if (!is_zero) begin
                // Saturating accumulation — clamp on overflow
                if (overflow_pos)
                    acc_out <= {1'b0, {(ACC_WIDTH-1){1'b1}}};  // Max positive
                else if (overflow_neg)
                    acc_out <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Max negative
                else
                    acc_out <= sum_result;
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
