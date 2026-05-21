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
    input  wire                          ready_in,   // Downstream ready signal for pipeline separation
    input  wire signed [DATA_WIDTH-1:0]  a,
    input  wire signed [DATA_WIDTH-1:0]  b,
    output reg  signed [ACC_WIDTH-1:0]   acc_out,
    output reg                           valid_out,
    output reg                           ready_out   // Upstream ready signal for pipeline separation
);

    wire signed [2*DATA_WIDTH-1:0] product;
    wire is_zero;
    wire signed [ACC_WIDTH-1:0] sum_result;
    wire overflow_pos, overflow_neg;
    
    // Clock enable for power optimization - disable accumulator update when zero detected
    wire acc_en;
    reg acc_en_reg; // Registered version for clock gating

    assign is_zero = (a_d == {DATA_WIDTH{1'b0}}) || (b_d == {DATA_WIDTH{1'b0}});
    assign product = a_d * b_d;
    assign acc_en = valid_in_d && !is_zero && ready_in_d; // Enable accumulation only when valid input, not zero, and downstream ready
    
    // Register the clock enable for proper clock gating
    always @(posedge clk) begin
        if (rst || clear_acc) begin
            acc_en_reg <= 1'b0;
        end else begin
            acc_en_reg <= acc_en;
        end
    end
    
    // Sign-extended product for accumulation
    wire signed [ACC_WIDTH-1:0] product_ext = {{(ACC_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
    assign sum_result = acc_out + product_ext;
    
    // Overflow detection: if signs of operands match but result sign differs
    assign overflow_pos = !acc_out[ACC_WIDTH-1] && !product_ext[ACC_WIDTH-1] && sum_result[ACC_WIDTH-1];
    assign overflow_neg = acc_out[ACC_WIDTH-1] && product_ext[ACC_WIDTH-1] && !sum_result[ACC_WIDTH-1];

    // Combinational ready_out: we are ready to accept new input if we don't have valid output holding or if downstream is ready to take our current output
    assign ready_out = ready_in || !valid_out;

    // Pipeline registers for proper handshake
    reg valid_in_d;
    reg ready_in_d;
    reg signed [DATA_WIDTH-1:0] a_d;
    reg signed [DATA_WIDTH-1:0] b_d;

    always @(posedge clk) begin
        // Pipeline registers for input data (one-shot per valid_in pulse)
        if (valid_in && ready_in) begin
            a_d <= a;
            b_d <= b;
            valid_in_d <= 1'b1;
            ready_in_d <= ready_in;
        end else begin
            valid_in_d <= 1'b0;
            ready_in_d <= 1'b0;
        end

        if (rst || clear_acc) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            // Clock gating/power optimization: only update accumulator when enabled
            if (acc_en_reg) begin
                // Saturating accumulation — clamp on overflow
                if (overflow_pos)
                    acc_out <= {1'b0, {(ACC_WIDTH-1){1'b1}}};  // Max positive
                else if (overflow_neg)
                    acc_out <= {1'b1, {(ACC_WIDTH-1){1'b0}}};  // Max negative
                else
                    acc_out <= sum_result;
            end
            // Update valid_out based on handshake
            valid_out <= valid_in_d && ready_in_d;
        end
    end

endmodule
