// ============================================================================
// Module: bf16_multiply
// Description: BF16 (Brain Float 16) multiply unit (Issue #22).
//   BF16 format: 1 sign + 8 exponent + 7 mantissa (same as FP32 top 16 bits)
//   Range: ±3.39×10^38, Resolution: ~0.8% relative error
//   Much better dynamic range than Q8.8 for transformer inference.
//
//   Pipeline: 2-stage (multiply mantissa + add exponents → normalize)
//   Zero-skip: If either input is ±0, skip to output 0.
//   Supports: multiply, with flush-to-zero for denormals.
//
// Format:
//   [15]    = sign
//   [14:7]  = exponent (bias 127)
//   [6:0]   = mantissa (implied leading 1)
// ============================================================================
module bf16_multiply (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [15:0] a,           // BF16 input A
    input  wire [15:0] b,           // BF16 input B
    output reg  [15:0] result,      // BF16 output
    output reg         valid_out,
    output reg         skipped      // Zero-skip flag
);

    // Unpack BF16
    wire        a_sign = a[15];
    wire [7:0]  a_exp  = a[14:7];
    wire [6:0]  a_man  = a[6:0];
    wire        b_sign = b[15];
    wire [7:0]  b_exp  = b[14:7];
    wire [6:0]  b_man  = b[6:0];

    // Zero detection (exponent = 0 means ±0 or denormal → flush to zero)
    wire a_is_zero = (a_exp == 8'd0);
    wire b_is_zero = (b_exp == 8'd0);
    wire either_zero = a_is_zero | b_is_zero;

    // Inf/NaN detection
    wire a_is_inf_nan = (a_exp == 8'hFF);
    wire b_is_inf_nan = (b_exp == 8'hFF);

    // Stage 1 registers
    reg        s1_valid;
    reg        s1_sign;
    reg [8:0]  s1_exp;        // 9-bit for overflow detection
    reg [15:0] s1_man_prod;   // (1.mantA) × (1.mantB) = up to 2.14 format
    reg        s1_zero;
    reg        s1_special;    // Inf or NaN

    // Mantissa multiply (with implied leading 1)
    wire [7:0] a_full_man = {1'b1, a_man};  // 1.7 format
    wire [7:0] b_full_man = {1'b1, b_man};  // 1.7 format
    wire [15:0] man_product = a_full_man * b_full_man;  // 2.14 format

    // Stage 1: Multiply + Add exponents
    always @(posedge clk) begin
        if (rst) begin
            s1_valid   <= 1'b0;
            s1_sign    <= 1'b0;
            s1_exp     <= 9'd0;
            s1_man_prod <= 16'd0;
            s1_zero    <= 1'b0;
            s1_special <= 1'b0;
        end else if (valid_in) begin
            s1_valid   <= 1'b1;
            s1_sign    <= a_sign ^ b_sign;
            s1_exp     <= a_exp + b_exp - 9'd127;  // Remove double bias
            s1_man_prod <= man_product;
            s1_zero    <= either_zero;
            s1_special <= a_is_inf_nan | b_is_inf_nan;
        end else begin
            s1_valid <= 1'b0;
        end
    end

    // Stage 2: Normalize + Pack
    always @(posedge clk) begin
        if (rst) begin
            result    <= 16'd0;
            valid_out <= 1'b0;
            skipped   <= 1'b0;
        end else if (s1_valid) begin
            valid_out <= 1'b1;

            if (s1_zero) begin
                // Zero-skip: output positive zero
                result  <= 16'd0;
                skipped <= 1'b1;
            end else if (s1_special) begin
                // Inf × anything = Inf, NaN propagation
                result  <= {s1_sign, 8'hFF, 7'd0};
                skipped <= 1'b0;
            end else begin
                skipped <= 1'b0;

                // Normalize: product is in 2.14 format
                // If bit 15 is set, shift right by 1 and increment exponent
                if (s1_man_prod[15]) begin
                    // Overflow: shift right, round, increment exp
                    if (s1_exp + 1 >= 9'd255) begin
                        // Overflow to infinity
                        result <= {s1_sign, 8'hFF, 7'd0};
                    end else begin
                        result <= {s1_sign, s1_exp[7:0] + 8'd1, s1_man_prod[14:8]};
                    end
                end else begin
                    // Normal: mantissa is 1.xx format
                    if (s1_exp[8] || s1_exp == 9'd0) begin
                        // Underflow: flush to zero
                        result <= {s1_sign, 15'd0};
                    end else begin
                        result <= {s1_sign, s1_exp[7:0], s1_man_prod[13:7]};
                    end
                end
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule

