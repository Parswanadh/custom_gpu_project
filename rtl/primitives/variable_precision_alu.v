// ============================================================================
// Module: variable_precision_alu
// Description: Variable-precision multiply unit with signed support.
//   Mode 00 (4-bit):  4 parallel signed 4x4-bit multiplications
//   Mode 01 (8-bit):  2 parallel signed 8x8-bit multiplications
//   Mode 10 (16-bit): 1 full signed 16x16-bit multiplication
//   Mode 11 (Q4):     INT4 weight × INT8 activation for Q4 model inference
//     - a[15:0] packs 4× signed 4-bit weights (w0=a[3:0] .. w3=a[15:12])
//     - b[15:0] packs 2× signed 8-bit activations (act_lo=b[7:0], act_hi=b[15:8])
//     - Computes: w0*act_lo, w1*act_lo, w2*act_hi, w3*act_hi
//     - Each product is signed 12-bit, sign-extended to 16-bit
//     - Result: {prod3[15:0], prod2[15:0], prod1[15:0], prod0[15:0]}
//   Uses synchronous reset (Issue #6 fix).
// ============================================================================
module variable_precision_alu (
    input  wire               clk,
    input  wire               rst,
    input  wire               valid_in,
    input  wire signed [15:0] a,          // 16-bit signed input operand A
    input  wire signed [15:0] b,          // 16-bit signed input operand B
    input  wire [1:0]         mode,       // Precision mode: 00=4bit, 01=8bit, 10=16bit
    output reg  signed [63:0] result,     // 64-bit result (packed sub-results)
    output reg                valid_out
);

    // 4-bit lanes: treat each nibble as signed 4-bit
    wire signed [3:0] a_n0 = a[3:0],   b_n0 = b[3:0];
    wire signed [3:0] a_n1 = a[7:4],   b_n1 = b[7:4];
    wire signed [3:0] a_n2 = a[11:8],  b_n2 = b[11:8];
    wire signed [3:0] a_n3 = a[15:12], b_n3 = b[15:12];

    wire signed [7:0]  prod4_0 = a_n0 * b_n0;
    wire signed [7:0]  prod4_1 = a_n1 * b_n1;
    wire signed [7:0]  prod4_2 = a_n2 * b_n2;
    wire signed [7:0]  prod4_3 = a_n3 * b_n3;

    // 8-bit lanes: treat each byte as signed 8-bit
    wire signed [7:0]  a_b0 = a[7:0],   b_b0 = b[7:0];
    wire signed [7:0]  a_b1 = a[15:8],  b_b1 = b[15:8];

    wire signed [15:0] prod8_0 = a_b0 * b_b0;
    wire signed [15:0] prod8_1 = a_b1 * b_b1;

    // 16-bit full multiply
    wire signed [31:0] prod16 = a * b;

    // Q4 inference: INT4 weights × INT8 activations
    wire signed [3:0] w0 = a[3:0], w1 = a[7:4], w2 = a[11:8], w3 = a[15:12];
    wire signed [7:0] act_lo = b[7:0], act_hi = b[15:8];
    wire signed [11:0] q4_prod0 = w0 * act_lo;
    wire signed [11:0] q4_prod1 = w1 * act_lo;
    wire signed [11:0] q4_prod2 = w2 * act_hi;
    wire signed [11:0] q4_prod3 = w3 * act_hi;

    always @(posedge clk) begin
        if (rst) begin
            result    <= 64'sd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            case (mode)
                2'b00: begin
                    // Mode 0: Four parallel signed 4x4 multiplications
                    result <= {
                        {8{prod4_3[7]}}, prod4_3,
                        {8{prod4_2[7]}}, prod4_2,
                        {8{prod4_1[7]}}, prod4_1,
                        {8{prod4_0[7]}}, prod4_0
                    };
                end
                2'b01: begin
                    // Mode 1: Two parallel signed 8x8 multiplications
                    result <= {16'sd0, 16'sd0, prod8_1, prod8_0};
                end
                2'b10: begin
                    // Mode 2: One full signed 16x16 multiplication
                    result <= {{32{prod16[31]}}, prod16};
                end
                2'b11: begin
                    // Mode 3: Q4 inference — 4× INT4 weight × INT8 activation
                    result <= {
                        {4{q4_prod3[11]}}, q4_prod3,
                        {4{q4_prod2[11]}}, q4_prod2,
                        {4{q4_prod1[11]}}, q4_prod1,
                        {4{q4_prod0[11]}}, q4_prod0
                    };
                end
            endcase
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
