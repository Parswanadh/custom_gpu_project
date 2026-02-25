// ============================================================================
// Module: variable_precision_alu
// Description: Variable-precision multiply unit.
//   Mode 00 (4-bit):  4 parallel 4x4-bit multiplications
//   Mode 01 (8-bit):  2 parallel 8x8-bit multiplications
//   Mode 10 (16-bit): 1 full 16x16-bit multiplication
// ============================================================================
module variable_precision_alu (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [15:0] a,          // 16-bit input operand A
    input  wire [15:0] b,          // 16-bit input operand B
    input  wire [1:0]  mode,       // Precision mode: 00=4bit, 01=8bit, 10=16bit
    output reg  [63:0] result,     // 64-bit result (packed sub-results)
    output reg         valid_out
);

    // Intermediate products computed combinationally, registered to output
    reg [7:0]  prod4_0, prod4_1, prod4_2, prod4_3;
    reg [15:0] prod8_0, prod8_1;
    reg [31:0] prod16;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            result    <= 64'd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            case (mode)
                2'b00: begin
                    // Mode 0: Four parallel 4x4 multiplications
                    // Each 4x4 → 8-bit result, packed into 16-bit slots
                    prod4_0 = a[3:0]   * b[3:0];
                    prod4_1 = a[7:4]   * b[7:4];
                    prod4_2 = a[11:8]  * b[11:8];
                    prod4_3 = a[15:12] * b[15:12];
                    result <= {8'd0, prod4_3, 8'd0, prod4_2, 8'd0, prod4_1, 8'd0, prod4_0};
                end
                2'b01: begin
                    // Mode 1: Two parallel 8x8 multiplications
                    prod8_0 = a[7:0]  * b[7:0];
                    prod8_1 = a[15:8] * b[15:8];
                    result <= {16'd0, 16'd0, prod8_1, prod8_0};
                end
                2'b10: begin
                    // Mode 2: One full 16x16 multiplication
                    prod16 = a * b;
                    result <= {32'd0, prod16};
                end
                default: begin
                    result <= 64'd0;
                end
            endcase
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
