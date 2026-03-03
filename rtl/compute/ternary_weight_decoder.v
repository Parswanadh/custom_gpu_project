// ============================================================================
// Module: ternary_weight_decoder
// Description: Unpacks ternary weights from compact 2-bit format.
//
//   Packing: 8 ternary weights per 16-bit word
//     bits [1:0]   = weight 0 (2'b00=0, 2'b01=+1, 2'b10=-1)
//     bits [3:2]   = weight 1
//     ...
//     bits [15:14] = weight 7
//
//   Compression: 4× over INT8 (8 weights vs 2 per 16-bit word)
// ============================================================================
module ternary_weight_decoder #(
    parameter WEIGHTS_PER_WORD = 8
)(
    input  wire                                          clk,
    input  wire                                          rst,
    input  wire                                          valid_in,
    input  wire [2*WEIGHTS_PER_WORD-1:0]                 packed_word,

    // Output: flat packed signed 8-bit weights
    output reg  [8*WEIGHTS_PER_WORD-1:0]                 weights_flat,
    output reg                                           valid_out
);

    integer i;
    reg [1:0] w;

    always @(posedge clk) begin
        if (rst) begin
            valid_out    <= 1'b0;
            weights_flat <= 0;
        end else if (valid_in) begin
            for (i = 0; i < WEIGHTS_PER_WORD; i = i + 1) begin
                w = packed_word[2*i +: 2];
                case (w)
                    2'b00:   weights_flat[8*i +: 8] <= 8'h00;  // Zero
                    2'b01:   weights_flat[8*i +: 8] <= 8'h01;  // +1
                    2'b10:   weights_flat[8*i +: 8] <= 8'hFF;  // -1 (signed)
                    default: weights_flat[8*i +: 8] <= 8'h00;  // Reserved
                endcase
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
