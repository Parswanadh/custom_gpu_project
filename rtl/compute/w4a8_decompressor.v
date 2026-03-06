`timescale 1ns / 1ps

/*
 * Module: w4a8_decompressor.v
 * Description: Hardware pipeline for Asymmetric W4A8 Weight Decompression (AWQ/GPTQ style).
 *              Takes a 32-bit packed word containing 8 4-bit weights.
 *              Executes: unpacked[i] = (packed[i] - zero_point) * scale_factor
 *              Clamps output to 8-bit signed range [-128, 127].
 *              1 cycle pipeline latency.
 */
module w4a8_decompressor #(
    parameter WEIGHTS_PER_WORD = 8,
    parameter W4_BITS = 4,
    parameter W8_BITS = 8
)(
    input  wire                               clk,
    input  wire                               rst_n,
    
    // Memory Interface (Group of INT4 Weights)
    input  wire [(WEIGHTS_PER_WORD*W4_BITS)-1:0] packed_w4_in,
    input  wire signed [W8_BITS-1:0]             scale_in,
    input  wire [W4_BITS-1:0]                    zero_point_in,
    input  wire                                  valid_in,
    
    // Core Interface (Unpacked INT8 Weights)
    output reg  [(WEIGHTS_PER_WORD*W8_BITS)-1:0] unpacked_w8_out,
    output reg                                   valid_out
);

    wire signed [5:0] diff [0:WEIGHTS_PER_WORD-1];
    wire signed [13:0] prod [0:WEIGHTS_PER_WORD-1];
    wire signed [W8_BITS-1:0] clamped [0:WEIGHTS_PER_WORD-1];
    
    genvar i;
    generate
        for (i = 0; i < WEIGHTS_PER_WORD; i = i + 1) begin : gen_decompress
            // Extract the 4-bit unsigned weight (0 to 15)
            wire [W4_BITS-1:0] w4_raw = packed_w4_in[(i*W4_BITS) +: W4_BITS];
            
            // diff = w4 - zero_point. Both are unsigned 4-bit [0-15]. 
            // The difference is signed 5-bit [-15 to +15]. We cast to signed 6-bit to be safe.
            assign diff[i] = $signed({2'b0, w4_raw}) - $signed({2'b0, zero_point_in});
            
            // prod = diff * scale. 
            // diff is 6-bit signed, scale is 8-bit signed. Result is 14-bit signed.
            assign prod[i] = diff[i] * scale_in;
            
            // Saturation logic for 8-bit signed output [-128, 127]
            assign clamped[i] = (prod[i] > 127)  ? 8'h7F :
                                (prod[i] < -128) ? 8'h80 :
                                prod[i][7:0];
        end
    endgenerate

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            unpacked_w8_out <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                for (j = 0; j < WEIGHTS_PER_WORD; j = j + 1) begin
                    unpacked_w8_out[(j*W8_BITS) +: W8_BITS] <= clamped[j];
                end
            end
        end
    end

endmodule
