// ============================================================================
// Module: block_dequantizer
// Description: GGML Q4_0-style block dequantization. Converts 4 packed INT4
//   weights to signed INT8 using per-block scale and zero point.
//   Formula: dequant_val = (int4_weight - zero_point) * scale
// ============================================================================
module block_dequantizer #(
    parameter BLOCK_SIZE = 32,    // weights per quantization group
    parameter LANES = 4           // parallel dequant lanes
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [15:0] packed_weights,  // 4x packed INT4 weights
    input  wire [7:0]  block_scale,     // Q4_0: per-block scale (unsigned fixed-point)
    input  wire [3:0]  block_zero,      // Q4_0: per-block zero point (typically 8)
    output reg  [4*8-1:0] dequant_out,  // 4x signed INT8 dequantized values
    output reg         valid_out
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lane
            wire signed [4:0]  shifted;
            wire signed [12:0] product;

            assign shifted = $signed({1'b0, packed_weights[i*4+:4]})
                           - $signed({1'b0, block_zero});
            assign product = shifted * $signed({1'b0, block_scale});

            always @(posedge clk) begin
                if (rst) begin
                    dequant_out[i*8+:8] <= 8'd0;
                end else if (valid_in) begin
                    if (product > 127)
                        dequant_out[i*8+:8] <= 8'sd127;
                    else if (product < -128)
                        dequant_out[i*8+:8] <= -8'sd128;
                    else
                        dequant_out[i*8+:8] <= product[7:0];
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst)
            valid_out <= 1'b0;
        else
            valid_out <= valid_in;
    end

endmodule
