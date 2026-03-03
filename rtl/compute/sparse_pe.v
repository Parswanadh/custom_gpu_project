// ============================================================================
// Module: sparse_pe
// Description: Processing Element for 2:4 Structured Sparsity.
//
//   NVIDIA's Ampere architecture showed that in every group of 4 weights,
//   pruning 2 to zero (keeping the 2 largest by magnitude) recovers full
//   accuracy after fine-tuning.
//
//   This PE exploits that structure:
//     - Receives COMPRESSED weights: 2 nonzero values + 2-bit index mask
//     - The mask selects which 2 of 4 activation values to multiply
//     - Computes 4 effective MACs in the time of 2 actual multiplies
//     - Output: partial sum ready for accumulation
//
//   Weight format per group of 4:
//     [val0: 8 bits] [val1: 8 bits] [mask: 4 bits]
//     mask encoding: 4 bits, 2 of which are 1 (6 possible patterns)
//     e.g., mask=0b1010 means positions 1 and 3 are nonzero
//
//   Reference: Pool & Yu, "Accelerating Sparse Deep Neural Networks"
//              (arXiv:2104.08378, NVIDIA 2021)
// ============================================================================
module sparse_pe #(
    parameter DATA_WIDTH = 8
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid_in,

    // Compressed weight input
    input  wire [DATA_WIDTH-1:0]   weight0,      // First nonzero weight
    input  wire [DATA_WIDTH-1:0]   weight1,      // Second nonzero weight
    input  wire [3:0]              mask,          // Which 2 of 4 positions are nonzero

    // Activation input: 4 values from the current group
    input  wire [DATA_WIDTH-1:0]   act0,
    input  wire [DATA_WIDTH-1:0]   act1,
    input  wire [DATA_WIDTH-1:0]   act2,
    input  wire [DATA_WIDTH-1:0]   act3,

    // Output
    output reg  signed [2*DATA_WIDTH-1:0] product_out,  // Partial dot product
    output reg                            valid_out,
    output reg  [1:0]                     zero_skips     // How many of the 4 were skipped
);

    // Select which activations to multiply using mask
    reg [DATA_WIDTH-1:0] sel_act0, sel_act1;

    // Find the two set bits in mask and select corresponding activations
    always @(*) begin
        // Default
        sel_act0 = 8'd0;
        sel_act1 = 8'd0;

        // 6 possible 2:4 patterns
        case (mask)
            4'b0011: begin sel_act0 = act0; sel_act1 = act1; end
            4'b0101: begin sel_act0 = act0; sel_act1 = act2; end
            4'b0110: begin sel_act0 = act1; sel_act1 = act2; end
            4'b1001: begin sel_act0 = act0; sel_act1 = act3; end
            4'b1010: begin sel_act0 = act1; sel_act1 = act3; end
            4'b1100: begin sel_act0 = act2; sel_act1 = act3; end
            default: begin sel_act0 = act0; sel_act1 = act1; end // fallback
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            product_out <= 0;
            valid_out   <= 1'b0;
            zero_skips  <= 2'd0;
        end else if (valid_in) begin
            // Multiply the two nonzero weight-activation pairs
            product_out <= $signed({1'b0, weight0}) * $signed({1'b0, sel_act0})
                         + $signed({1'b0, weight1}) * $signed({1'b0, sel_act1});
            valid_out   <= 1'b1;
            zero_skips  <= 2'd2;  // Always skip exactly 2 out of 4
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
