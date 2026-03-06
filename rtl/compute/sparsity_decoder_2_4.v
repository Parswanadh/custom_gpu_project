// ============================================================================
// Module: sparsity_decoder_2_4
// Description: 2:4 Structured Sparsity Engine.
//   Matches NVIDIA Ampere A100 feature: in every group of 4 weights,
//   exactly 2 are zero (50% guaranteed sparsity).
//
//   Input:  4 weights + 2-bit bitmap indicating which 2 of 4 are non-zero
//   Output: 2 non-zero weights + their original indices
//
//   This gives 2× computation savings with proven minimal accuracy loss.
//   The decoder reads a compact bitmap and routes only the non-zero weights
//   to the multiplier, skipping the other 2 entirely.
//
//   Bitmap encoding (2 bits select pattern):
//     00 → positions [0,1] are non-zero (skip [2,3])
//     01 → positions [0,2] are non-zero (skip [1,3])
//     10 → positions [1,2] are non-zero (skip [0,3])
//     11 → positions [0,3] are non-zero (skip [1,2])
//
// Parameters: DATA_WIDTH
// ============================================================================
module sparsity_decoder_2_4 #(
    parameter DATA_WIDTH = 16
)(
    input  wire                            clk,
    input  wire                            rst,
    input  wire                            valid_in,
    input  wire [4*DATA_WIDTH-1:0]         weights_in,     // 4 weights (2 are zero)
    input  wire [4*DATA_WIDTH-1:0]         activations_in, // 4 matching activations
    input  wire [1:0]                      sparsity_bitmap, // Which 2 of 4 are non-zero
    
    output reg  [1:0]                      nz_idx_0, nz_idx_1, // Indices of non-zero weights
    output reg  signed [DATA_WIDTH-1:0]    nz_weight_0, nz_weight_1,
    output reg  signed [DATA_WIDTH-1:0]    nz_act_0, nz_act_1,
    output reg  signed [2*DATA_WIDTH-1:0]  result,         // Sum of 2 products
    output reg                             valid_out,
    output reg  [31:0]                     skipped_count,  // Total multiplications skipped
    output reg  [31:0]                     computed_count   // Total multiplications done
);

    // Wires for individual weights and activations
    wire signed [DATA_WIDTH-1:0] w [0:3];
    wire signed [DATA_WIDTH-1:0] a [0:3];
    
    genvar gi;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : unpack
            assign w[gi] = weights_in[gi*DATA_WIDTH +: DATA_WIDTH];
            assign a[gi] = activations_in[gi*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Decode sparsity bitmap to get non-zero indices
    always @(posedge clk) begin
        if (rst) begin
            valid_out      <= 1'b0;
            result         <= 0;
            skipped_count  <= 32'd0;
            computed_count <= 32'd0;
            nz_idx_0       <= 2'd0;
            nz_idx_1       <= 2'd0;
            nz_weight_0    <= 0;
            nz_weight_1    <= 0;
            nz_act_0       <= 0;
            nz_act_1       <= 0;
        end else begin
            valid_out <= 1'b0;
            
            if (valid_in) begin
                // Decode bitmap to select 2 non-zero positions
                case (sparsity_bitmap)
                    2'b00: begin // positions [0,1] non-zero
                        nz_idx_0    <= 2'd0; nz_idx_1    <= 2'd1;
                        nz_weight_0 <= w[0]; nz_weight_1 <= w[1];
                        nz_act_0    <= a[0]; nz_act_1    <= a[1];
                        result      <= (w[0] * a[0]) + (w[1] * a[1]);
                    end
                    2'b01: begin // positions [0,2] non-zero
                        nz_idx_0    <= 2'd0; nz_idx_1    <= 2'd2;
                        nz_weight_0 <= w[0]; nz_weight_1 <= w[2];
                        nz_act_0    <= a[0]; nz_act_1    <= a[2];
                        result      <= (w[0] * a[0]) + (w[2] * a[2]);
                    end
                    2'b10: begin // positions [1,2] non-zero
                        nz_idx_0    <= 2'd1; nz_idx_1    <= 2'd2;
                        nz_weight_0 <= w[1]; nz_weight_1 <= w[2];
                        nz_act_0    <= a[1]; nz_act_1    <= a[2];
                        result      <= (w[1] * a[1]) + (w[2] * a[2]);
                    end
                    2'b11: begin // positions [0,3] non-zero
                        nz_idx_0    <= 2'd0; nz_idx_1    <= 2'd3;
                        nz_weight_0 <= w[0]; nz_weight_1 <= w[3];
                        nz_act_0    <= a[0]; nz_act_1    <= a[3];
                        result      <= (w[0] * a[0]) + (w[3] * a[3]);
                    end
                endcase
                
                valid_out      <= 1'b1;
                computed_count <= computed_count + 32'd2;  // Only 2 mults done
                skipped_count  <= skipped_count + 32'd2;   // 2 mults skipped
            end
        end
    end

endmodule
