// ============================================================================
// Module: sparsity_encoder
// Description: Encodes dense weights into 2:4 structured sparse format.
//
//   Takes 4 input weights and produces:
//     - The 2 largest by magnitude
//     - A 4-bit mask indicating which positions are nonzero
//     - The 2 pruned values are discarded (set to zero)
//
//   This is an OFFLINE (design-time/weight-loading) operation.
//   It runs once when weights are loaded, not during inference.
//
//   Example:
//     Input:  [3, -7, 1, 5]
//     Magnitudes: [3, 7, 1, 5]
//     Keep 2 largest: positions 1 (mag=7) and 3 (mag=5)
//     Mask: 4'b1010
//     Output: weight0 = -7, weight1 = 5
// ============================================================================
module sparsity_encoder #(
    parameter DATA_WIDTH = 8
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid_in,

    // Dense input: 4 weights
    input  wire signed [DATA_WIDTH-1:0] w0,
    input  wire signed [DATA_WIDTH-1:0] w1,
    input  wire signed [DATA_WIDTH-1:0] w2,
    input  wire signed [DATA_WIDTH-1:0] w3,

    // Sparse output: 2 nonzero weights + mask
    output reg  [DATA_WIDTH-1:0]   out_weight0,
    output reg  [DATA_WIDTH-1:0]   out_weight1,
    output reg  [3:0]              out_mask,
    output reg                     valid_out
);

    // Absolute values (for magnitude comparison)
    wire [DATA_WIDTH-1:0] mag0 = (w0 < 0) ? (-w0) : w0;
    wire [DATA_WIDTH-1:0] mag1 = (w1 < 0) ? (-w1) : w1;
    wire [DATA_WIDTH-1:0] mag2 = (w2 < 0) ? (-w2) : w2;
    wire [DATA_WIDTH-1:0] mag3 = (w3 < 0) ? (-w3) : w3;

    // Find the 2 smallest magnitudes (to prune)
    // Strategy: find the minimum, then the second minimum
    reg [1:0] min1_idx, min2_idx;
    reg [DATA_WIDTH-1:0] mags [0:3];
    reg [DATA_WIDTH-1:0] min1_val;
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            valid_out   <= 1'b0;
            out_weight0 <= 0;
            out_weight1 <= 0;
            out_mask    <= 4'b0000;
        end else if (valid_in) begin
            // Load magnitudes
            mags[0] = mag0; mags[1] = mag1;
            mags[2] = mag2; mags[3] = mag3;

            // Find smallest magnitude
            min1_idx = 0;
            min1_val = mags[0];
            for (i = 1; i < 4; i = i + 1) begin
                if (mags[i] < min1_val) begin
                    min1_val = mags[i];
                    min1_idx = i[1:0];
                end
            end

            // Find second smallest (excluding first)
            min2_idx = (min1_idx == 0) ? 2'd1 : 2'd0;
            for (i = 0; i < 4; i = i + 1) begin
                if (i[1:0] != min1_idx && mags[i] < mags[min2_idx]) begin
                    min2_idx = i[1:0];
                end
            end

            // Build mask: set bits for the 2 KEPT (larger) positions
            out_mask <= 4'b1111 & ~(4'b0001 << min1_idx) & ~(4'b0001 << min2_idx);

            // Output the two kept weights in order (lower index first)
            case ({min1_idx, min2_idx})
                // Pruned 0,1 → keep 2,3
                {2'd0, 2'd1}: begin out_weight0 <= w2; out_weight1 <= w3; end
                {2'd1, 2'd0}: begin out_weight0 <= w2; out_weight1 <= w3; end
                // Pruned 0,2 → keep 1,3
                {2'd0, 2'd2}: begin out_weight0 <= w1; out_weight1 <= w3; end
                {2'd2, 2'd0}: begin out_weight0 <= w1; out_weight1 <= w3; end
                // Pruned 0,3 → keep 1,2
                {2'd0, 2'd3}: begin out_weight0 <= w1; out_weight1 <= w2; end
                {2'd3, 2'd0}: begin out_weight0 <= w1; out_weight1 <= w2; end
                // Pruned 1,2 → keep 0,3
                {2'd1, 2'd2}: begin out_weight0 <= w0; out_weight1 <= w3; end
                {2'd2, 2'd1}: begin out_weight0 <= w0; out_weight1 <= w3; end
                // Pruned 1,3 → keep 0,2
                {2'd1, 2'd3}: begin out_weight0 <= w0; out_weight1 <= w2; end
                {2'd3, 2'd1}: begin out_weight0 <= w0; out_weight1 <= w2; end
                // Pruned 2,3 → keep 0,1
                {2'd2, 2'd3}: begin out_weight0 <= w0; out_weight1 <= w1; end
                {2'd3, 2'd2}: begin out_weight0 <= w0; out_weight1 <= w1; end
                default:       begin out_weight0 <= w0; out_weight1 <= w1; end
            endcase

            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
