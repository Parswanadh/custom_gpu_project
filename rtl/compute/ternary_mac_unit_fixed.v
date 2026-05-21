// ============================================================================
// Module: ternary_mac_unit
// Description: Multiplier-free MAC unit for ternary {-1, 0, +1} weights.
//
//   BitNet b1.58 (Ma et al., arXiv:2402.17764) showed that ternary weights
//   match FP16 accuracy at same model size while eliminating all multipliers.
//
//   Instead of: product = weight × activation  (needs 8×8 multiplier)
//   We do:      product = (weight == +1) ? +activation :
//                         (weight == -1) ? -activation :
//                         (weight ==  0) ?  0
//
//   This replaces an 8×8 multiplier (~150 gates) with a 2-bit MUX + negator
//   (~20 gates). Energy per operation drops by ~10×.
//
//   Weight encoding (2 bits):
//     2'b00 = 0 (zero, skip)
//     2'b01 = +1 (pass activation through)
//     2'b10 = -1 (negate activation)
//     2'b11 = reserved
//
//   Packing: 8 ternary weights per 16-bit word (vs 2 INT8 weights)
//            → 4× weight compression
// ============================================================================
module ternary_mac_unit #(
    parameter ACT_WIDTH = 8,           // Activation bit width
    parameter NUM_WEIGHTS = 4          // Number of ternary weights per cycle
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              valid_in,
    input  wire                              ready_in,           // Downstream ready signal for pipeline separation
    input  wire                              acc_clear,          // Accumulator clear without full reset

    // Ternary weight inputs: 2 bits each
    input  wire [2*NUM_WEIGHTS-1:0]          weights_packed,  // 2 bits per weight

    // Activation input (broadcast to all weights)
    input  wire [ACT_WIDTH-1:0]              activation_in,

    // Accumulated output
    output reg  signed [2*ACT_WIDTH-1:0]     acc_out,
    output reg                               valid_out,
    output reg                               ready_out,        // Pipeline ready signal
    output reg  [7:0]                        zero_count,       // How many were zero
    output reg  [7:0]                        skipped_count     // How many MACs were skipped
);

    integer i;
    reg signed [2*ACT_WIDTH-1:0] partial_sum;
    reg [7:0] zeros;
    reg [1:0] w;
    reg signed [2*ACT_WIDTH:0] temp_sum;
    reg skipped;

    always @(posedge clk) begin
        if (rst || acc_clear) begin
            acc_out    <= {2*ACT_WIDTH{1'b0}};
            valid_out  <= 1'b0;
            ready_out  <= 1'b0;
            zero_count <= 0;
            skipped_count <= 0;
        end else if (valid_in && ready_in) begin
            partial_sum = 0;
            zeros = 0;
            skipped = 0;

            for (i = 0; i < NUM_WEIGHTS; i = i + 1) begin
                w = weights_packed[2*i +: 2];
                case (w)
                    2'b01:   partial_sum = partial_sum + $signed({{(ACT_WIDTH){1'b0}}, activation_in));  // +1 with proper sign extension
                    2'b10:   partial_sum = partial_sum - $signed({{(ACT_WIDTH){1'b0}}, activation_in});  // -1 with proper sign extension
                    default: begin 
                        partial_sum = partial_sum; 
                        zeros = zeros + 1; 
                        skipped = skipped + 1;  // Track actual zero-skipped operations
                    end
                endcase
            end

            // Proper signed accumulation with saturation
            if (skipped < NUM_WEIGHTS) begin  // Only update if we have non-skipped operations
                // Check for overflow before adding
                temp_sum = acc_out + partial_sum;
                if ((temp_sum[2*ACT_WIDTH] && !partial_sum[2*ACT_WIDTH-1]) ||  // Positive overflow
                    (!temp_sum[2*ACT_WIDTH] && partial_sum[2*ACT_WIDTH-1])) {  // Negative overflow
                    if (temp_sum[2*ACT_WIDTH])
                        acc_out <= {1'b1, {(2*ACT_WIDTH-1){1'b0}}};  // Max negative
                    else
                        acc_out <= {1'b0, {(2*ACT_WIDTH-1){1'b1}}};  // Max positive
                end else begin
                    acc_out <= temp_sum[2*ACT_WIDTH-1:0];
                end
            end
            
            valid_out  <= 1'b1;
            ready_out  <= 1'b1;
            zero_count <= zero_count + zeros;
            skipped_count <= skipped_count + skipped;
        end else begin
            valid_out <= 1'b0;
            ready_out <= 1'b0;
        end
    end
endmodule