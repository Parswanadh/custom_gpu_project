`timescale 1ns / 1ps

/*
 * Module: moe_router.v
 * Description: Hardware Mixture of Experts (MoE) Router
 *              Takes 4 expert scores (logits) from the ALU, finds the maximum
 *              (Top-1 routing), and outputs the selected expert ID and a 
 *              1-hot enable mask to trigger the PMU to spin up that FFN block.
 *              Combinational path with 1-cycle registered output.
 */
module moe_router #(
    parameter NUM_EXPERTS = 4,
    parameter SCORE_WIDTH = 16
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    
    // Inputs (Flat array of 4 * SCORE_WIDTH bits)
    input  wire [(NUM_EXPERTS*SCORE_WIDTH)-1:0] scores_in,
    input  wire                                 valid_in,
    
    // Outputs
    output reg  [$clog2(NUM_EXPERTS)-1:0]       expert_id_out,
    output reg  [NUM_EXPERTS-1:0]               expert_mask_out,
    output reg                                  valid_out
);

    // Combinational Max-Finding Tree (for 4 Experts)
    // Extracting individual 16-bit signed scores
    wire signed [SCORE_WIDTH-1:0] s0 = scores_in[1*SCORE_WIDTH-1 : 0*SCORE_WIDTH];
    wire signed [SCORE_WIDTH-1:0] s1 = scores_in[2*SCORE_WIDTH-1 : 1*SCORE_WIDTH];
    wire signed [SCORE_WIDTH-1:0] s2 = scores_in[3*SCORE_WIDTH-1 : 2*SCORE_WIDTH];
    wire signed [SCORE_WIDTH-1:0] s3 = scores_in[4*SCORE_WIDTH-1 : 3*SCORE_WIDTH];

    // Stage 1: Compare pairs
    wire signed [SCORE_WIDTH-1:0] max_01 = (s0 > s1) ? s0 : s1;
    wire [1:0]                    id_01  = (s0 > s1) ? 2'd0 : 2'd1;

    wire signed [SCORE_WIDTH-1:0] max_23 = (s2 > s3) ? s2 : s3;
    wire [1:0]                    id_23  = (s2 > s3) ? 2'd2 : 2'd3;

    // Stage 2: Final comparison for Top-1
    wire signed [SCORE_WIDTH-1:0] max_final_val = (max_01 > max_23) ? max_01 : max_23;
    wire [1:0]                    final_id      = (max_01 > max_23) ? id_01 : id_23;

    // 1-hot mask generation for Power Management Unit (PMU)
    wire [NUM_EXPERTS-1:0] final_mask = (1 << final_id);

    // Register Outputs (1 cycle pipeline matching the rest of the compute engine)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            expert_id_out <= 0;
            expert_mask_out <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                expert_id_out <= final_id;
                expert_mask_out <= final_mask;
            end
        end
    end

endmodule
