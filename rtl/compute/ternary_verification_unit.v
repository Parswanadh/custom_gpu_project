`timescale 1ns / 1ps

// ============================================================================
// Module: ternary_verification_unit
// Description: Multiplier-Free Speculative Decoding Verification Head.
//
//   CONCEPT: This unit verifies "draft" tokens by performing an extremely 
//   low-power "Fast-Match" using ternary weights (+1, -1, 0).
//   It allows the hardware to evaluate multiple candidate tokens in parallel
//   during speculative decoding phases.
// ============================================================================
module ternary_verification_unit #(
    parameter DATA_WIDTH = 16
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   valid_in,
    
    // Inputs: Activation (A) and Ternary Weight (W_t)
    input  wire signed [DATA_WIDTH-1:0] activation,
    input  wire [1:0]                   ternary_weight, // 01=+1, 10=-1, 00=0
    
    // Feedback and Output
    input  wire signed [DATA_WIDTH+7:0] prev_acc,
    output reg  signed [DATA_WIDTH+7:0] next_acc,
    output reg                          valid_out
);

    always @(posedge clk) begin
        if (rst) begin
            next_acc  <= 0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            case (ternary_weight)
                2'b01:   next_acc <= prev_acc + activation;
                2'b10:   next_acc <= prev_acc - activation;
                2'b00:   next_acc <= prev_acc;
                default: next_acc <= prev_acc;
            endcase
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
