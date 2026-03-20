`timescale 1ns / 1ps

// ============================================================================
// Module: imprinted_mini_transformer_core
// Description: Optional hardwired mini-model core for imprint mode.
//
// - Implements a fixed-latency affine/mixing transform over 8 lanes.
// - Emulates model-specific, silicon-wired datapath behavior.
// - Additive path: enabled only when top-level imprint profile requests it.
// ============================================================================
module imprinted_mini_transformer_core #(
    parameter DIM      = 8,
    parameter DATA_W   = 16,
    parameter LATENCY  = 8
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     start,
    input  wire [DIM*DATA_W-1:0]    token_embedding,
    input  wire [5:0]               position,
    output reg                      done,
    output reg  [DIM*DATA_W-1:0]    output_vector,
    output reg  [15:0]              cycles_used
);
    integer i;
    integer next_i;
    reg [7:0] coeff;
    reg signed [15:0] bias;
    reg signed [31:0] lane_cur;
    reg signed [31:0] lane_next;
    reg signed [31:0] mix;
    reg signed [31:0] sat_mix;
    reg [DIM*DATA_W-1:0] calc_out;

    reg running;
    reg [15:0] countdown;

    always @* begin
        calc_out = {DIM*DATA_W{1'b0}};
        for (i = 0; i < DIM; i = i + 1) begin
            case (i)
                0: begin coeff = 8'd3; bias = 16'sd11; end
                1: begin coeff = 8'd5; bias = -16'sd7; end
                2: begin coeff = 8'd9; bias = 16'sd13; end
                3: begin coeff = 8'd7; bias = -16'sd5; end
                4: begin coeff = 8'd4; bias = 16'sd3; end
                5: begin coeff = 8'd6; bias = -16'sd9; end
                6: begin coeff = 8'd8; bias = 16'sd15; end
                default: begin coeff = 8'd10; bias = -16'sd1; end
            endcase

            next_i = (i == DIM-1) ? 0 : (i + 1);
            lane_cur  = $signed(token_embedding[i*DATA_W +: DATA_W]);
            lane_next = $signed(token_embedding[next_i*DATA_W +: DATA_W]);

            // Fixed-function hardwired transform:
            // lane + position*coeff + neighbor/4 + bias
            mix = lane_cur +
                  ($signed({26'd0, position}) * $signed({24'd0, coeff})) +
                  (lane_next >>> 2) +
                  $signed(bias);

            if (mix > 32'sd32767)
                sat_mix = 32'sd32767;
            else if (mix < -32'sd32768)
                sat_mix = -32'sd32768;
            else
                sat_mix = mix;

            calc_out[i*DATA_W +: DATA_W] = sat_mix[DATA_W-1:0];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            done         <= 1'b0;
            output_vector <= {DIM*DATA_W{1'b0}};
            cycles_used  <= 16'd0;
            running      <= 1'b0;
            countdown    <= 16'd0;
        end else begin
            done <= 1'b0;

            if (start && !running) begin
                running       <= 1'b1;
                countdown     <= LATENCY - 1;
                output_vector <= calc_out;
                cycles_used   <= LATENCY;
            end else if (running) begin
                if (countdown == 0) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end else begin
                    countdown <= countdown - 1'b1;
                end
            end
        end
    end

endmodule

