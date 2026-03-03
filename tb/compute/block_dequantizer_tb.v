// ============================================================================
// Testbench: block_dequantizer_tb
// Tests: basic dequant, all-zero, max range clamping, scale=0 edge case
// ============================================================================
`timescale 1ns / 1ps

module block_dequantizer_tb;

    reg         clk, rst, valid_in;
    reg  [15:0] packed_weights;
    reg  [7:0]  block_scale;
    reg  [3:0]  block_zero;
    wire [31:0] dequant_out;
    wire        valid_out;

    block_dequantizer #(.BLOCK_SIZE(32), .LANES(4)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .packed_weights(packed_weights), .block_scale(block_scale),
        .block_zero(block_zero), .dequant_out(dequant_out),
        .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    wire signed [7:0] out0 = dequant_out[7:0];
    wire signed [7:0] out1 = dequant_out[15:8];
    wire signed [7:0] out2 = dequant_out[23:16];
    wire signed [7:0] out3 = dequant_out[31:24];

    integer pass_count = 0;
    integer fail_count = 0;

    task check_lane;
        input signed [7:0] actual;
        input signed [7:0] expected;
        input integer lane;
        input [80*8-1:0] test_name;
        begin
            if (actual === expected) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s lane %0d: got %0d, expected %0d",
                         test_name, lane, actual, expected);
            end
        end
    endtask

    task apply_input;
        input [15:0] pw;
        input [7:0]  sc;
        input [3:0]  zp;
        begin
            @(negedge clk);
            packed_weights = pw;
            block_scale    = sc;
            block_zero     = zp;
            valid_in       = 1'b1;
            @(posedge clk); #1; // outputs registered on this edge
            valid_in = 1'b0;    // deassert after sampling
        end
    endtask

    initial begin
        rst = 1; valid_in = 0;
        packed_weights = 0; block_scale = 0; block_zero = 0;
        repeat (3) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // ---- Test 1: Basic ----
        // weights=[3,5,7,9], scale=2, zero=8
        // shifted=[-5,-3,-1,1], products=[-10,-6,-2,2]
        apply_input({4'd9, 4'd7, 4'd5, 4'd3}, 8'd2, 4'd8);
        $display("Test 1 - Basic dequant:");
        check_lane(out0, -8'sd10, 0, "Basic");
        check_lane(out1, -8'sd6,  1, "Basic");
        check_lane(out2, -8'sd2,  2, "Basic");
        check_lane(out3,  8'sd2,  3, "Basic");
        if (valid_out)
            $display("  valid_out=1 [PASS]");
        else begin
            $display("  valid_out=0 [FAIL]");
            fail_count = fail_count + 1;
        end

        // ---- Test 2: All-zero weights, zero=0, scale=1 ----
        apply_input(16'h0000, 8'd1, 4'd0);
        $display("Test 2 - All zeros:");
        check_lane(out0, 8'sd0, 0, "AllZero");
        check_lane(out1, 8'sd0, 1, "AllZero");
        check_lane(out2, 8'sd0, 2, "AllZero");
        check_lane(out3, 8'sd0, 3, "AllZero");

        // ---- Test 3: Max range ----
        // weights=[0,15,0,15], scale=16, zero=8
        // shifted=[-8,7,-8,7], products=[-128,112,-128,112]
        apply_input({4'd15, 4'd0, 4'd15, 4'd0}, 8'd16, 4'd8);
        $display("Test 3 - Max range:");
        check_lane(out0, -8'sd128, 0, "MaxRange");
        check_lane(out1,  8'sd112, 1, "MaxRange");
        check_lane(out2, -8'sd128, 2, "MaxRange");
        check_lane(out3,  8'sd112, 3, "MaxRange");

        // ---- Test 4: Scale=0 edge case ----
        apply_input({4'd15, 4'd7, 4'd3, 4'd1}, 8'd0, 4'd8);
        $display("Test 4 - Scale=0:");
        check_lane(out0, 8'sd0, 0, "ScaleZero");
        check_lane(out1, 8'sd0, 1, "ScaleZero");
        check_lane(out2, 8'sd0, 2, "ScaleZero");
        check_lane(out3, 8'sd0, 3, "ScaleZero");

        // ---- Summary ----
        $display("========================================");
        $display("  %0d [PASS], %0d [FAIL]", pass_count, fail_count);
        if (fail_count == 0)
            $display("  ALL TESTS PASSED [PASS]");
        else
            $display("  SOME TESTS FAILED [FAIL]");
        $display("========================================");
        $finish;
    end

endmodule
