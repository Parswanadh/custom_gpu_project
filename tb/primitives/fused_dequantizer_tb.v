// ============================================================================
// Testbench: fused_dequantizer_tb
// Tests various INT4 values with different scale/offset combinations
// ============================================================================
`timescale 1ns / 1ps

module fused_dequantizer_tb;

    reg        clk, rst, valid_in;
    reg  [3:0] int4_in, scale, offset;
    wire [7:0] int8_out;
    wire       valid_out;

    fused_dequantizer uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .int4_in(int4_in), .scale(scale), .offset(offset),
        .int8_out(int8_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task test_dequant;
        input [3:0] in_val, in_scale, in_offset;
        input [7:0] expected;
        input [80*8-1:0] test_name;
        begin
            @(posedge clk);
            int4_in  = in_val;
            scale    = in_scale;
            offset   = in_offset;
            valid_in = 1'b1;
            @(posedge clk);
            valid_in = 1'b0;
            @(posedge clk);
            #1;
            if (int8_out === expected) begin
                $display("[PASS] %0s | in=%0d scale=%0d offset=%0d => out=%0d",
                         test_name, in_val, in_scale, in_offset, int8_out);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s | in=%0d scale=%0d offset=%0d => out=%0d (expected %0d)",
                         test_name, in_val, in_scale, in_offset, int8_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/fused_dequantizer.vcd");
        $dumpvars(0, fused_dequantizer_tb);
    end

    initial begin
        $display("============================================");
        $display("  Fused Dequantizer Testbench");
        $display("============================================");

        rst = 1; valid_in = 0; int4_in = 0; scale = 0; offset = 0;
        #20; rst = 0; #10;

        // Test 1: (6 - 0) * 5 = 30
        test_dequant(4'd6, 4'd5, 4'd0, 8'd30, "Basic: (6-0)*5=30");

        // Test 2: (10 - 2) * 3 = 24
        test_dequant(4'd10, 4'd3, 4'd2, 8'd24, "With offset: (10-2)*3=24");

        // Test 3: (0 - 0) * 10 = 0
        test_dequant(4'd0, 4'd10, 4'd0, 8'd0, "Zero input: (0-0)*10=0");

        // Test 4: (15 - 0) * 15 = 225
        test_dequant(4'd15, 4'd15, 4'd0, 8'd225, "Max no offset: (15-0)*15=225");

        // Test 5: (5 - 5) * 10 = 0
        test_dequant(4'd5, 4'd10, 4'd5, 8'd0, "Equal to offset: (5-5)*10=0");

        // Test 6: (3 - 8) * 2 = -10 → clamped to 0
        test_dequant(4'd3, 4'd2, 4'd8, 8'd0, "Negative clamp: (3-8)*2 => 0");

        // Test 7: (1 - 0) * 1 = 1
        test_dequant(4'd1, 4'd1, 4'd0, 8'd1, "Minimum: (1-0)*1=1");

        // Test 8: (8 - 0) * 1 = 8
        test_dequant(4'd8, 4'd1, 4'd0, 8'd8, "Scale 1: (8-0)*1=8");

        #20;

        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
