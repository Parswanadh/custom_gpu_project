// ============================================================================
// Testbench: zero_detect_mult_tb
// Tests: Normal multiply, zero first, zero second, both zero, max values
// ============================================================================
`timescale 1ns / 1ps

module zero_detect_mult_tb;

    reg        clk;
    reg        rst;
    reg        valid_in;
    reg  [7:0] a, b;
    wire [15:0] result;
    wire       skipped;
    wire       valid_out;

    zero_detect_mult uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .a(a), .b(b),
        .result(result), .skipped(skipped), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num = 0;

    task apply_and_check;
        input [7:0] in_a, in_b;
        input [15:0] expected_result;
        input expected_skip;
        input [80*8-1:0] test_name;
        begin
            test_num = test_num + 1;
            // Setup inputs at negedge (halfway between posedges = safe setup time)
            @(negedge clk);
            a = in_a;
            b = in_b;
            valid_in = 1'b1;
            // Posedge arrives: module registers input
            @(negedge clk);
            valid_in = 1'b0;
            // Check output: after the posedge, result should be registered
            #1;
            if (valid_out === 1'b1 && result === expected_result && skipped === expected_skip) begin
                $display("[PASS] Test %0d: %0s | a=%0d b=%0d => result=%0d skipped=%0b",
                         test_num, test_name, in_a, in_b, result, skipped);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Test %0d: %0s | a=%0d b=%0d => result=%0d (exp %0d) skipped=%0b (exp %0b) valid=%0b",
                         test_num, test_name, in_a, in_b, result, expected_result, skipped, expected_skip, valid_out);
                fail_count = fail_count + 1;
            end
            // Ensure valid_out deasserts cleanly
            @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/zero_detect_mult.vcd");
        $dumpvars(0, zero_detect_mult_tb);
    end

    initial begin
        $display("============================================");
        $display("  Zero Detect Multiplier Testbench");
        $display("============================================");

        rst = 1; valid_in = 0; a = 0; b = 0;
        #25; rst = 0; #15;

        apply_and_check(8'd5, 8'd3, 16'd15, 1'b0, "Normal: 5 x 3 = 15");
        apply_and_check(8'd0, 8'd7, 16'd0, 1'b1, "Zero A: 0 x 7 = 0");
        apply_and_check(8'd9, 8'd0, 16'd0, 1'b1, "Zero B: 9 x 0 = 0");
        apply_and_check(8'd0, 8'd0, 16'd0, 1'b1, "Both zero: 0 x 0 = 0");
        apply_and_check(8'd255, 8'd255, 16'd65025, 1'b0, "Max: 255 x 255 = 65025");
        apply_and_check(8'd1, 8'd200, 16'd200, 1'b0, "Identity: 1 x 200 = 200");
        apply_and_check(8'd12, 8'd10, 16'd120, 1'b0, "Typical: 12 x 10 = 120");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
