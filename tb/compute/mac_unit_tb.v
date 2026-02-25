// ============================================================================
// Testbench: mac_unit_tb
// Tests: dot product accumulation, zero-skip, clear
// ============================================================================
`timescale 1ns / 1ps

module mac_unit_tb;

    reg         clk, rst, clear_acc, valid_in;
    reg  [15:0] a, b;
    wire [31:0] acc_out;
    wire        valid_out;

    mac_unit #(.DATA_WIDTH(16), .ACC_WIDTH(32)) uut (
        .clk(clk), .rst(rst), .clear_acc(clear_acc), .valid_in(valid_in),
        .a(a), .b(b), .acc_out(acc_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task feed;
        input [15:0] in_a, in_b;
        begin
            @(negedge clk);
            a = in_a; b = in_b; valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;
        end
    endtask

    task check_acc;
        input [31:0] expected;
        input [80*8-1:0] test_name;
        begin
            #1;
            if (acc_out === expected) begin
                $display("[PASS] %0s | acc=%0d", test_name, acc_out);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s | acc=%0d (expected %0d)", test_name, acc_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/mac_unit.vcd");
        $dumpvars(0, mac_unit_tb);
    end

    initial begin
        $display("============================================");
        $display("  MAC Unit Testbench");
        $display("============================================");

        rst = 1; clear_acc = 0; valid_in = 0; a = 0; b = 0;
        #25; rst = 0; #15;

        // Dot product: [3,4,5] . [2,6,1] = 6+24+5 = 35
        feed(16'd3, 16'd2);  // acc = 6
        check_acc(32'd6, "Step 1: 3*2=6");

        feed(16'd4, 16'd6);  // acc = 6+24 = 30
        check_acc(32'd30, "Step 2: +4*6=30");

        feed(16'd5, 16'd1);  // acc = 30+5 = 35
        check_acc(32'd35, "Step 3: +5*1=35");

        // Zero-skip: adding 0*7 should keep acc at 35
        feed(16'd0, 16'd7);
        check_acc(32'd35, "Zero-skip: 0*7, acc stays 35");

        feed(16'd10, 16'd0);
        check_acc(32'd35, "Zero-skip: 10*0, acc stays 35");

        // Clear accumulator
        @(negedge clk); clear_acc = 1'b1;
        @(negedge clk); clear_acc = 1'b0;
        #1;
        check_acc(32'd0, "Clear: acc=0");

        // New dot product: [10,20] . [3,5] = 30+100 = 130
        feed(16'd10, 16'd3);
        check_acc(32'd30, "New dot 1: 10*3=30");

        feed(16'd20, 16'd5);
        check_acc(32'd130, "New dot 2: +20*5=130");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
