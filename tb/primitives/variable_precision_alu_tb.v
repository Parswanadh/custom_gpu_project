// ============================================================================
// Testbench: variable_precision_alu_tb
// Tests all three precision modes. Setup on negedge, check on negedge.
// ============================================================================
`timescale 1ns / 1ps

module variable_precision_alu_tb;

    reg         clk, rst, valid_in;
    reg  [15:0] a, b;
    reg  [1:0]  mode;
    wire [63:0] result;
    wire        valid_out;

    variable_precision_alu uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .a(a), .b(b), .mode(mode),
        .result(result), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task apply_and_check;
        input [15:0] in_a, in_b;
        input [1:0]  in_mode;
        input [63:0] expected;
        input [80*8-1:0] test_name;
        begin
            @(negedge clk);
            a = in_a; b = in_b; mode = in_mode; valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;
            #1;
            if (valid_out && result === expected) begin
                $display("[PASS] %0s", test_name);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s | got=0x%016h expected=0x%016h valid=%b",
                         test_name, result, expected, valid_out);
                fail_count = fail_count + 1;
            end
            @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/variable_precision_alu.vcd");
        $dumpvars(0, variable_precision_alu_tb);
    end

    initial begin
        $display("============================================");
        $display("  Variable Precision ALU Testbench");
        $display("============================================");

        rst = 1; valid_in = 0; a = 0; b = 0; mode = 0;
        #25; rst = 0; #15;

        // Mode 00: 4-bit parallel
        // a=0x3251 nibbles: 3,2,5,1  b=0x4162 nibbles: 4,1,6,2
        // Products: 1*2=2, 5*6=30, 2*1=2, 3*4=12
        apply_and_check(16'h3251, 16'h4162, 2'b00,
            {8'd0, 8'd12, 8'd0, 8'd2, 8'd0, 8'd30, 8'd0, 8'd2},
            "4-bit: 0x3251 * 0x4162");

        apply_and_check(16'h1111, 16'h2222, 2'b00,
            {8'd0, 8'd2, 8'd0, 8'd2, 8'd0, 8'd2, 8'd0, 8'd2},
            "4-bit: all 1s * all 2s");

        // Mode 01: 8-bit parallel
        apply_and_check(16'h0A05, 16'h0304, 2'b01,
            {16'd0, 16'd0, 16'd30, 16'd20},
            "8-bit: {10,5} * {3,4}");

        // Mode 10: 16-bit
        apply_and_check(16'd100, 16'd200, 2'b10,
            {32'd0, 32'd20000},
            "16-bit: 100 * 200 = 20000");

        apply_and_check(16'hFFFF, 16'hFFFF, 2'b10,
            {32'd0, 32'hFFFE0001},
            "16-bit: max values");

        apply_and_check(16'h0000, 16'h0000, 2'b00,
            64'd0,
            "4-bit: all zeros");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
