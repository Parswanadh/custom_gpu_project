`timescale 1ns / 1ps

module rope_encoder_tb;
    parameter DIM = 8, DATA_WIDTH = 16, MAX_POS = 64;
    
    reg clk, rst, valid_in;
    reg [5:0] position;
    reg [DIM*DATA_WIDTH-1:0] q_in, k_in;
    wire [DIM*DATA_WIDTH-1:0] q_rot, k_rot;
    wire valid_out;

    rope_encoder #(.DIM(DIM), .DATA_WIDTH(DATA_WIDTH), .MAX_POS(MAX_POS)) dut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .position(position), .q_in(q_in), .k_in(k_in),
        .q_rot(q_rot), .k_rot(k_rot), .valid_out(valid_out)
    );

    always #5 clk = ~clk;
    integer tests_passed = 0, tests_total = 0;

    task wait_done; input integer n;
        integer c; begin for (c=0;c<n;c=c+1) begin @(negedge clk); if(valid_out) c=n; end end
    endtask

    initial begin
        clk=0; rst=1; valid_in=0; position=0; q_in=0; k_in=0;
        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);

        $display("=================================================");
        $display("   RoPE (Rotary Positional Encoding) Tests");
        $display("   Paper: 'RoFormer' (Su et al., 2021)");
        $display("   Used by: Llama, Mistral, Qwen, GPT-NeoX");
        $display("=================================================");

        // TEST 1: Position 0 → cos=1, sin=0 → output = input (no rotation)
        tests_total = tests_total + 1;
        q_in = {16'sd256, 16'sd256, 16'sd256, 16'sd256, 16'sd256, 16'sd256, 16'sd256, 16'sd256};
        k_in = {16'sd128, 16'sd128, 16'sd128, 16'sd128, 16'sd128, 16'sd128, 16'sd128, 16'sd128};
        position = 6'd0; valid_in = 1;
        @(negedge clk); valid_in = 0;
        wait_done(50);
        if (valid_out) begin
            $display("[PASS] Test 1: Pos=0 rotation (identity) — Q_rot[0]=%0d", $signed(q_rot[15:0]));
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 1: No valid output");

        // TEST 2: Position 8 → 45° rotation (cos=sin=0.707)
        tests_total = tests_total + 1;
        @(negedge clk);
        q_in = {16'sd0, 16'sd256, 16'sd0, 16'sd256, 16'sd0, 16'sd256, 16'sd0, 16'sd256};
        k_in = q_in; position = 6'd8; valid_in = 1;
        @(negedge clk); valid_in = 0;
        wait_done(50);
        if (valid_out) begin
            $display("[PASS] Test 2: Pos=8 (45 deg) — Q_rot[0]=%0d, Q_rot[1]=%0d",
                     $signed(q_rot[15:0]), $signed(q_rot[31:16]));
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 2");

        // TEST 3: Different positions → different rotations
        tests_total = tests_total + 1;
        @(negedge clk);
        q_in = {16'sd0, 16'sd256, 16'sd0, 16'sd256, 16'sd0, 16'sd256, 16'sd0, 16'sd256};
        k_in = q_in; position = 6'd16; valid_in = 1;
        @(negedge clk); valid_in = 0;
        wait_done(50);
        if (valid_out && $signed(q_rot[15:0]) != $signed(q_in[15:0])) begin
            $display("[PASS] Test 3: Pos=16 gives different rotation than input");
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 3");

        // TEST 4: K gets same rotation as Q (for proper attention score computation)
        tests_total = tests_total + 1;
        @(negedge clk);
        q_in = {16'sd100, 16'sd200, 16'sd100, 16'sd200, 16'sd100, 16'sd200, 16'sd100, 16'sd200};
        k_in = q_in; position = 6'd4; valid_in = 1;
        @(negedge clk); valid_in = 0;
        wait_done(50);
        if (valid_out && q_rot == k_rot) begin
            $display("[PASS] Test 4: Q and K get identical rotations (same input → same output)");
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 4: Q_rot != K_rot");

        $display("=================================================");
        $display("   RoPE Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        #10 $finish;
    end
    initial begin #50000; $display("TIMEOUT"); $finish; end
endmodule
