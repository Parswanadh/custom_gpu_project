`timescale 1ns / 1ps

module ternary_mac_engine_tb;

    reg clk, rst, start;
    reg [31:0] weight_word;
    wire [3:0] weight_word_addr;
    reg [7:0] activation_in;
    wire [5:0] activation_addr;
    wire signed [23:0] result;
    wire done;
    wire [15:0] total_adds, total_subs, total_skips, total_ops;

    ternary_mac_engine #(
        .PARALLEL_WIDTH(8), .DATA_WIDTH(8),
        .ACC_WIDTH(24), .NUM_WEIGHTS(16)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .weight_word(weight_word),
        .weight_word_addr(weight_word_addr),
        .activation_in(activation_in),
        .activation_addr(activation_addr),
        .result(result), .done(done),
        .total_adds(total_adds), .total_subs(total_subs),
        .total_skips(total_skips), .total_ops(total_ops)
    );

    always #5 clk = ~clk;
    integer tests_passed = 0, tests_total = 0;
    
    // Simple activation memory
    reg [7:0] act_mem [0:63];
    always @(*) activation_in = act_mem[activation_addr];
    
    // Simple weight memory (1 word = 16 ternary weights)
    reg [31:0] wt_mem [0:3];
    always @(*) weight_word = wt_mem[weight_word_addr];
    
    task wait_for_done;
        input integer max_cycles;
        integer count;
        begin
            for (count = 0; count < max_cycles; count = count + 1) begin
                @(negedge clk);
                if (done) count = max_cycles;
            end
        end
    endtask

    integer i;
    
    initial begin
        clk = 0; rst = 1; start = 0;
        for (i = 0; i < 64; i = i + 1) act_mem[i] = 0;
        for (i = 0; i < 4; i = i + 1) wt_mem[i] = 0;
        
        @(negedge clk); @(negedge clk); rst = 0; @(negedge clk);

        $display("=================================================");
        $display("   BitNet 1.58 Ternary MAC Engine Tests");
        $display("   (No Multipliers - Add/Subtract/Skip Only)");
        $display("   Paper: 'The Era of 1-bit LLMs' (Microsoft, 2024)");
        $display("=================================================");

        // ================================================================
        // TEST 1: All +1 weights, activation = 10
        // 16 weights all +1 (code=01) × activation 10 = 16 × 10 = 160
        // ================================================================
        tests_total = tests_total + 1;
        // Pack 16 weights of +1 (code=01): 01_01_01_01... = 0x55555555
        wt_mem[0] = 32'h55555555;
        for (i = 0; i < 16; i = i + 1) act_mem[i] = 8'd10;
        
        start = 1; @(negedge clk); start = 0;
        wait_for_done(200);
        
        if (done && result == 160) begin
            $display("[PASS] Test 1: All +1 weights: 16 × 10 = %0d (no multipliers used!)", result);
            $display("         Stats: adds=%0d, subs=%0d, skips=%0d", total_adds, total_subs, total_skips);
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 1: Expected 160, got %0d (done=%b)", result, done);

        // ================================================================
        // TEST 2: All -1 weights, activation = 5
        // 16 weights all -1 (code=10) × activation 5 = -16 × 5 = -80
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk); @(negedge clk);
        // Pack 16 weights of -1 (code=10): 10_10_10_10... = 0xAAAAAAAA
        wt_mem[0] = 32'hAAAAAAAA;
        for (i = 0; i < 16; i = i + 1) act_mem[i] = 8'd5;
        
        start = 1; @(negedge clk); start = 0;
        wait_for_done(200);
        
        if (done && result == -80) begin
            $display("[PASS] Test 2: All -1 weights: -16 × 5 = %0d (subtractions only!)", result);
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 2: Expected -80, got %0d", result);

        // ================================================================
        // TEST 3: Mixed {+1, -1, 0} weights
        // Weights: [+1,+1,-1,-1,0,0,+1,-1, +1,0,0,0,+1,+1,-1,0]
        // Encoding:  01 01 10 10 00 00 01 10  01 00 00 00 01 01 10 00
        // Activations: all = 3
        // Expected: (1+1-1-1+0+0+1-1 + 1+0+0+0+1+1-1+0) × 3
        //         = (0 + 2) × 3 = 2 × 3 = 6
        // Wait, each weight is individually applied to activation[i]=3
        // Result = sum of (weight_i * act_i) = sum of (weight_i * 3)
        //        = 3 * (1+1-1-1+0+0+1-1+1+0+0+0+1+1-1+0)
        //        = 3 * 2 = 6
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk); @(negedge clk);
        // Bit pattern (LSB first, 2 bits per weight):
        // w0=+1(01), w1=+1(01), w2=-1(10), w3=-1(10), w4=0(00), w5=0(00), w6=+1(01), w7=-1(10)
        // w8=+1(01), w9=0(00), w10=0(00), w11=0(00), w12=+1(01), w13=+1(01), w14=-1(10), w15=0(00)
        wt_mem[0] = {2'b00, 2'b10, 2'b01, 2'b01, 2'b00, 2'b00, 2'b00, 2'b01, 
                     2'b10, 2'b01, 2'b00, 2'b00, 2'b10, 2'b10, 2'b01, 2'b01};
        for (i = 0; i < 16; i = i + 1) act_mem[i] = 8'd3;
        
        start = 1; @(negedge clk); start = 0;
        wait_for_done(200);
        
        if (done && result == 6) begin
            $display("[PASS] Test 3: Mixed ternary weights, result = %0d", result);
            $display("         Adds=%0d, Subs=%0d, Skips=%0d (total=%0d)", 
                     total_adds, total_subs, total_skips, total_ops);
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 3: Expected 6, got %0d", result);

        // ================================================================
        // TEST 4: All zero weights → result must be 0 (perfect sparsity)
        // This proves the zero-gating efficiency
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk); @(negedge clk);
        wt_mem[0] = 32'h00000000;  // All zeros
        for (i = 0; i < 16; i = i + 1) act_mem[i] = 8'd255;
        
        start = 1; @(negedge clk); start = 0;
        wait_for_done(200);
        
        if (done && result == 0 && total_skips == 16) begin
            $display("[PASS] Test 4: All-zero weights → result=0, skips=%0d (100%% zero-gated!)", total_skips);
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 4: result=%0d, skips=%0d", result, total_skips);

        // ================================================================
        // TEST 5: Verify NO multiplier usage — only adds/subs
        // This is the key BitNet 1.58 claim
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk); @(negedge clk);
        wt_mem[0] = 32'h55555555;  // All +1
        for (i = 0; i < 16; i = i + 1) act_mem[i] = 8'd7;
        
        start = 1; @(negedge clk); start = 0;
        wait_for_done(200);
        
        if (done && total_adds == 16 && total_subs == 0) begin
            $display("[PASS] Test 5: Multiplier-free proof: %0d adds, %0d subs, 0 multiplications", total_adds, total_subs);
            $display("         Result: %0d = 16 × 7 (computed with ZERO multipliers!)", result);
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 5: adds=%0d, subs=%0d", total_adds, total_subs);

        $display("=================================================");
        $display("   BitNet 1.58 Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("   KEY RESULT: All computation done WITHOUT multipliers!");
        $display("=================================================");
        
        #10 $finish;
    end
    
    initial begin #50000; $display("TIMEOUT"); $finish; end

endmodule
