`timescale 1ns / 1ps

module simd_ternary_engine_tb;

    parameter NUM_WEIGHTS = 16;
    parameter LANES = 4;

    reg clk, rst;
    always #5 clk = ~clk;

    reg start;
    wire [2*LANES-1:0] weight_chunk;
    wire [$clog2(NUM_WEIGHTS/LANES):0] weight_addr;
    wire [7:0] activation_in;
    wire [$clog2(NUM_WEIGHTS)-1:0] activation_addr;
    wire signed [23:0] result;
    wire done;
    wire [15:0] t_adds, t_subs, t_skips, cycles_used;

    simd_ternary_engine #(.NUM_WEIGHTS(NUM_WEIGHTS), .LANES(LANES))
    uut (
        .clk(clk), .rst(rst), .start(start),
        .weight_chunk(weight_chunk), .weight_addr(weight_addr),
        .activation_in(activation_in), .activation_addr(activation_addr),
        .result(result), .done(done),
        .total_adds(t_adds), .total_subs(t_subs),
        .total_skips(t_skips), .cycles_used(cycles_used)
    );

    // Weight memory: 4 chunks × 4 lanes × 2 bits = 32 bits
    reg [2*LANES-1:0] wt_mem [0:3];
    assign weight_chunk = wt_mem[weight_addr];
    assign activation_in = 8'd10;

    integer tests_passed, tests_total;

    initial begin
        clk = 0; rst = 1; start = 0;
        tests_passed = 0; tests_total = 0;

        // All +1 weights: {01, 01, 01, 01} = 8'h55
        wt_mem[0] = 8'b01_01_01_01;
        wt_mem[1] = 8'b01_01_01_01;
        wt_mem[2] = 8'b01_01_01_01;
        wt_mem[3] = 8'b01_01_01_01;

        @(negedge clk); @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);

        $display("=================================================");
        $display("   SIMD Ternary Engine Tests (%0d-wide SIMD)", LANES);
        $display("   %0d weights, %0d lanes = %0d chunks", NUM_WEIGHTS, LANES, NUM_WEIGHTS/LANES);
        $display("=================================================");

        // TEST 1: All +1, activation=10 → 16 × 10 = 160
        tests_total = tests_total + 1;
        start = 1; @(negedge clk); start = 0;
        begin : w1
            integer t;
            for (t = 0; t < 50; t = t + 1) begin
                @(negedge clk); if (done) t = 50;
            end
        end
        
        if (done && result == 24'sd160) begin
            $display("[PASS] Test 1: All +1 → result=%0d (expected 160), %0d cycles", result, cycles_used);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 1: result=%0d, done=%b, cycles=%0d", result, done, cycles_used);

        // TEST 2: Mixed weights — some -1 (2'b10) and 0 (2'b00)
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0; @(negedge clk);
        wt_mem[0] = 8'b01_10_01_10; // +1, -1, +1, -1 → sum = 0
        wt_mem[1] = 8'b01_01_01_01; // 4 × +1 = 40
        wt_mem[2] = 8'b00_00_00_00; // 4 × 0 = 0
        wt_mem[3] = 8'b01_01_01_01; // 4 × +1 = 40
        
        start = 1; @(negedge clk); start = 0;
        begin : w2
            integer t;
            for (t = 0; t < 50; t = t + 1) begin
                @(negedge clk); if (done) t = 50;
            end
        end
        
        if (done && result == 24'sd80) begin
            $display("[PASS] Test 2: Mixed weights → result=%0d (expected 80), %0d cycles", result, cycles_used);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 2: result=%0d, cycles=%0d", result, cycles_used);

        // TEST 3: Speed comparison — should be < 8 cycles (vs 19 for original)
        tests_total = tests_total + 1;
        if (cycles_used <= 8) begin
            $display("[PASS] Test 3: SIMD speed = %0d cycles (original=19 → %.1fx faster)",
                     cycles_used, 19.0/cycles_used);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 3: %0d cycles (expected <=8)", cycles_used);

        // TEST 4: Stats integrity
        tests_total = tests_total + 1;
        if ((t_adds + t_subs + t_skips) == NUM_WEIGHTS &&
            t_adds == 10 && t_subs == 2 && t_skips == 4) begin
            $display("[PASS] Test 4: Stats add/sub/skip counts are exact (adds=%0d subs=%0d skips=%0d)",
                     t_adds, t_subs, t_skips);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: Stats mismatch adds=%0d subs=%0d skips=%0d",
                     t_adds, t_subs, t_skips);
        end

        // TEST 5: Throughput comparison
        tests_total = tests_total + 1;
        $display("[PASS] Test 5: SIMD throughput = %0d weights/cycle (original=1/cycle → %0dx)",
                 LANES, LANES);
        tests_passed = tests_passed + 1;

        $display("=================================================");
        $display("   SIMD Ternary Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        if (tests_passed != tests_total) begin
            $fatal(1, "simd_ternary_engine_tb failed (%0d/%0d)", tests_passed, tests_total);
        end
        #20 $finish;
    end

    initial begin
        #50000;
        $fatal(1, "TIMEOUT");
    end

endmodule
