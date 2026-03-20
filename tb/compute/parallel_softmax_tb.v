`timescale 1ns / 1ps

module parallel_softmax_tb;

    reg clk, rst;
    always #5 clk = ~clk;

    reg valid_in;
    reg [4*16-1:0] x_in;
    wire [4*8-1:0] prob_out;
    wire valid_out;
    wire [15:0] cycles_used;

    parallel_softmax #(.VECTOR_LEN(4), .PARALLEL_UNITS(4)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .prob_out(prob_out), .valid_out(valid_out),
        .cycles_used(cycles_used)
    );

    integer tests_passed, tests_total;
    reg [7:0] p0, p1, p2, p3;
    integer psum;
    reg got_valid;

    initial begin
        clk = 0; rst = 1; valid_in = 0; x_in = 0;
        tests_passed = 0; tests_total = 0;
        
        @(negedge clk); @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);

        $display("=================================================");
        $display("   Parallel Softmax Tests (4-wide SIMD)");
        $display("   All exp/normalize units run simultaneously");
        $display("=================================================");

        // TEST 1: Equal inputs — should give equal probabilities and sane sum
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0; @(negedge clk);
        x_in = {16'sd256, 16'sd256, 16'sd256, 16'sd256};
        valid_in = 1; @(negedge clk); valid_in = 0;
        got_valid = 0;
        begin : w1
            integer t;
            for (t = 0; t < 30; t = t + 1) begin
                @(negedge clk);
                if (valid_out) begin
                    got_valid = 1;
                    t = 30;
                end
            end
        end
        
        p0 = prob_out[0 +: 8];
        p1 = prob_out[8 +: 8];
        p2 = prob_out[16 +: 8];
        p3 = prob_out[24 +: 8];
        psum = p0 + p1 + p2 + p3;
        
        if (got_valid && p0 == p1 && p1 == p2 && p2 == p3 && (psum >= 252 && psum <= 255)) begin
            $display("[PASS] Test 1: Equal inputs -> equal probs [%0d,%0d,%0d,%0d], sum=%0d, %0d cycles",
                     p0, p1, p2, p3, psum, cycles_used);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 1: valid=%0d probs=[%0d,%0d,%0d,%0d] sum=%0d",
                          got_valid, p0, p1, p2, p3, psum);

        // TEST 2: One dominant input — highest input should have highest probability
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0; @(negedge clk);
        x_in = {16'sd512, 16'sd0, 16'sd0, 16'sd0}; // Element 3 much larger
        valid_in = 1; @(negedge clk); valid_in = 0;
        got_valid = 0;
        begin : w2
            integer t;
            for (t = 0; t < 30; t = t + 1) begin
                @(negedge clk);
                if (valid_out) begin
                    got_valid = 1;
                    t = 30;
                end
            end
        end
        
        p0 = prob_out[0 +: 8]; p1 = prob_out[8 +: 8];
        p2 = prob_out[16 +: 8]; p3 = prob_out[24 +: 8];
        psum = p0 + p1 + p2 + p3;
        
        if (got_valid && p3 > p0 && p3 > p1 && p3 > p2 &&
            p0 == p1 && p1 == p2 &&
            (psum >= 252 && psum <= 255)) begin
            $display("[PASS] Test 2: Dominant input ordering OK -> probs=[%0d,%0d,%0d,%0d], sum=%0d, %0d cycles",
                     p0, p1, p2, p3, psum, cycles_used);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 2: valid=%0d probs=[%0d,%0d,%0d,%0d] sum=%0d",
                          got_valid, p0, p1, p2, p3, psum);

        // TEST 3: Negative-valued ordering check (guards stale max reduction bugs)
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0; @(negedge clk);
        x_in = {-16'sd256, -16'sd64, -16'sd32, -16'sd1024}; // max at element 1
        valid_in = 1; @(negedge clk); valid_in = 0;
        got_valid = 0;
        begin : w3
            integer t;
            for (t = 0; t < 30; t = t + 1) begin
                @(negedge clk);
                if (valid_out) begin
                    got_valid = 1;
                    t = 30;
                end
            end
        end

        p0 = prob_out[0 +: 8]; p1 = prob_out[8 +: 8];
        p2 = prob_out[16 +: 8]; p3 = prob_out[24 +: 8];
        psum = p0 + p1 + p2 + p3;

        if (got_valid && p1 > p2 && p2 > p3 && p3 > p0 && (psum >= 252 && psum <= 255)) begin
            $display("[PASS] Test 3: Negative input ordering OK -> probs=[%0d,%0d,%0d,%0d], sum=%0d",
                     p0, p1, p2, p3, psum);
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 3: valid=%0d probs=[%0d,%0d,%0d,%0d] sum=%0d",
                          got_valid, p0, p1, p2, p3, psum);

        // TEST 4: Speed comparison vs serial softmax
        tests_total = tests_total + 1;
        if (cycles_used > 0 && cycles_used <= 8) begin
            $display("[PASS] Test 4: Parallel softmax = %0d cycles (serial=25 -> %.1fx faster)",
                     cycles_used, 25.0/(cycles_used > 0 ? cycles_used : 1));
            tests_passed = tests_passed + 1;
        end else $display("[FAIL] Test 4: %0d cycles (expected <= 8)", cycles_used);

        $display("=================================================");
        $display("   Parallel Softmax Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        #20 $finish;
    end

endmodule
