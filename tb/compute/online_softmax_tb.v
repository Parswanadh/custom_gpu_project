// ============================================================================
// Testbench: online_softmax_tb
// Description: Tests the online softmax unit against known golden values.
//
// Test 1: Two equal scores → should produce average of the two V vectors
//   score[0] = 0, V[0] = [256, 0, 0, 0]     (1.0 in Q8.8)
//   score[1] = 0, V[1] = [0, 256, 0, 0]
//   Expected: softmax([0,0]) = [0.5, 0.5]
//   Output ≈ [128, 128, 0, 0]  (0.5 in Q8.8)
//
// Test 2: One dominant score → should output nearly the dominant V
//   score[0] = 512 (2.0), V[0] = [256, 0, 128, 0]
//   score[1] = 0  (0.0), V[1] = [0, 256, 0, 128]
//   exp(2) / (exp(2) + exp(0)) ≈ 0.88
//   exp(0) / (exp(2) + exp(0)) ≈ 0.12
//   Output ≈ [225, 30, 113, 15]
//
// Test 3: Three scores with clear ordering → verify monotonic attention
// ============================================================================
`timescale 1ns/1ps

module online_softmax_tb;

    parameter EMBED_DIM  = 4;
    parameter DATA_WIDTH = 16;

    reg                              clk;
    reg                              rst;
    reg                              score_valid;
    reg  signed [DATA_WIDTH-1:0]     score_in;
    reg  [EMBED_DIM*DATA_WIDTH-1:0]  value_in;
    reg                              start;
    reg                              finalize;
    wire [EMBED_DIM*DATA_WIDTH-1:0]  result_out;
    wire                             result_valid;

    // DUT
    online_softmax_unit #(
        .EMBED_DIM(EMBED_DIM),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .rst(rst),
        .score_valid(score_valid),
        .score_in(score_in),
        .value_in(value_in),
        .start(start),
        .finalize(finalize),
        .result_out(result_out),
        .result_valid(result_valid)
    );

    // Clock: 100 MHz
    always #5 clk = ~clk;

    integer test_num;
    integer pass_count;
    integer fail_count;
    reg signed [DATA_WIDTH-1:0] out_vals [0:EMBED_DIM-1];
    integer d;

    task extract_results;
        begin
            for (d = 0; d < EMBED_DIM; d = d + 1)
                out_vals[d] = $signed(result_out[d*DATA_WIDTH +: DATA_WIDTH]);
        end
    endtask

    task feed_score_and_value;
        input signed [DATA_WIDTH-1:0] s;
        input signed [DATA_WIDTH-1:0] v0, v1, v2, v3;
        begin
            @(posedge clk);
            score_valid <= 1;
            score_in    <= s;
            value_in    <= {v3, v2, v1, v0};  // Pack as flat vector
            @(posedge clk);
            score_valid <= 0;
            // Wait 2 cycles for pipeline to settle
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    task do_finalize;
        begin
            @(posedge clk);
            finalize <= 1;
            @(posedge clk);
            finalize <= 0;
            // Wait for result
            begin : wait_result
                integer timeout;
                timeout = 0;
                while (!result_valid && timeout < 100) begin
                    @(posedge clk);
                    timeout = timeout + 1;
                end
                if (timeout >= 100) begin
                    $display("  [TIMEOUT] Finalize did not produce result!");
                    fail_count = fail_count + 1;
                end
            end
        end
    endtask

    initial begin
        clk = 0;
        rst = 1;
        score_valid = 0;
        score_in = 0;
        value_in = 0;
        start = 0;
        finalize = 0;
        pass_count = 0;
        fail_count = 0;
        test_num = 0;

        // Reset
        #20;
        rst = 0;
        #10;

        // ==================================================================
        // TEST 1: Two equal scores → average of V vectors
        // ==================================================================
        test_num = 1;
        $display("\n=== TEST %0d: Equal scores → average of V vectors ===", test_num);

        // Start new softmax
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        @(posedge clk);

        // Feed score 0, V = [256, 0, 0, 0]  (1.0, 0, 0, 0 in Q8.8)
        feed_score_and_value(16'sd0, 16'sd256, 16'sd0, 16'sd0, 16'sd0);

        // Feed score 0, V = [0, 256, 0, 0]
        feed_score_and_value(16'sd0, 16'sd0, 16'sd256, 16'sd0, 16'sd0);

        do_finalize;
        extract_results;
        $display("  Output: [%0d, %0d, %0d, %0d]", out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        // Expected: ~[128, 128, 0, 0] (each should be ~0.5 × 256 = 128 in Q8.8)
        if (out_vals[0] > 100 && out_vals[0] < 160 &&
            out_vals[1] > 100 && out_vals[1] < 160 &&
            out_vals[2] >= -10 && out_vals[2] <= 10 &&
            out_vals[3] >= -10 && out_vals[3] <= 10) begin
            $display("  PASS: Outputs approximately equal (expected ~128 each)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected ~[128, 128, 0, 0]");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 2: One dominant score → output biased toward dominant V
        // ==================================================================
        test_num = 2;
        $display("\n=== TEST %0d: Dominant score → biased output ===", test_num);

        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        @(posedge clk);

        // Score = 512 (2.0 in Q8.8), V = [256, 0, 128, 0]
        feed_score_and_value(16'sd512, 16'sd256, 16'sd0, 16'sd128, 16'sd0);

        // Score = 0, V = [0, 256, 0, 128]
        feed_score_and_value(16'sd0, 16'sd0, 16'sd256, 16'sd0, 16'sd128);

        do_finalize;
        extract_results;
        $display("  Output: [%0d, %0d, %0d, %0d]", out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        // softmax([2.0, 0.0]) ≈ [0.88, 0.12]
        // out[0] ≈ 0.88 * 256 = 225, out[1] ≈ 0.12 * 256 = 31
        if (out_vals[0] > out_vals[1]) begin
            $display("  PASS: Dim 0 > Dim 1 (dominant score wins)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected dim 0 > dim 1");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 3: Three scores with ordering → verify monotonic
        // ==================================================================
        test_num = 3;
        $display("\n=== TEST %0d: Three scores with ordering ===", test_num);

        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        @(posedge clk);

        // Highest score → V = [256, 0, 0, 0]
        feed_score_and_value(16'sd768, 16'sd256, 16'sd0, 16'sd0, 16'sd0);
        // Medium score → V = [0, 256, 0, 0]
        feed_score_and_value(16'sd256, 16'sd0, 16'sd256, 16'sd0, 16'sd0);
        // Lowest score → V = [0, 0, 256, 0]
        feed_score_and_value(16'sd0, 16'sd0, 16'sd0, 16'sd256, 16'sd0);

        do_finalize;
        extract_results;
        $display("  Output: [%0d, %0d, %0d, %0d]", out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        if (out_vals[0] > out_vals[1] && out_vals[1] > out_vals[2]) begin
            $display("  PASS: dim0 > dim1 > dim2 (monotonic ordering preserved)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected dim0 > dim1 > dim2");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // SUMMARY
        // ==================================================================
        $display("\n=== ONLINE SOFTMAX TEST SUMMARY ===");
        $display("  Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ALL PASSED ✓");
        else
            $display("  %0d FAILURES", fail_count);

        #100;
        $finish;
    end

endmodule
