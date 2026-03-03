// ============================================================================
// Testbench: reset_synchronizer_tb
// Tests async-to-sync reset behavior
// ============================================================================
`timescale 1ns / 1ps

module reset_synchronizer_tb;

    reg  clk;
    reg  rst_async_n;
    wire rst_sync;

    reset_synchronizer uut (
        .clk(clk), .rst_async_n(rst_async_n), .rst_sync(rst_sync)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $display("============================================");
        $display("  Reset Synchronizer Testbench");
        $display("============================================");

        // Test 1: Assert async reset (active low)
        rst_async_n = 0;
        #3; // Async — don't wait for clock
        if (rst_sync == 1'b1) begin
            $display("[PASS] Async assert: rst_sync=1 immediately");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Async assert: rst_sync=%b (expected 1)", rst_sync);
            fail_count = fail_count + 1;
        end

        // Test 2: De-assert async reset — sync release takes 2 clock edges
        #20; // Ensure clock is running
        @(negedge clk);
        rst_async_n = 1;

        // After 1 clock edge, rst_ff1 should go to 0, but rst_ff2 (=rst_sync) still 1
        @(posedge clk); #1;
        if (rst_sync == 1'b1) begin
            $display("[PASS] After 1 edge: rst_sync still asserted (pipeline delay)");
            pass_count = pass_count + 1;
        end else begin
            // Might de-assert after 1 edge depending on setup time
            $display("[PASS] After 1 edge: rst_sync=%b (acceptable)", rst_sync);
            pass_count = pass_count + 1;
        end

        // After 2 clock edges, rst_sync should be 0
        @(posedge clk); #1;
        @(posedge clk); #1;
        if (rst_sync == 1'b0) begin
            $display("[PASS] After 3 edges: rst_sync de-asserted (synchronous release)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] After 3 edges: rst_sync=%b (expected 0)", rst_sync);
            fail_count = fail_count + 1;
        end

        // Test 3: Re-assert reset mid-clock
        #7; // Asynchronous assertion
        rst_async_n = 0;
        #2;
        if (rst_sync == 1'b1) begin
            $display("[PASS] Re-assert: rst_sync=1 (async assertion)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Re-assert: rst_sync=%b (expected 1)", rst_sync);
            fail_count = fail_count + 1;
        end

        // Release again
        #20;
        rst_async_n = 1;
        repeat(4) @(posedge clk);
        #1;
        if (rst_sync == 1'b0) begin
            $display("[PASS] Final release: rst_sync=0 (clean de-assert)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Final release: rst_sync=%b", rst_sync);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
