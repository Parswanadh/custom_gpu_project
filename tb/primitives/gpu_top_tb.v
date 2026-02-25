// ============================================================================
// Testbench: gpu_top_tb
// Integration test: loads sparse weights, runs inference, checks pipeline
// ============================================================================
`timescale 1ns / 1ps

module gpu_top_tb;

    reg         clk, rst;
    reg  [1:0]  mode;
    reg  [3:0]  dq_scale, dq_offset;
    reg         mem_write_en;
    reg  [7:0]  mem_write_val;
    reg  [3:0]  mem_write_idx;
    reg         start;
    reg  [3:0]  weight_addr;
    reg  [7:0]  activation_in;
    wire [63:0] result_out;
    wire        valid_out;
    wire        zero_skipped;

    gpu_top uut (
        .clk(clk), .rst(rst),
        .mode(mode), .dq_scale(dq_scale), .dq_offset(dq_offset),
        .mem_write_en(mem_write_en), .mem_write_val(mem_write_val), .mem_write_idx(mem_write_idx),
        .start(start), .weight_addr(weight_addr), .activation_in(activation_in),
        .result_out(result_out), .valid_out(valid_out), .zero_skipped(zero_skipped)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task write_weight;
        input [7:0] val;
        input [3:0] idx;
        begin
            @(negedge clk);
            mem_write_en  = 1'b1;
            mem_write_val = val;
            mem_write_idx = idx;
            @(negedge clk);
            mem_write_en = 1'b0;
        end
    endtask

    task run_inference;
        input [3:0]  addr;
        input [7:0]  act;
        input        expect_skip;
        input [80*8-1:0] test_name;
        integer timeout;
        begin
            @(negedge clk);
            weight_addr   = addr;
            activation_in = act;
            start         = 1'b1;
            @(negedge clk);
            start = 1'b0;

            // Wait for valid_out (max 30 cycles for full pipeline)
            timeout = 0;
            while (!valid_out && timeout < 30) begin
                @(negedge clk);
                timeout = timeout + 1;
            end

            if (valid_out) begin
                $display("[INFO] %0s | addr=%0d act=%0d => result=%0d skipped=%0b (cycles=%0d)",
                         test_name, addr, act, result_out[15:0], zero_skipped, timeout);
                if (zero_skipped === expect_skip) begin
                    $display("[PASS] %0s", test_name);
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] %0s | skip=%0b (expected %0b)", test_name, zero_skipped, expect_skip);
                    fail_count = fail_count + 1;
                end
            end else begin
                $display("[FAIL] %0s | TIMEOUT after %0d cycles", test_name, timeout);
                fail_count = fail_count + 1;
            end

            // Wait a few cycles between tests
            repeat(3) @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/gpu_top.vcd");
        $dumpvars(0, gpu_top_tb);
    end

    initial begin
        $display("============================================");
        $display("  GPU Top-Level Pipeline Testbench");
        $display("============================================");

        rst = 1; start = 0; mem_write_en = 0; mem_write_val = 0; mem_write_idx = 0;
        weight_addr = 0; activation_in = 0;
        mode = 2'b10;       // 16-bit mode
        dq_scale = 4'd2;    // scale = 2
        dq_offset = 4'd0;   // offset = 0
        #35; rst = 0; #25;

        // Load weights
        write_weight(8'd6, 4'd0);   // idx 0: val 6 → dequant (6-0)*2=12
        write_weight(8'd10, 4'd3);  // idx 3: val 10 → dequant (10-0)*2=20
        write_weight(8'd0, 4'd7);   // idx 7: val 0 → dequant=0

        #20;

        // Test 1: weight@0 dq=12, act=5 → 12*5=60, no skip
        run_inference(4'd0, 8'd5, 1'b0, "Weight=6 dq=12, act=5");

        // Test 2: weight@3 dq=20, act=3 → 20*3=60, no skip
        run_inference(4'd3, 8'd3, 1'b0, "Weight=10 dq=20, act=3");

        // Test 3: weight@7 dq=0, act=9 → skip
        run_inference(4'd7, 8'd9, 1'b1, "Weight=0 dq=0 SKIP");

        // Test 4: missing weight (idx 5) → 0, skip
        run_inference(4'd5, 8'd4, 1'b1, "Missing weight SKIP");

        // Test 5: zero activation, skip
        run_inference(4'd0, 8'd0, 1'b1, "Zero activation SKIP");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
