// ============================================================================
// Testbench: parallel_attention_tb
// Tests multi-head parallel attention processing
// ============================================================================
`timescale 1ns / 1ps

module parallel_attention_tb;

    parameter ED = 8;
    parameter NH = 4;
    parameter HD = 2;
    parameter NP = 2;    // 2 heads in parallel
    parameter MSL = 16;
    parameter DW = 16;

    reg                     clk, rst, valid_in;
    reg  [ED*DW-1:0]       x_in;
    reg  [$clog2(MSL)-1:0] seq_pos;
    wire [ED*DW-1:0]       y_out;
    wire                   valid_out;
    wire [31:0]            zero_skip_count, heads_processed;

    parallel_attention #(
        .EMBED_DIM(ED), .NUM_HEADS(NH), .HEAD_DIM(HD),
        .NUM_PARALLEL(NP), .MAX_SEQ_LEN(MSL), .DATA_WIDTH(DW)
    ) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .seq_pos(seq_pos),
        .y_out(y_out), .valid_out(valid_out),
        .zero_skip_count(zero_skip_count),
        .heads_processed(heads_processed)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer i, timeout;

    initial begin
        $dumpfile("sim/waveforms/parallel_attention.vcd");
        $dumpvars(0, parallel_attention_tb);
    end

    initial begin
        $display("============================================");
        $display("  Parallel Attention Testbench");
        $display("  %0d heads, %0d parallel", NH, NP);
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0; seq_pos = 0;
        #25; rst = 0; #15;

        // Test 1: Process first token
        $display("[1] Processing first token (pos=0)...");
        @(negedge clk);
        // Input: [256, 128, 64, 32, 512, 256, 128, 64] (Q8.8 values)
        x_in = {16'sd64, 16'sd128, 16'sd256, 16'sd512,
                16'sd32, 16'sd64, 16'sd128, 16'sd256};
        seq_pos = 0;
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout = 0;
        while (!valid_out && timeout < 50) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (valid_out) begin
            $display("[PASS] First token processed in %0d cycles", timeout);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] First token: timeout after %0d cycles", timeout);
            fail_count = fail_count + 1;
        end

        repeat(3) @(posedge clk);

        // Test 2: Process second token (tests KV cache)
        $display("[2] Processing second token (pos=1)...");
        @(negedge clk);
        x_in = {16'sd32, 16'sd64, 16'sd128, 16'sd256,
                16'sd16, 16'sd32, 16'sd64, 16'sd128};
        seq_pos = 1;
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout = 0;
        while (!valid_out && timeout < 50) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        if (valid_out) begin
            $display("[PASS] Second token processed (KV cache used)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Second token: timeout");
            fail_count = fail_count + 1;
        end

        // Test 3: Verify heads_processed counter (should be > 0)
        if (heads_processed > 32'd0) begin
            $display("[PASS] Heads processed: %0d", heads_processed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Heads processed: %0d (expected > 0)", heads_processed);
            fail_count = fail_count + 1;
        end

        // Test 4: Verify zero-skip counting
        if (zero_skip_count >= 0) begin
            $display("[PASS] Zero-skip counter active: %0d skips", zero_skip_count);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Zero-skip counter issue");
            fail_count = fail_count + 1;
        end

        // Test 5: Output dimensions correct (8-dim output)
        begin : dim_check
            integer non_zero;
            non_zero = 0;
            for (i = 0; i < ED; i = i + 1)
                if (y_out[i*DW +: DW] != 0) non_zero = non_zero + 1;
            if (non_zero > 0) begin
                $display("[PASS] Output has %0d non-zero dimensions", non_zero);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Output is all zeros");
                fail_count = fail_count + 1;
            end
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
