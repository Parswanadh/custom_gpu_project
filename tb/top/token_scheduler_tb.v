// ============================================================================
// Testbench: token_scheduler_tb
// Tests autonomous token generation loop
// ============================================================================
`timescale 1ns / 1ps

module token_scheduler_tb;

    parameter VB = 4;   // VOCAB_BITS
    parameter SB = 4;   // SEQ_BITS
    parameter MGL = 16; // MAX_GEN_LEN

    reg                     clk, rst;
    reg                     start;
    reg  [VB-1:0]          seed_token;
    reg  [7:0]             num_tokens;

    wire                   engine_valid_in;
    wire [VB-1:0]          engine_token_in;
    wire [SB-1:0]          engine_position;
    reg                    engine_valid_out;
    reg  [VB-1:0]          engine_token_out;

    wire [MGL*VB-1:0]     generated_sequence;
    wire [7:0]            tokens_generated;
    wire                  generation_done;
    wire                  busy;

    token_scheduler #(
        .VOCAB_BITS(VB), .SEQ_BITS(SB), .MAX_GEN_LEN(MGL)
    ) uut (
        .clk(clk), .rst(rst),
        .start(start), .seed_token(seed_token), .num_tokens(num_tokens),
        .engine_valid_in(engine_valid_in), .engine_token_in(engine_token_in),
        .engine_position(engine_position),
        .engine_valid_out(engine_valid_out), .engine_token_out(engine_token_out),
        .generated_sequence(generated_sequence),
        .tokens_generated(tokens_generated),
        .generation_done(generation_done), .busy(busy)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer cycle_count;

    // Simple mock engine: output = (input + 1) % 16
    always @(posedge clk) begin
        engine_valid_out <= 1'b0;
        if (engine_valid_in) begin
            engine_token_out <= (engine_token_in + 1) & 4'hF;
            engine_valid_out <= 1'b1;
        end
    end

    initial begin
        $dumpfile("sim/waveforms/token_scheduler.vcd");
        $dumpvars(0, token_scheduler_tb);
    end

    initial begin
        $display("============================================");
        $display("  Token Scheduler Testbench");
        $display("============================================");

        rst = 1; start = 0; seed_token = 0; num_tokens = 0;
        engine_valid_out = 0; engine_token_out = 0;
        #25; rst = 0; #15;

        // Test 1: Generate 4 tokens starting from token 2
        // Expected: 2 → 3 → 4 → 5 → 6
        $display("[1] Generating 4 tokens from seed=2...");
        @(posedge clk);
        seed_token = 4'd2;
        num_tokens = 8'd4;
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        // Wait for generation to complete
        cycle_count = 0;
        while (!generation_done && cycle_count < 100) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        if (generation_done && tokens_generated == 8'd4) begin
            $display("[PASS] Generated %0d tokens in %0d cycles", tokens_generated, cycle_count);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Generation: done=%b, count=%0d", generation_done, tokens_generated);
            fail_count = fail_count + 1;
        end

        // Verify sequence: seed=2, then 3, 4, 5
        if (generated_sequence[0 +: VB] == 4'd2) begin
            $display("[PASS] Seed token correct: %0d", generated_sequence[0 +: VB]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Seed token: %0d (expected 2)", generated_sequence[0 +: VB]);
            fail_count = fail_count + 1;
        end

        // Test 2: Verify busy flag
        #20;
        @(posedge clk);
        seed_token = 4'd5;
        num_tokens = 8'd2;
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        if (busy) begin
            $display("[PASS] Busy flag set during generation");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Busy flag not set");
            fail_count = fail_count + 1;
        end

        // Wait for completion
        while (!generation_done) @(posedge clk);

        if (!busy) begin
            $display("[PASS] Busy flag cleared after completion");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Busy flag still set after completion");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
