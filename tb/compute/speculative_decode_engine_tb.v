`timescale 1ns / 1ps

module speculative_decode_engine_tb;

    reg clk;
    reg rst;
    
    // Draft prediction
    reg predict_valid;
    reg [7:0] prev_token;
    wire [23:0] draft_tokens;   // 3 * 8 bits
    wire draft_valid;
    
    // Cache programming
    reg cache_write_en;
    reg [5:0] cache_write_addr;
    reg [23:0] cache_write_data;
    
    // Verification
    reg verify_valid;
    reg [23:0] actual_tokens;
    wire [3:0] accepted_count;
    wire verify_done;
    wire all_accepted;
    
    // Stats
    wire [31:0] total_predictions;
    wire [31:0] total_accepted;
    wire [31:0] total_rejected;

    speculative_decode_engine #(
        .VOCAB_BITS(8),
        .DRAFT_LEN(3),
        .CACHE_DEPTH(64),
        .CACHE_ADDR(6)
    ) dut (
        .clk(clk),
        .rst(rst),
        .predict_valid(predict_valid),
        .prev_token(prev_token),
        .draft_tokens(draft_tokens),
        .draft_valid(draft_valid),
        .cache_write_en(cache_write_en),
        .cache_write_addr(cache_write_addr),
        .cache_write_data(cache_write_data),
        .verify_valid(verify_valid),
        .actual_tokens(actual_tokens),
        .accepted_count(accepted_count),
        .verify_done(verify_done),
        .all_accepted(all_accepted),
        .total_predictions(total_predictions),
        .total_accepted(total_accepted),
        .total_rejected(total_rejected)
    );

    always #5 clk = ~clk;

    integer tests_passed = 0;
    integer tests_total = 0;

    initial begin
        clk = 0;
        rst = 1;
        predict_valid = 0;
        prev_token = 0;
        cache_write_en = 0;
        cache_write_addr = 0;
        cache_write_data = 0;
        verify_valid = 0;
        actual_tokens = 0;
        
        #20 rst = 0;
        
        $display("=================================================");
        $display("   Speculative Decoding Engine Tests");
        $display("   (N-gram Cache + Prefix Verification)");
        $display("=================================================");

        // ---- Program the n-gram cache ----
        // Entry 5: "The" → predicts ["quick", "brown", "fox"] = [10, 20, 30]
        @(negedge clk);
        cache_write_en = 1;
        cache_write_addr = 6'd5;
        cache_write_data = {8'd30, 8'd20, 8'd10};  // fox, brown, quick
        @(negedge clk);
        cache_write_en = 0;
        
        // Entry 42: "Hello" → predicts ["world", "!", "how"] = [50, 33, 99]
        @(negedge clk);
        cache_write_en = 1;
        cache_write_addr = 6'd42;
        cache_write_data = {8'd99, 8'd33, 8'd50};
        @(negedge clk);
        cache_write_en = 0;

        // ================================================================
        // TEST 1: Cache hit — all 3 drafts match actual output
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk);
        predict_valid = 1;
        prev_token = 8'd5;  // "The"
        @(negedge clk);
        predict_valid = 0;
        
        // Wait for draft
        @(negedge clk);
        
        // Verify: actual matches exactly
        verify_valid = 1;
        actual_tokens = {8'd30, 8'd20, 8'd10};  // Exact match!
        @(negedge clk);
        verify_valid = 0;
        @(negedge clk);
        
        if (accepted_count == 3 && all_accepted == 1) begin
            $display("[PASS] Test 1: All 3 drafts accepted (perfect prediction)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: Expected 3 accepted, got %d (all=%b)", accepted_count, all_accepted);
        end

        // ================================================================
        // TEST 2: Partial match — first 2 match, 3rd mismatches
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk);
        predict_valid = 1;
        prev_token = 8'd5;
        @(negedge clk);
        predict_valid = 0;
        @(negedge clk);
        
        verify_valid = 1;
        actual_tokens = {8'd99, 8'd20, 8'd10};  // first 2 match, 3rd different
        @(negedge clk);
        verify_valid = 0;
        @(negedge clk);
        
        if (accepted_count == 2 && all_accepted == 0) begin
            $display("[PASS] Test 2: 2 of 3 drafts accepted (partial match)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: Expected 2 accepted, got %d", accepted_count);
        end

        // ================================================================
        // TEST 3: Complete mismatch — first token wrong
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk);
        predict_valid = 1;
        prev_token = 8'd5;
        @(negedge clk);
        predict_valid = 0;
        @(negedge clk);
        
        verify_valid = 1;
        actual_tokens = {8'd30, 8'd20, 8'd77};  // first token wrong
        @(negedge clk);
        verify_valid = 0;
        @(negedge clk);
        
        if (accepted_count == 0) begin
            $display("[PASS] Test 3: 0 drafts accepted (first token mismatch)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: Expected 0 accepted, got %d", accepted_count);
        end

        // ================================================================
        // TEST 4: Cache miss — uncached token returns zeros
        // ================================================================
        tests_total = tests_total + 1;
        @(negedge clk);
        predict_valid = 1;
        prev_token = 8'd99;  // Not in cache
        @(negedge clk);
        predict_valid = 0;
        @(negedge clk);  // Wait for output to register
        
        if (draft_tokens == 24'd0) begin
            $display("[PASS] Test 4: Cache miss returns zero draft (graceful fallback)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: Expected zero draft on cache miss, got %h", draft_tokens);
        end

        // ================================================================
        // TEST 5: Statistics tracking
        // ================================================================
        tests_total = tests_total + 1;
        if (total_predictions == 4 && total_accepted == 5 && total_rejected == 4) begin
            $display("[PASS] Test 5: Stats correct (4 predictions, 5 accepted, 4 rejected)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 5: Stats wrong (pred=%0d, acc=%0d, rej=%0d)", 
                     total_predictions, total_accepted, total_rejected);
        end

        $display("=================================================");
        $display("   SpecDecode Tests Complete: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        
        #10 $finish;
    end
    
    // Safety timeout
    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
