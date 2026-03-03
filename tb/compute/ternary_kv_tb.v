// ============================================================================
// Testbench: ternary_kv_tb
// Combined test for ternary MAC + weight decoder + KV paging
// ============================================================================
`timescale 1ns/1ps

module ternary_kv_tb;

    reg clk, rst;
    always #5 clk = ~clk;

    integer pass_count, fail_count;

    // ---- Ternary Weight Decoder ----
    reg             dec_valid;
    reg  [15:0]     dec_packed;
    wire [63:0]     dec_weights;  // 8 × 8-bit flat
    wire            dec_valid_out;

    ternary_weight_decoder #(.WEIGHTS_PER_WORD(8)) u_dec (
        .clk(clk), .rst(rst), .valid_in(dec_valid),
        .packed_word(dec_packed),
        .weights_flat(dec_weights), .valid_out(dec_valid_out)
    );

    // ---- Ternary MAC Unit ----
    reg             mac_valid;
    reg  [7:0]      mac_weights;
    reg  [7:0]      mac_activation;
    wire signed [15:0] mac_acc;
    wire            mac_valid_out;
    wire [7:0]      mac_zeros;

    ternary_mac_unit #(.ACT_WIDTH(8), .NUM_WEIGHTS(4)) u_mac (
        .clk(clk), .rst(rst), .valid_in(mac_valid),
        .weights_packed(mac_weights),
        .activation_in(mac_activation),
        .acc_out(mac_acc), .valid_out(mac_valid_out),
        .zero_count(mac_zeros)
    );

    // ---- Page Allocator ----
    reg             alloc_req, free_req;
    reg  [5:0]      free_page;
    wire [5:0]      alloc_page;
    wire            alloc_valid;
    wire [6:0]      free_count;

    page_allocator #(.NUM_PAGES(64), .PAGE_ID_WIDTH(6)) u_alloc (
        .clk(clk), .rst(rst),
        .alloc_req(alloc_req), .alloc_page_id(alloc_page), .alloc_valid(alloc_valid),
        .free_req(free_req), .free_page_id(free_page),
        .free_count(free_count)
    );

    // ---- KV Page Table ----
    reg             pt_write, pt_read, pt_inval;
    reg  [5:0]      pt_write_pos, pt_write_page, pt_read_pos, pt_inval_pos;
    wire [5:0]      pt_read_page;
    wire            pt_read_valid;
    wire [6:0]      pt_active;

    kv_page_table #(.NUM_PAGES(64), .PAGE_ID_WIDTH(6), .MAX_SEQ_LEN(64), .SEQ_ID_WIDTH(6)) u_pt (
        .clk(clk), .rst(rst),
        .write_en(pt_write), .write_logical_pos(pt_write_pos), .write_page_id(pt_write_page),
        .read_en(pt_read), .read_logical_pos(pt_read_pos),
        .read_page_id(pt_read_page), .read_valid(pt_read_valid),
        .invalidate_en(pt_inval), .invalidate_pos(pt_inval_pos),
        .active_entries(pt_active)
    );

    initial begin
        clk = 0; rst = 1;
        pass_count = 0; fail_count = 0;
        dec_valid = 0; mac_valid = 0; alloc_req = 0; free_req = 0;
        pt_write = 0; pt_read = 0; pt_inval = 0;
        dec_packed = 0; mac_weights = 0; mac_activation = 0;
        free_page = 0; pt_write_pos = 0; pt_write_page = 0;
        pt_read_pos = 0; pt_inval_pos = 0;

        #20; rst = 0;

        // ==================================================================
        // TEST 1: Ternary Weight Decoder
        // Pack: [+1, -1, 0, +1, -1, 0, +1, -1] = 01_10_00_01_10_00_01_10
        // ==================================================================
        $display("\n=== TEST 1: Ternary Weight Decoder ===");
        @(posedge clk);
        // Bit layout: w7=10, w6=01, w5=00, w4=10, w3=01, w2=00, w1=10, w0=01
        dec_packed <= 16'b10_01_00_10_01_00_10_01;
        dec_valid  <= 1;
        @(posedge clk); dec_valid <= 0;
        @(posedge clk);

        $display("  Decoded: [%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                 $signed(dec_weights[7:0]), $signed(dec_weights[15:8]),
                 $signed(dec_weights[23:16]), $signed(dec_weights[31:24]),
                 $signed(dec_weights[39:32]), $signed(dec_weights[47:40]),
                 $signed(dec_weights[55:48]), $signed(dec_weights[63:56]));

        if ($signed(dec_weights[7:0]) == 1 && $signed(dec_weights[15:8]) == -1 &&
            $signed(dec_weights[23:16]) == 0 &&
            $signed(dec_weights[31:24]) == 1 && $signed(dec_weights[39:32]) == -1 &&
            $signed(dec_weights[47:40]) == 0) begin
            $display("  PASS: Correct ternary decoding");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Unexpected values");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 2: Ternary MAC
        // Weights: [+1, -1, 0, +1] = 01_10_00_01, activation = 42
        // Expected: 42 + (-42) + 0 + 42 = 42, zero_count = 1
        // ==================================================================
        $display("\n=== TEST 2: Ternary MAC ===");
        @(posedge clk);
        mac_weights    <= 8'b01_00_10_01;  // w0=+1, w1=-1, w2=0, w3=+1
        mac_activation <= 8'd42;
        mac_valid      <= 1;
        @(posedge clk); mac_valid <= 0;
        @(posedge clk);

        $display("  ACC: %0d (expected 42), zeros: %0d (expected 1)", mac_acc, mac_zeros);
        if (mac_acc == 42 && mac_zeros == 1) begin
            $display("  PASS: Correct ternary MAC result");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected acc=42, zeros=1");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 3: Page Allocator - alloc 3, free 1, check counts
        // ==================================================================
        $display("\n=== TEST 3: Page Allocator ===");
        $display("  Initial free pages: %0d", free_count);

        // Allocate 3 pages
        @(posedge clk); alloc_req <= 1;
        @(posedge clk); alloc_req <= 0; @(posedge clk);
        $display("  Allocated page %0d, free=%0d", alloc_page, free_count);

        @(posedge clk); alloc_req <= 1;
        @(posedge clk); alloc_req <= 0; @(posedge clk);
        $display("  Allocated page %0d, free=%0d", alloc_page, free_count);

        @(posedge clk); alloc_req <= 1;
        @(posedge clk); alloc_req <= 0; @(posedge clk);
        $display("  Allocated page %0d, free=%0d", alloc_page, free_count);

        if (free_count == 61) begin
            $display("  PASS: 3 pages allocated (64 - 3 = 61 free)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected 61 free, got %0d", free_count);
            fail_count = fail_count + 1;
        end

        // Free one page back
        @(posedge clk); free_req <= 1; free_page <= 6'd10;
        @(posedge clk); free_req <= 0; @(posedge clk);

        if (free_count == 62) begin
            $display("  PASS: Freed 1 page (62 free now)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected 62 free, got %0d", free_count);
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 4: KV Page Table - map, lookup, invalidate
        // ==================================================================
        $display("\n=== TEST 4: KV Page Table ===");

        // Map logical position 5 → physical page 42
        @(posedge clk);
        pt_write <= 1; pt_write_pos <= 6'd5; pt_write_page <= 6'd42;
        @(posedge clk); pt_write <= 0;
        @(posedge clk);

        // Lookup position 5
        pt_read <= 1; pt_read_pos <= 6'd5;
        @(posedge clk); pt_read <= 0;
        @(posedge clk);

        $display("  Lookup pos 5: page=%0d, valid=%0d", pt_read_page, pt_read_valid);
        if (pt_read_page == 42 && pt_read_valid == 1) begin
            $display("  PASS: Correct mapping (5 → 42)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected page=42, valid=1");
            fail_count = fail_count + 1;
        end

        // Lookup unmapped position 10
        @(posedge clk);
        pt_read <= 1; pt_read_pos <= 6'd10;
        @(posedge clk); pt_read <= 0;
        @(posedge clk);

        $display("  Lookup pos 10 (unmapped): valid=%0d", pt_read_valid);
        if (pt_read_valid == 0) begin
            $display("  PASS: Unmapped position returns invalid");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected valid=0 for unmapped");
            fail_count = fail_count + 1;
        end

        // Invalidate position 5
        @(posedge clk);
        pt_inval <= 1; pt_inval_pos <= 6'd5;
        @(posedge clk); pt_inval <= 0;
        @(posedge clk);

        // Lookup after invalidation
        @(posedge clk);
        pt_read <= 1; pt_read_pos <= 6'd5;
        @(posedge clk); pt_read <= 0;
        @(posedge clk);

        $display("  Lookup pos 5 (after invalidate): valid=%0d", pt_read_valid);
        if (pt_read_valid == 0) begin
            $display("  PASS: Invalidated position returns invalid");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected valid=0 after invalidation");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // SUMMARY
        // ==================================================================
        $display("\n=== TERNARY + KV PAGING TEST SUMMARY ===");
        $display("  Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ALL PASSED");
        else
            $display("  %0d FAILURES", fail_count);

        #50;
        $finish;
    end

endmodule
