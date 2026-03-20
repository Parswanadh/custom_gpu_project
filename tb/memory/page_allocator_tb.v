`timescale 1ns / 1ps

module page_allocator_tb;

    localparam NUM_PAGES = 8;
    localparam PAGE_ID_WIDTH = 3;

    reg clk;
    reg rst;
    reg alloc_req;
    wire [PAGE_ID_WIDTH-1:0] alloc_page_id;
    wire alloc_valid;
    reg free_req;
    reg [PAGE_ID_WIDTH-1:0] free_page_id;
    wire [PAGE_ID_WIDTH:0] free_count;

    page_allocator #(
        .NUM_PAGES(NUM_PAGES),
        .PAGE_ID_WIDTH(PAGE_ID_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .alloc_req(alloc_req),
        .alloc_page_id(alloc_page_id),
        .alloc_valid(alloc_valid),
        .free_req(free_req),
        .free_page_id(free_page_id),
        .free_count(free_count)
    );

    always #5 clk = ~clk;

    integer tests_passed = 0;
    integer tests_total = 0;
    reg [PAGE_ID_WIDTH-1:0] first_alloc_page;
    reg [PAGE_ID_WIDTH-1:0] got_page;
    reg seen [0:NUM_PAGES-1];
    reg duplicate_seen;
    integer alloc_successes;
    integer i;

    initial begin
        clk = 0;
        rst = 1;
        alloc_req = 0;
        free_req = 0;
        free_page_id = 0;
        first_alloc_page = 0;
        got_page = 0;

        @(negedge clk);
        @(negedge clk);
        rst = 0;
        @(negedge clk);

        // ================================================================
        // TEST 1: Reset state
        // ================================================================
        tests_total = tests_total + 1;
        if (free_count == NUM_PAGES && !alloc_valid) begin
            $display("[PASS] Test 1: Reset initializes all pages as free");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: free_count=%0d alloc_valid=%b", free_count, alloc_valid);
        end

        // ================================================================
        // TEST 2: First allocation pops top of free stack
        // ================================================================
        tests_total = tests_total + 1;
        alloc_req = 1;
        @(negedge clk);
        first_alloc_page = alloc_page_id;
        if (alloc_valid && alloc_page_id == (NUM_PAGES - 1) && free_count == (NUM_PAGES - 1)) begin
            $display("[PASS] Test 2: First alloc returned page %0d", alloc_page_id);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: alloc_valid=%b page=%0d free_count=%0d",
                     alloc_valid, alloc_page_id, free_count);
        end
        alloc_req = 0;
        @(negedge clk);

        // ================================================================
        // TEST 3: Same-cycle alloc+free has deterministic net count
        // ================================================================
        tests_total = tests_total + 1;
        alloc_req = 1;
        free_req = 1;
        free_page_id = first_alloc_page;
        @(negedge clk);
        if (alloc_valid && alloc_page_id == (NUM_PAGES - 2) && free_count == (NUM_PAGES - 1)) begin
            $display("[PASS] Test 3: alloc/free same cycle is race-free");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: alloc_valid=%b page=%0d free_count=%0d",
                     alloc_valid, alloc_page_id, free_count);
        end
        alloc_req = 0;
        free_req = 0;
        @(negedge clk);

        // ================================================================
        // TEST 4: Double-free request is ignored
        // ================================================================
        tests_total = tests_total + 1;
        free_req = 1;
        free_page_id = first_alloc_page;
        @(negedge clk);
        if (free_count == (NUM_PAGES - 1)) begin
            $display("[PASS] Test 4: Double-free did not corrupt free count");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: free_count=%0d (expected %0d)", free_count, NUM_PAGES - 1);
        end
        free_req = 0;
        @(negedge clk);

        // ================================================================
        // TEST 5: Previously freed page can be allocated exactly once
        // ================================================================
        tests_total = tests_total + 1;
        alloc_req = 1;
        @(negedge clk);
        if (alloc_valid && alloc_page_id == first_alloc_page && free_count == (NUM_PAGES - 2)) begin
            $display("[PASS] Test 5: Freed page %0d recycled once", alloc_page_id);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 5: alloc_valid=%b page=%0d free_count=%0d",
                     alloc_valid, alloc_page_id, free_count);
        end
        alloc_req = 0;
        @(negedge clk);

        // ================================================================
        // TEST 6: Drain allocator and check uniqueness (no duplicates)
        // ================================================================
        tests_total = tests_total + 1;
        for (i = 0; i < NUM_PAGES; i = i + 1)
            seen[i] = 1'b0;
        seen[first_alloc_page] = 1'b1;     // Allocated in Test 5
        seen[NUM_PAGES - 2] = 1'b1;        // Allocated in Test 3
        duplicate_seen = 1'b0;
        alloc_successes = 0;

        for (i = 0; i < NUM_PAGES; i = i + 1) begin
            alloc_req = 1;
            @(negedge clk);
            if (alloc_valid) begin
                alloc_successes = alloc_successes + 1;
                got_page = alloc_page_id;
                if (seen[got_page])
                    duplicate_seen = 1'b1;
                seen[got_page] = 1'b1;
            end
            alloc_req = 0;
            @(negedge clk);
        end

        if (!duplicate_seen && alloc_successes == (NUM_PAGES - 2) && free_count == 0) begin
            $display("[PASS] Test 6: No duplicate allocations after double-free attempt");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 6: duplicate=%b alloc_successes=%0d free_count=%0d",
                     duplicate_seen, alloc_successes, free_count);
        end

        // ================================================================
        // TEST 7: Empty allocator does not underflow
        // ================================================================
        tests_total = tests_total + 1;
        alloc_req = 1;
        @(negedge clk);
        if (!alloc_valid && free_count == 0) begin
            $display("[PASS] Test 7: Empty allocator rejects alloc without corruption");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 7: alloc_valid=%b free_count=%0d", alloc_valid, free_count);
        end
        alloc_req = 0;

        $display("=================================================");
        $display("   Page Allocator Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        #10 $finish;
    end

    initial begin
        #10000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
