`timescale 1ns / 1ps

module paged_attention_mmu_tb;

    reg clk;
    reg rst;
    
    // Translation
    reg translate_valid;
    reg [4:0] virtual_page;
    reg [7:0] page_offset;
    wire [13:0] physical_addr;
    wire translate_done;
    wire page_fault;
    
    // Allocation
    reg alloc_valid;
    reg [4:0] alloc_virtual_page;
    wire [5:0] alloc_physical_page;
    wire alloc_done;
    wire alloc_fail;
    
    // Free
    reg free_valid;
    reg [4:0] free_virtual_page;
    wire free_done;
    
    // Stats
    wire [31:0] total_translations;
    wire [31:0] total_page_faults;
    wire [6:0] pages_allocated;
    wire [6:0] pages_free;

    paged_attention_mmu #(
        .PAGE_SIZE_BITS(8),
        .NUM_VIRTUAL_PAGES(32),
        .NUM_PHYSICAL_PAGES(64),
        .VIRT_ADDR_BITS(5),
        .PHYS_ADDR_BITS(6)
    ) dut (
        .clk(clk),
        .rst(rst),
        .translate_valid(translate_valid),
        .virtual_page(virtual_page),
        .page_offset(page_offset),
        .physical_addr(physical_addr),
        .translate_done(translate_done),
        .page_fault(page_fault),
        .alloc_valid(alloc_valid),
        .alloc_virtual_page(alloc_virtual_page),
        .alloc_physical_page(alloc_physical_page),
        .alloc_done(alloc_done),
        .alloc_fail(alloc_fail),
        .free_valid(free_valid),
        .free_virtual_page(free_virtual_page),
        .free_done(free_done),
        .total_translations(total_translations),
        .total_page_faults(total_page_faults),
        .pages_allocated(pages_allocated),
        .pages_free(pages_free)
    );

    always #5 clk = ~clk;

    integer tests_passed = 0;
    integer tests_total = 0;

    // Strategy: Assert input on negedge, check output on the NEXT negedge
    // (the posedge inbetween is when the FSM computes the result)
    
    initial begin
        clk = 0;
        rst = 1;
        translate_valid = 0;
        virtual_page = 0;
        page_offset = 0;
        alloc_valid = 0;
        alloc_virtual_page = 0;
        free_valid = 0;
        free_virtual_page = 0;
        
        @(negedge clk);
        @(negedge clk);
        rst = 0;
        @(negedge clk);
        
        $display("=================================================");
        $display("   PagedAttention MMU Tests");
        $display("   (Virtual KV Cache Page Management)");
        $display("=================================================");

        // ================================================================
        // TEST 1: Allocate page, verify mapping
        // Cycle 0: assert alloc on negedge
        // Cycle 1: posedge processes, result ready on NEXT negedge
        // ================================================================
        tests_total = tests_total + 1;
        alloc_valid = 1;
        alloc_virtual_page = 5'd3;
        @(negedge clk);  // posedge processes, outputs updated
        // Check right here — done flag is set on this posedge
        if (alloc_done && !alloc_fail && alloc_physical_page == 6'd0) begin
            $display("[PASS] Test 1: Allocated vpage 3 -> ppage 0");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: alloc_done=%b fail=%b phys=%d", alloc_done, alloc_fail, alloc_physical_page);
        end
        alloc_valid = 0;
        @(negedge clk); // Let done clear

        // ================================================================
        // TEST 2: Translate virtual address to physical
        // ================================================================
        tests_total = tests_total + 1;
        translate_valid = 1;
        virtual_page = 5'd3;
        page_offset = 8'hAB;
        @(negedge clk);  // posedge processes
        if (translate_done && !page_fault && physical_addr == {6'd0, 8'hAB}) begin
            $display("[PASS] Test 2: Translated vpage 3 + offset 0xAB -> phys 0x%04h", physical_addr);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: done=%b fault=%b addr=0x%04h (expected 0x%04h)", 
                     translate_done, page_fault, physical_addr, {6'd0, 8'hAB});
        end
        translate_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 3: Page fault on unmapped page
        // ================================================================
        tests_total = tests_total + 1;
        translate_valid = 1;
        virtual_page = 5'd10;
        page_offset = 8'h00;
        @(negedge clk);
        if (translate_done && page_fault) begin
            $display("[PASS] Test 3: Page fault on unmapped vpage 10");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: done=%b fault=%b", translate_done, page_fault);
        end
        translate_valid = 0;
        @(negedge clk);

        // ================================================================  
        // TEST 4: Allocate second page, verify non-contiguous
        // ================================================================
        tests_total = tests_total + 1;
        alloc_valid = 1;
        alloc_virtual_page = 5'd10;
        @(negedge clk);
        if (alloc_done && !alloc_fail && alloc_physical_page == 6'd1) begin
            $display("[PASS] Test 4: Allocated vpage 10 -> ppage 1 (non-contiguous mapping)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: alloc_done=%b fail=%b phys=%d", alloc_done, alloc_fail, alloc_physical_page);
        end
        alloc_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 5: Free a page and reallocate
        // ================================================================
        tests_total = tests_total + 1;
        free_valid = 1;
        free_virtual_page = 5'd3;
        @(negedge clk);
        free_valid = 0;
        @(negedge clk);
        
        // Allocate again — should get physical page 0 back
        alloc_valid = 1;
        alloc_virtual_page = 5'd20;
        @(negedge clk);
        if (alloc_done && !alloc_fail && alloc_physical_page == 6'd0) begin
            $display("[PASS] Test 5: Freed ppage 0, re-allocated to vpage 20");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 5: alloc_done=%b fail=%b phys=%d", alloc_done, alloc_fail, alloc_physical_page);
        end
        alloc_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 6: Remap existing virtual page
        // ================================================================
        tests_total = tests_total + 1;
        alloc_valid = 1;
        alloc_virtual_page = 5'd10; // Currently mapped to ppage 1
        @(negedge clk);
        if (alloc_done && !alloc_fail && alloc_physical_page == 6'd2) begin
            $display("[PASS] Test 6: Remapped vpage 10 -> ppage 2");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 6: alloc_done=%b fail=%b phys=%d", alloc_done, alloc_fail, alloc_physical_page);
        end
        alloc_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 7: Translation sees remapped page
        // ================================================================
        tests_total = tests_total + 1;
        translate_valid = 1;
        virtual_page = 5'd10;
        page_offset = 8'h34;
        @(negedge clk);
        if (translate_done && !page_fault && physical_addr == {6'd2, 8'h34}) begin
            $display("[PASS] Test 7: Translation reflects remap to ppage 2");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 7: done=%b fault=%b addr=0x%04h (expected 0x%04h)",
                     translate_done, page_fault, physical_addr, {6'd2, 8'h34});
        end
        translate_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 8: Old remap target page is released back to free pool
        // ================================================================
        tests_total = tests_total + 1;
        alloc_valid = 1;
        alloc_virtual_page = 5'd21;
        @(negedge clk);
        if (alloc_done && !alloc_fail && alloc_physical_page == 6'd1) begin
            $display("[PASS] Test 8: Old remap page ppage 1 was safely recycled");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 8: alloc_done=%b fail=%b phys=%d", alloc_done, alloc_fail, alloc_physical_page);
        end
        alloc_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 9: Concurrent alloc+free is fail-closed (race hardening)
        // ================================================================
        tests_total = tests_total + 1;
        alloc_valid = 1;
        alloc_virtual_page = 5'd22;
        free_valid = 1;
        free_virtual_page = 5'd31; // unmapped page; should be a no-op free
        @(negedge clk);
        if (alloc_done && alloc_fail && free_done) begin
            $display("[PASS] Test 9: Concurrent alloc/free rejected deterministically");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 9: alloc_done=%b alloc_fail=%b free_done=%b",
                     alloc_done, alloc_fail, free_done);
        end
        alloc_valid = 0;
        free_valid = 0;
        @(negedge clk);

        // ================================================================
        // TEST 10: Stats tracking
        // ================================================================
        tests_total = tests_total + 1;
        if (total_translations == 3 && total_page_faults == 1 &&
            pages_allocated == 3 && pages_free == 61) begin
            $display("[PASS] Test 10: Stats correct (3 xlat, 1 fault, 3 alloc, 61 free)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 10: xlat=%0d faults=%0d alloc=%0d free=%0d",
                     total_translations, total_page_faults, pages_allocated, pages_free);
        end

        $display("=================================================");
        $display("   PagedAttention MMU Tests: %0d / %0d PASSED", tests_passed, tests_total);
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
