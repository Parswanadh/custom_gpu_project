`timescale 1ns / 1ps

// ============================================================================
// Module: paged_attention_mmu
// Description: Hardware PagedAttention Memory Management Unit.
//   Implements virtual-to-physical KV cache page translation in silicon,
//   inspired by vLLM's PagedAttention algorithm.
//
//   Problem: Standard KV cache allocates contiguous memory per sequence.
//   With multiple concurrent users, memory fragments and wastes >50% RAM.
//
//   Solution: Treat KV cache memory like OS virtual memory pages:
//     - Each sequence gets a page table mapping virtual → physical pages
//     - Physical pages can be scattered across memory
//     - When attention requests KV[sequence][position], the MMU translates
//       the logical address to the physical page and offset
//
//   Architecture:
//     ┌─────────────────────────────────────────────┐
//     │          Page Table (32 entries)              │
//     │  [virtual_page] → [physical_page, valid]     │
//     └─────────────────────────────────────────────┘
//              │
//     ┌────────▼──────────────────────────────────┐
//     │     Address Translation Logic              │
//     │  phys_addr = page_table[virt_page] * SIZE  │
//     │            + offset_within_page            │
//     └────────────────────────────────────────────┘
//              │
//     ┌────────▼──────────────────────────────────┐
//     │     Free Page Pool                         │
//     │  Tracks which physical pages are available │
//     └────────────────────────────────────────────┘
//
//   Parameters: PAGE_SIZE_BITS, NUM_VIRTUAL_PAGES, NUM_PHYSICAL_PAGES
// ============================================================================
module paged_attention_mmu #(
    parameter PAGE_SIZE_BITS    = 8,    // log2(page_size), e.g. 256 bytes/page
    parameter NUM_VIRTUAL_PAGES = 32,   // Max virtual pages per sequence  
    parameter NUM_PHYSICAL_PAGES = 64,  // Total physical pages in memory
    parameter VIRT_ADDR_BITS    = 5,    // $clog2(NUM_VIRTUAL_PAGES)
    parameter PHYS_ADDR_BITS    = 6     // $clog2(NUM_PHYSICAL_PAGES)
)(
    input  wire                         clk,
    input  wire                         rst,
    
    // Address translation interface
    input  wire                         translate_valid,
    input  wire [VIRT_ADDR_BITS-1:0]    virtual_page,       // Virtual page number
    input  wire [PAGE_SIZE_BITS-1:0]    page_offset,        // Offset within page
    output reg  [PHYS_ADDR_BITS+PAGE_SIZE_BITS-1:0] physical_addr, // Full physical address
    output reg                          translate_done,
    output reg                          page_fault,         // Virtual page not mapped
    
    // Page allocation interface
    input  wire                         alloc_valid,        // Request a new page
    input  wire [VIRT_ADDR_BITS-1:0]    alloc_virtual_page, // Virtual page to map
    output reg  [PHYS_ADDR_BITS-1:0]    alloc_physical_page,// Allocated physical page
    output reg                          alloc_done,
    output reg                          alloc_fail,         // No free pages
    
    // Page deallocation interface
    input  wire                         free_valid,
    input  wire [VIRT_ADDR_BITS-1:0]    free_virtual_page,
    output reg                          free_done,
    
    // Statistics
    output reg  [31:0]                  total_translations,
    output reg  [31:0]                  total_page_faults,
    output reg  [PHYS_ADDR_BITS:0]      pages_allocated,    // Currently mapped pages
    output reg  [PHYS_ADDR_BITS:0]      pages_free          // Available pages
);

    // Page table: virtual_page → physical_page mapping
    reg [PHYS_ADDR_BITS-1:0] page_table     [0:NUM_VIRTUAL_PAGES-1];
    reg                      page_valid      [0:NUM_VIRTUAL_PAGES-1];
    
    // Free page bitmap: 1 = free, 0 = allocated
    reg free_bitmap [0:NUM_PHYSICAL_PAGES-1];
    
    integer i;
    
    // Find first free page (priority encoder)
    reg [PHYS_ADDR_BITS-1:0] first_free_page;
    reg                      has_free_page;
    
    always @(*) begin
        first_free_page = 0;
        has_free_page = 1'b0;
        for (i = 0; i < NUM_PHYSICAL_PAGES; i = i + 1) begin
            if (free_bitmap[i] && !has_free_page) begin
                first_free_page = i[PHYS_ADDR_BITS-1:0];
                has_free_page = 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            translate_done    <= 1'b0;
            page_fault        <= 1'b0;
            alloc_done        <= 1'b0;
            alloc_fail        <= 1'b0;
            free_done         <= 1'b0;
            physical_addr     <= 0;
            alloc_physical_page <= 0;
            total_translations <= 32'd0;
            total_page_faults  <= 32'd0;
            pages_allocated    <= 0;
            pages_free         <= NUM_PHYSICAL_PAGES;
            
            for (i = 0; i < NUM_VIRTUAL_PAGES; i = i + 1) begin
                page_table[i] <= 0;
                page_valid[i] <= 1'b0;
            end
            for (i = 0; i < NUM_PHYSICAL_PAGES; i = i + 1) begin
                free_bitmap[i] <= 1'b1;  // All pages start free
            end
        end else begin
            translate_done <= 1'b0;
            alloc_done     <= 1'b0;
            alloc_fail     <= 1'b0;
            free_done      <= 1'b0;
            page_fault     <= 1'b0;
            
            // ---- Address Translation ----
            if (translate_valid) begin
                total_translations <= total_translations + 1;
                if (page_valid[virtual_page]) begin
                    // Hit: compute physical address
                    physical_addr <= {page_table[virtual_page], page_offset};
                    translate_done <= 1'b1;
                    page_fault <= 1'b0;
                end else begin
                    // Page fault: virtual page not mapped
                    physical_addr <= 0;
                    translate_done <= 1'b1;
                    page_fault <= 1'b1;
                    total_page_faults <= total_page_faults + 1;
                end
            end
            
            // ---- Allocation/Deallocation arbitration ----
            // Hard fail on concurrent alloc+free to avoid same-cycle alias races.
            if (alloc_valid && free_valid) begin
                if (page_valid[free_virtual_page]) begin
                    free_bitmap[page_table[free_virtual_page]] <= 1'b1;
                    page_valid[free_virtual_page] <= 1'b0;
                    page_table[free_virtual_page] <= {PHYS_ADDR_BITS{1'b0}};
                    pages_allocated <= pages_allocated - 1;
                    pages_free      <= pages_free + 1;
                end
                free_done  <= 1'b1;
                alloc_done <= 1'b1;
                alloc_fail <= 1'b1;
            end else begin
                // ---- Page Allocation ----
                if (alloc_valid) begin
                    if (page_valid[alloc_virtual_page]) begin
                        // Remap policy: if already mapped, move to a fresh page and
                        // release the previous one in the same transaction.
                        if (has_free_page) begin
                            free_bitmap[page_table[alloc_virtual_page]] <= 1'b1;
                            free_bitmap[first_free_page] <= 1'b0;
                            page_table[alloc_virtual_page] <= first_free_page;
                            alloc_physical_page <= first_free_page;
                            alloc_done <= 1'b1;
                            alloc_fail <= 1'b0;
                        end else begin
                            // Preserve existing mapping if remap cannot be satisfied.
                            alloc_physical_page <= page_table[alloc_virtual_page];
                            alloc_done <= 1'b1;
                            alloc_fail <= 1'b1;
                        end
                    end else if (has_free_page) begin
                        // Map new virtual page to first free physical page.
                        page_table[alloc_virtual_page] <= first_free_page;
                        page_valid[alloc_virtual_page] <= 1'b1;
                        free_bitmap[first_free_page]   <= 1'b0;  // Mark as allocated
                        alloc_physical_page <= first_free_page;
                        alloc_done <= 1'b1;
                        alloc_fail <= 1'b0;
                        pages_allocated <= pages_allocated + 1;
                        pages_free      <= pages_free - 1;
                    end else begin
                        // No free pages — Out of Memory
                        alloc_done <= 1'b1;
                        alloc_fail <= 1'b1;
                    end
                end
                
                // ---- Page Deallocation ----
                if (free_valid) begin
                    if (page_valid[free_virtual_page]) begin
                        // Return the physical page to the free pool
                        free_bitmap[page_table[free_virtual_page]] <= 1'b1;
                        page_valid[free_virtual_page] <= 1'b0;
                        page_table[free_virtual_page] <= {PHYS_ADDR_BITS{1'b0}};
                        pages_allocated <= pages_allocated - 1;
                        pages_free      <= pages_free + 1;
                    end
                    free_done <= 1'b1;
                end
            end
        end
    end

endmodule
