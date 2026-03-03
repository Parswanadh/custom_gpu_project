// ============================================================================
// Module: page_allocator
// Description: Free-list based physical page allocator for KV cache.
//
//   Maintains a stack of free page IDs.
//   - alloc: Pop a free page for a new token's KV data
//   - free:  Push a page back when its token is evicted
//
//   Supports sliding window eviction (StreamingLLM):
//     When sequence exceeds WINDOW_SIZE + SINK_SIZE, evict oldest
//     non-sink token and reuse its page.
//
//   Reference: Xiao et al., "Efficient Streaming Language Models with
//   Attention Sinks" (arXiv:2309.17453, ICLR 2024)
// ============================================================================
module page_allocator #(
    parameter NUM_PAGES     = 64,
    parameter PAGE_ID_WIDTH = 6         // log2(NUM_PAGES)
)(
    input  wire                         clk,
    input  wire                         rst,

    // Allocate: request a free page
    input  wire                         alloc_req,
    output reg  [PAGE_ID_WIDTH-1:0]     alloc_page_id,
    output reg                          alloc_valid,      // 1 = page available, 0 = out of pages

    // Free: return a page to the free pool
    input  wire                         free_req,
    input  wire [PAGE_ID_WIDTH-1:0]     free_page_id,

    // Status
    output reg  [PAGE_ID_WIDTH:0]       free_count        // How many pages are free
);

    // Free page stack
    reg [PAGE_ID_WIDTH-1:0] free_stack [0:NUM_PAGES-1];
    reg [PAGE_ID_WIDTH:0]   stack_ptr;   // Points to top of stack (next free slot)

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            // Initialize: all pages are free
            stack_ptr   <= NUM_PAGES;
            free_count  <= NUM_PAGES;
            alloc_valid <= 1'b0;
            alloc_page_id <= 0;
            for (i = 0; i < NUM_PAGES; i = i + 1)
                free_stack[i] <= i[PAGE_ID_WIDTH-1:0];  // Page 0, 1, 2, ...
        end else begin
            alloc_valid <= 1'b0;

            // Allocate: pop from stack
            if (alloc_req && stack_ptr > 0) begin
                stack_ptr     <= stack_ptr - 1;
                alloc_page_id <= free_stack[stack_ptr - 1];
                alloc_valid   <= 1'b1;
                free_count    <= free_count - 1;
            end

            // Free: push back to stack
            if (free_req && stack_ptr < NUM_PAGES) begin
                free_stack[stack_ptr] <= free_page_id;
                stack_ptr  <= stack_ptr + 1;
                free_count <= free_count + 1;
            end
        end
    end

endmodule
