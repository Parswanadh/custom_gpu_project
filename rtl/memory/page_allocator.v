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
    reg                     page_is_free [0:NUM_PAGES-1];

    localparam [PAGE_ID_WIDTH:0] COUNT_ZERO = {(PAGE_ID_WIDTH+1){1'b0}};
    localparam [PAGE_ID_WIDTH:0] COUNT_ONE  = {{PAGE_ID_WIDTH{1'b0}}, 1'b1};

    wire alloc_can;
    wire [PAGE_ID_WIDTH-1:0] alloc_candidate;
    wire [PAGE_ID_WIDTH:0] stack_ptr_after_alloc;
    wire free_can;

    assign alloc_can = alloc_req && (stack_ptr > 0);
    assign alloc_candidate = (stack_ptr > 0) ? free_stack[stack_ptr - 1] : {PAGE_ID_WIDTH{1'b0}};
    assign stack_ptr_after_alloc = alloc_can ? (stack_ptr - COUNT_ONE) : stack_ptr;
    assign free_can = free_req &&
                      !page_is_free[free_page_id] &&
                      (stack_ptr_after_alloc < NUM_PAGES);

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            // Initialize: all pages are free
            stack_ptr   <= NUM_PAGES;
            free_count  <= NUM_PAGES;
            alloc_valid <= 1'b0;
            alloc_page_id <= 0;
            for (i = 0; i < NUM_PAGES; i = i + 1) begin
                free_stack[i] <= i[PAGE_ID_WIDTH-1:0];  // Page 0, 1, 2, ...
                page_is_free[i] <= 1'b1;
            end
        end else begin
            alloc_valid <= 1'b0;

            // Deterministic ordering for same-cycle alloc/free:
            // alloc is evaluated first, then free pushes to the post-alloc top.
            if (alloc_can) begin
                alloc_page_id <= alloc_candidate;
                alloc_valid   <= 1'b1;
                page_is_free[alloc_candidate] <= 1'b0;
            end

            // Drop invalid/double-free requests by requiring page_is_free=0.
            if (free_can) begin
                free_stack[stack_ptr_after_alloc] <= free_page_id;
                page_is_free[free_page_id] <= 1'b1;
            end

            stack_ptr  <= stack_ptr_after_alloc + (free_can ? COUNT_ONE : COUNT_ZERO);
            free_count <= free_count
                        + (free_can ? COUNT_ONE : COUNT_ZERO)
                        - (alloc_can ? COUNT_ONE : COUNT_ZERO);
        end
    end

endmodule
