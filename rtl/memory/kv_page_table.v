// ============================================================================
// Module: kv_page_table
// Description: Virtual-to-physical address mapping for paged KV cache.
//
//   Borrows virtual memory concepts from PagedAttention (vLLM):
//     - KV cache is stored in non-contiguous "pages" in SRAM
//     - A page table maps logical token positions to physical page IDs
//     - Eliminates memory fragmentation (near-zero waste)
//     - Enables KV sharing across requests
//
//   Each page stores one token's K and V vectors for one head.
//
//   Address computation:
//     physical_addr = page_table[token_id] * PAGE_SIZE + offset_within_page
//
//   Reference: Kwon et al., "Efficient Memory Management for Large Language
//   Model Serving with PagedAttention" (arXiv:2309.06180, SOSP 2023)
// ============================================================================
module kv_page_table #(
    parameter NUM_PAGES     = 64,       // Total physical pages
    parameter PAGE_ID_WIDTH = 6,        // log2(NUM_PAGES)
    parameter MAX_SEQ_LEN   = 64,       // Max logical tokens
    parameter SEQ_ID_WIDTH  = 6         // log2(MAX_SEQ_LEN)
)(
    input  wire                         clk,
    input  wire                         rst,

    // Write port: map logical position → physical page
    input  wire                         write_en,
    input  wire [SEQ_ID_WIDTH-1:0]      write_logical_pos,
    input  wire [PAGE_ID_WIDTH-1:0]     write_page_id,

    // Read port: lookup physical page for a logical position
    input  wire                         read_en,
    input  wire [SEQ_ID_WIDTH-1:0]      read_logical_pos,
    output reg  [PAGE_ID_WIDTH-1:0]     read_page_id,
    output reg                          read_valid,

    // Invalidate: unmap a logical position (for eviction)
    input  wire                         invalidate_en,
    input  wire [SEQ_ID_WIDTH-1:0]      invalidate_pos,

    // Status
    output reg  [SEQ_ID_WIDTH:0]        active_entries   // Count of mapped entries
);

    // Page table storage
    reg [PAGE_ID_WIDTH-1:0] page_table [0:MAX_SEQ_LEN-1];
    reg                     entry_valid [0:MAX_SEQ_LEN-1];

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            active_entries <= 0;
            read_valid     <= 1'b0;
            read_page_id   <= 0;
            for (i = 0; i < MAX_SEQ_LEN; i = i + 1) begin
                page_table[i]  <= 0;
                entry_valid[i] <= 1'b0;
            end
        end else begin
            read_valid <= 1'b0;

            // Write: map logical → physical
            if (write_en) begin
                page_table[write_logical_pos]  <= write_page_id;
                if (!entry_valid[write_logical_pos])
                    active_entries <= active_entries + 1;
                entry_valid[write_logical_pos] <= 1'b1;
            end

            // Read: lookup
            if (read_en) begin
                if (entry_valid[read_logical_pos]) begin
                    read_page_id <= page_table[read_logical_pos];
                    read_valid   <= 1'b1;
                end else begin
                    read_page_id <= 0;
                    read_valid   <= 1'b0;  // Unmapped position
                end
            end

            // Invalidate: unmap position
            if (invalidate_en) begin
                if (entry_valid[invalidate_pos]) begin
                    entry_valid[invalidate_pos] <= 1'b0;
                    active_entries <= active_entries - 1;
                end
            end
        end
    end

endmodule
