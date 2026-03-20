`timescale 1ns / 1ps

// ============================================================================
// Module: multibank_sram_controller
// Description: Multi-Bank SRAM Controller (AMD X3D V-Cache Inspired).
//
//   REFERENCE: AMD 3D V-Cache (Zen 3D, 2022-2025) — stacks SRAM dies on
//   top of logic die, tripling L3 cache. SK Hynix 3D DRAM stacking.
//   Google "4 Architectural Opportunities for LLM Inference" (Jan 2026).
//
//   RATIONALE: Standard FPGA uses a single SRAM bank → 1 read OR 1 write
//   per cycle → bandwidth = DATA_WIDTH × clock_freq.
//   Multi-bank: N independent banks → N reads AND N writes per cycle →
//   bandwidth = N × DATA_WIDTH × clock_freq (N× improvement).
//
//   This models a 3D-stacked memory hierarchy:
//     Layer 0 (bottom): Logic die (our compute units)
//     Layer 1-N (stacked): Independent SRAM banks
//     Vertical TSVs (Through-Silicon Vias) connect each bank to logic
//
//   WHY FOR BITBYBIT:
//   - Our prefetch_engine uses 1 buffer → 1 read/cycle
//   - With 4 banks, we get 4 reads/cycle → 4× weight throughput
//   - KV cache spread across banks → parallel Q·K for all heads
//   - Combined with GQA + INT4 → fits 32× more context in SRAM
//
// Parameters: NUM_BANKS, BANK_DEPTH, DATA_WIDTH
// ============================================================================
module multibank_sram_controller #(
    parameter NUM_BANKS   = 4,        // Number of independent SRAM banks
    parameter BANK_DEPTH  = 256,      // Words per bank
    parameter DATA_WIDTH  = 32,       // Bits per word
    parameter ADDR_WIDTH  = 8         // log2(BANK_DEPTH)
)(
    input  wire                              clk,
    input  wire                              rst,
    
    // Multi-port read interface (one per bank — all parallel)
    input  wire [NUM_BANKS-1:0]              read_en,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0]   read_addr,
    output reg  [NUM_BANKS*DATA_WIDTH-1:0]   read_data,
    output reg  [NUM_BANKS-1:0]              read_valid,
    
    // Multi-port write interface (one per bank — all parallel)
    input  wire [NUM_BANKS-1:0]              write_en,
    input  wire [NUM_BANKS*ADDR_WIDTH-1:0]   write_addr,
    input  wire [NUM_BANKS*DATA_WIDTH-1:0]   write_data,
    
    // Striped access (auto-distribute across banks)
    input  wire                              stripe_read_en,
    input  wire [ADDR_WIDTH+$clog2(NUM_BANKS)-1:0] stripe_addr,
    output reg  [DATA_WIDTH-1:0]             stripe_read_data,
    output reg                               stripe_read_valid,
    
    // Statistics
    output reg  [31:0]                       total_parallel_reads,
    output reg  [31:0]                       total_parallel_writes,
    output reg  [31:0]                       bank_conflicts   // Should stay 0 with proper striping
);

    // Bank memories — each is independent (models stacked SRAM dies)
    reg [DATA_WIDTH-1:0] bank_mem [0:NUM_BANKS-1][0:BANK_DEPTH-1];
    
    // Striping: addr[1:0] selects bank, addr[ADDR_WIDTH+1:2] selects within bank
    wire [$clog2(NUM_BANKS)-1:0] stripe_bank = stripe_addr[$clog2(NUM_BANKS)-1:0];
    wire [ADDR_WIDTH-1:0] stripe_offset = stripe_addr[ADDR_WIDTH+$clog2(NUM_BANKS)-1:$clog2(NUM_BANKS)];
    
    integer b;
    integer reads_this_cycle, writes_this_cycle;

    always @(posedge clk) begin
        if (rst) begin
            read_data           <= 0;
            read_valid          <= 0;
            stripe_read_data    <= 0;
            stripe_read_valid   <= 1'b0;
            total_parallel_reads  <= 0;
            total_parallel_writes <= 0;
            bank_conflicts      <= 0;
        end else begin
            read_valid        <= 0;
            stripe_read_valid <= 1'b0;
            reads_this_cycle  = 0;
            writes_this_cycle = 0;
            
            // === PARALLEL BANK ACCESS (the 3D V-Cache advantage) ===
            // All banks operate SIMULTANEOUSLY — this is the key innovation
            for (b = 0; b < NUM_BANKS; b = b + 1) begin
                // Parallel writes
                if (write_en[b]) begin
                    bank_mem[b][write_addr[b*ADDR_WIDTH +: ADDR_WIDTH]] <= 
                        write_data[b*DATA_WIDTH +: DATA_WIDTH];
                    writes_this_cycle = writes_this_cycle + 1;
                end
                
                // Parallel reads
                if (read_en[b]) begin
                    read_data[b*DATA_WIDTH +: DATA_WIDTH] <= 
                        bank_mem[b][read_addr[b*ADDR_WIDTH +: ADDR_WIDTH]];
                    read_valid[b] <= 1'b1;
                    reads_this_cycle = reads_this_cycle + 1;
                end
            end
            
            // Striped access (auto-route to correct bank)
            if (stripe_read_en) begin
                stripe_read_data <= bank_mem[stripe_bank][stripe_offset];
                stripe_read_valid <= 1'b1;
            end
            
            // Track statistics
            if (reads_this_cycle > 0)
                total_parallel_reads <= total_parallel_reads + reads_this_cycle;
            if (writes_this_cycle > 0)
                total_parallel_writes <= total_parallel_writes + writes_this_cycle;
        end
    end

endmodule
