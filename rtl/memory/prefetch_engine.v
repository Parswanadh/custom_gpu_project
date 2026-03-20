`timescale 1ns / 1ps

// ============================================================================
// Module: prefetch_engine
// Description: Hardware Weight Prefetch Engine.
//
//   PAPER: "Four Architectural Opportunities for LLM Inference Hardware"
//          (Google Research, Jan 2026)
//
//   RATIONALE: LLM inference is memory-bound. The compute units sit idle
//   while waiting for weights to load from DRAM. A prefetch engine overlaps
//   memory transfers with computation:
//     Without: [LOAD layer1] → [COMPUTE layer1] → [LOAD layer2] → [COMPUTE layer2]
//     With:    [LOAD layer1] → [COMPUTE layer1 + LOAD layer2] → [COMPUTE layer2 + LOAD layer3]
//   This hides memory latency and keeps compute units busy.
//
//   WHY THIS MATTERS FOR BITBYBIT:
//   - Our weight_double_buffer already provides 2 buffers
//   - The prefetch engine AUTOMATES the scheduling: it pre-loads the next
//     layer's weights into the idle buffer while current layer computes
//   - Combined with our DMA engine, this creates a fully pipelined weight
//     loading system with zero stalls
//
//   DESIGN: Ping-pong buffer control with look-ahead scheduling.
//     Buffer A: currently computing (systolic array reads from here)
//     Buffer B: prefetching next layer's weights (DMA writes here)
//     On layer transition: swap A↔B, start prefetching layer+2
//
// Parameters: BUFFER_DEPTH, DATA_WIDTH
// ============================================================================
module prefetch_engine #(
    parameter BUFFER_DEPTH = 64,     // Words per buffer
    parameter DATA_WIDTH   = 32,     // Width per word
    parameter ADDR_WIDTH   = 6,      // log2(BUFFER_DEPTH)
    parameter WAIT_TIMEOUT_CYCLES = 16'd4096
)(
    input  wire                    clk,
    input  wire                    rst,
    
    // Control
    input  wire                    start,           // Begin prefetch pipeline
    input  wire                    layer_done,      // Current layer computation finished
    input  wire [7:0]              total_layers,    // Total layers in model
    
    // DMA request interface (to start background weight loads)
    output reg                     dma_request,
    output reg  [31:0]             dma_src_addr,    // External memory address
    output reg  [15:0]             dma_length,      // Transfer length
    input  wire                    dma_done,        // DMA transfer complete
    
    // Buffer read interface (for compute units)
    input  wire                    buf_read_en,
    input  wire [ADDR_WIDTH-1:0]   buf_read_addr,
    output wire [DATA_WIDTH-1:0]   buf_read_data,
    
    // Buffer write interface (from DMA)
    input  wire                    buf_write_en,
    input  wire [ADDR_WIDTH-1:0]   buf_write_addr,
    input  wire [DATA_WIDTH-1:0]   buf_write_data,
    
    // Status
    output reg                     compute_ready,   // Current buffer ready for computation
    output reg                     prefetch_active,  // Prefetch in progress
    output reg  [7:0]              current_layer,
    output reg  [7:0]              prefetch_layer,
    output reg                     all_done,
    output reg                     error
);

    // Dual buffers (ping-pong)
    reg [DATA_WIDTH-1:0] buffer_a [0:BUFFER_DEPTH-1];
    reg [DATA_WIDTH-1:0] buffer_b [0:BUFFER_DEPTH-1];
    
    reg active_buffer;  // 0 = A is compute, B is prefetch
                        // 1 = B is compute, A is prefetch
    reg dma_done_seen;  // Latch dma_done pulses until state machine consumes them
    
    // FSM
    reg [2:0] state;
    localparam IDLE           = 3'd0;
    localparam PREFETCH_FIRST = 3'd1;  // Load first layer
    localparam COMPUTING      = 3'd2;  // Compute active + prefetch next
    localparam WAIT_PREFETCH  = 3'd3;  // Wait for prefetch to finish before swap
    localparam SWAP           = 3'd4;  // Swap buffers
    localparam DONE_STATE     = 3'd5;
    reg [15:0] wait_counter;
    
    // Layer base address calculation (each layer at layer_num * BUFFER_DEPTH * 4)
    wire [31:0] layer_base_addr = prefetch_layer * (BUFFER_DEPTH * 4);

    always @(posedge clk) begin
        if (rst) begin
            state           <= IDLE;
            active_buffer   <= 1'b0;
            compute_ready   <= 1'b0;
            prefetch_active <= 1'b0;
            current_layer   <= 0;
            prefetch_layer  <= 0;
            dma_request     <= 1'b0;
            all_done        <= 1'b0;
            dma_done_seen   <= 1'b0;
            wait_counter    <= 16'd0;
            error           <= 1'b0;
        end else begin
            dma_request <= 1'b0;  // Single-cycle pulse
            error <= 1'b0;
            
            case (state)
                IDLE: begin
                    all_done <= 1'b0;
                    wait_counter <= 16'd0;
                    if (start) begin
                        current_layer  <= 0;
                        prefetch_layer <= 0;
                        active_buffer  <= 1'b0;
                        compute_ready  <= 1'b0;
                        // Start loading first layer
                        dma_request    <= 1'b1;
                        dma_src_addr   <= 0;
                        dma_length     <= BUFFER_DEPTH * 4;
                        prefetch_active <= 1'b1;
                        dma_done_seen  <= 1'b0;
                        state <= PREFETCH_FIRST;
                    end
                end
                
                // Wait for first layer to load
                PREFETCH_FIRST: begin
                    if (dma_done) begin
                        wait_counter <= 16'd0;
                        compute_ready   <= 1'b1;  // Buffer ready for compute
                        prefetch_active <= 1'b0;
                        
                        // Start prefetching layer 1 into the other buffer
                        if (total_layers > 1) begin
                            prefetch_layer  <= 1;
                            dma_request     <= 1'b1;
                            dma_src_addr    <= BUFFER_DEPTH * 4;
                            dma_length      <= BUFFER_DEPTH * 4;
                            prefetch_active <= 1'b1;
                            dma_done_seen   <= 1'b0;
                        end
                        
                        state <= COMPUTING;
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        error <= 1'b1;
                        compute_ready <= 1'b0;
                        prefetch_active <= 1'b0;
                        all_done <= 1'b1;
                        wait_counter <= 16'd0;
                        state <= IDLE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end
                
                // Main operational state: compute + prefetch overlap
                COMPUTING: begin
                    if (layer_done) begin
                        wait_counter <= 16'd0;
                        current_layer <= current_layer + 1;
                        
                        if (current_layer + 1 >= total_layers) begin
                            // All layers done
                            state <= DONE_STATE;
                        end else if (prefetch_active) begin
                            // Wait for prefetch to finish before we can swap
                            state <= WAIT_PREFETCH;
                        end else begin
                            // Prefetch already done, swap immediately
                            state <= SWAP;
                        end
                    end
                    
                    // Mark prefetch complete when DMA finishes
                    if (dma_done) begin
                        prefetch_active <= 1'b0;
                        dma_done_seen   <= 1'b1;
                        wait_counter    <= 16'd0;
                    end
                end
                
                WAIT_PREFETCH: begin
                    if (dma_done || dma_done_seen || !prefetch_active) begin
                        prefetch_active <= 1'b0;
                        dma_done_seen   <= 1'b0;
                        wait_counter    <= 16'd0;
                        state <= SWAP;
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        error <= 1'b1;
                        prefetch_active <= 1'b0;
                        compute_ready <= 1'b0;
                        all_done <= 1'b1;
                        wait_counter <= 16'd0;
                        state <= IDLE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end
                
                // Swap active buffer
                SWAP: begin
                    active_buffer <= ~active_buffer;
                    compute_ready <= 1'b1;
                    wait_counter <= 16'd0;
                    
                    // Start prefetching next-next layer
                    if (current_layer + 1 < total_layers) begin
                        prefetch_layer  <= current_layer + 1;
                        dma_request     <= 1'b1;
                        dma_src_addr    <= (current_layer + 1) * (BUFFER_DEPTH * 4);
                        dma_length      <= BUFFER_DEPTH * 4;
                        prefetch_active <= 1'b1;
                        dma_done_seen   <= 1'b0;
                    end else begin
                        dma_done_seen   <= 1'b0;
                    end
                    
                    state <= COMPUTING;
                end
                
                DONE_STATE: begin
                    compute_ready <= 1'b0;
                    all_done      <= 1'b1;
                    dma_done_seen <= 1'b0;
                    wait_counter  <= 16'd0;
                    state         <= IDLE;
                end
            endcase
            
            // Buffer write: during first load, write to active buffer
            // After that, write to the non-active (prefetch) buffer
            if (buf_write_en) begin
                if (state == PREFETCH_FIRST) begin
                    // First load: fill the active compute buffer
                    if (active_buffer == 1'b0)
                        buffer_a[buf_write_addr] <= buf_write_data;
                    else
                        buffer_b[buf_write_addr] <= buf_write_data;
                end else begin
                    // Normal operation: write to prefetch (non-active) buffer
                    if (active_buffer == 1'b0)
                        buffer_b[buf_write_addr] <= buf_write_data;
                    else
                        buffer_a[buf_write_addr] <= buf_write_data;
                end
            end
        end
    end
    
    // Buffer read (always from the active buffer = compute buffer)
    assign buf_read_data = (active_buffer == 1'b0) ? buffer_a[buf_read_addr] : buffer_b[buf_read_addr];

endmodule
