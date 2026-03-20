// ============================================================================
// Module: perf_counters
// Description: Performance Monitoring Unit (PMU) for GPU debugging (Issue #23).
//   Tracks cycle counts, operation counts, stall events, and zero-skip rates.
//
//   Counters (each 32-bit):
//     [0] CYCLE_COUNT      — Total clock cycles since last reset
//     [1] ACTIVE_CYCLES    — Cycles where compute pipeline is active
//     [2] STALL_CYCLES     — Cycles where pipeline is stalled
//     [3] TOTAL_MACS       — Total multiply-accumulate operations
//     [4] ZERO_SKIP_COUNT  — MACs skipped due to zero detection
//     [5] MEMORY_READS     — Total memory read operations
//     [6] MEMORY_WRITES    — Total memory write operations
//     [7] PARITY_ERRORS    — Total parity errors detected
//
//   Control:
//     counter_enable — Global enable for all counters
//     counter_clear  — Synchronous clear of all counters
//
//   Access: Read via index (0-7) for AXI register exposure
// ============================================================================
`timescale 1ns / 1ps

module perf_counters #(
    parameter NUM_COUNTERS = 8,
    parameter COUNTER_W    = 32
)(
    input  wire                    clk,
    input  wire                    rst,

    // Control
    input  wire                    counter_enable,
    input  wire                    counter_clear,

    // Event inputs (active-high pulses)
    input  wire                    evt_active,         // Pipeline active
    input  wire                    evt_stall,          // Pipeline stalled
    input  wire [15:0]             evt_macs,           // MACs this cycle
    input  wire [15:0]             evt_zero_skips,     // Zero-skips this cycle
    input  wire                    evt_mem_read,       // Memory read
    input  wire                    evt_mem_write,      // Memory write
    input  wire                    evt_parity_error,   // Parity error

    // Read interface
    input  wire [$clog2(NUM_COUNTERS)-1:0] read_idx,
    output reg  [COUNTER_W-1:0]            read_data,

    // Direct counter outputs for high-priority monitoring
    output wire [COUNTER_W-1:0]    cycle_count,
    output wire [COUNTER_W-1:0]    zero_skip_total,
    output wire [COUNTER_W-1:0]    mac_total
);

    reg [COUNTER_W-1:0] counter_cycle;
    reg [COUNTER_W-1:0] counter_active;
    reg [COUNTER_W-1:0] counter_stall;
    reg [COUNTER_W-1:0] counter_macs;
    reg [COUNTER_W-1:0] counter_zero_skips;
    reg [COUNTER_W-1:0] counter_mem_reads;
    reg [COUNTER_W-1:0] counter_mem_writes;
    reg [COUNTER_W-1:0] counter_parity_errors;

    assign cycle_count     = counter_cycle;
    assign zero_skip_total = counter_zero_skips;
    assign mac_total       = counter_macs;

    always @(posedge clk) begin
        if (rst || counter_clear) begin
            counter_cycle         <= {COUNTER_W{1'b0}};
            counter_active        <= {COUNTER_W{1'b0}};
            counter_stall         <= {COUNTER_W{1'b0}};
            counter_macs          <= {COUNTER_W{1'b0}};
            counter_zero_skips    <= {COUNTER_W{1'b0}};
            counter_mem_reads     <= {COUNTER_W{1'b0}};
            counter_mem_writes    <= {COUNTER_W{1'b0}};
            counter_parity_errors <= {COUNTER_W{1'b0}};
        end else if (counter_enable) begin
            // [0] Cycle count — always increments
            counter_cycle <= counter_cycle + 1'b1;

            // [1] Active cycles
            if (evt_active)
                counter_active <= counter_active + 1'b1;

            // [2] Stall cycles
            if (evt_stall)
                counter_stall <= counter_stall + 1'b1;

            // [3] Total MACs
            counter_macs <= counter_macs + {{(COUNTER_W-16){1'b0}}, evt_macs};

            // [4] Zero-skip count
            counter_zero_skips <= counter_zero_skips + {{(COUNTER_W-16){1'b0}}, evt_zero_skips};

            // [5] Memory reads
            if (evt_mem_read)
                counter_mem_reads <= counter_mem_reads + 1'b1;

            // [6] Memory writes
            if (evt_mem_write)
                counter_mem_writes <= counter_mem_writes + 1'b1;

            // [7] Parity errors
            if (evt_parity_error)
                counter_parity_errors <= counter_parity_errors + 1'b1;
        end
    end

    // Read multiplexer
    always @(*) begin
        case (read_idx)
            3'd0: read_data = counter_cycle;
            3'd1: read_data = counter_active;
            3'd2: read_data = counter_stall;
            3'd3: read_data = counter_macs;
            3'd4: read_data = counter_zero_skips;
            3'd5: read_data = counter_mem_reads;
            3'd6: read_data = counter_mem_writes;
            3'd7: read_data = counter_parity_errors;
            default: read_data = {COUNTER_W{1'b0}};
        endcase
    end

endmodule

