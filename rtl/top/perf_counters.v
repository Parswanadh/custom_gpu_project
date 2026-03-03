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

    reg [COUNTER_W-1:0] counters [0:NUM_COUNTERS-1];

    integer i;

    assign cycle_count    = counters[0];
    assign zero_skip_total = counters[4];
    assign mac_total      = counters[3];

    always @(posedge clk) begin
        if (rst || counter_clear) begin
            for (i = 0; i < NUM_COUNTERS; i = i + 1)
                counters[i] <= {COUNTER_W{1'b0}};
        end else if (counter_enable) begin
            // [0] Cycle count — always increments
            counters[0] <= counters[0] + 1;

            // [1] Active cycles
            if (evt_active)
                counters[1] <= counters[1] + 1;

            // [2] Stall cycles
            if (evt_stall)
                counters[2] <= counters[2] + 1;

            // [3] Total MACs
            counters[3] <= counters[3] + {{(COUNTER_W-16){1'b0}}, evt_macs};

            // [4] Zero-skip count
            counters[4] <= counters[4] + {{(COUNTER_W-16){1'b0}}, evt_zero_skips};

            // [5] Memory reads
            if (evt_mem_read)
                counters[5] <= counters[5] + 1;

            // [6] Memory writes
            if (evt_mem_write)
                counters[6] <= counters[6] + 1;

            // [7] Parity errors
            if (evt_parity_error)
                counters[7] <= counters[7] + 1;
        end
    end

    // Read multiplexer
    always @(*) begin
        read_data = counters[read_idx];
    end

endmodule

