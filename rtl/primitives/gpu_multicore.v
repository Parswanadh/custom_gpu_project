// ============================================================================
// Module: gpu_multicore
// Description: Multi-core GPU top-level with configurable core count and lanes.
//   Instantiates NUM_CORES gpu_core instances, each with LANES_PER_CORE
//   parallel compute lanes. A round-robin scheduler distributes work.
//
// Total throughput: NUM_CORES × LANES_PER_CORE products per cycle
//
// Example configurations:
//   NUM_CORES=1, LANES=4:    4 products/cycle (current baseline)
//   NUM_CORES=4, LANES=32: 128 products/cycle (32× improvement)
//   NUM_CORES=8, LANES=128: 1024 products/cycle (256× improvement)
// ============================================================================
module gpu_multicore #(
    parameter NUM_CORES      = 4,
    parameter LANES_PER_CORE = 32,
    parameter MEM_DEPTH      = 256,
    parameter ADDR_W         = 8
)(
    input  wire                    clk,
    input  wire                    rst,

    // Configuration (shared across all cores)
    input  wire [3:0]              dq_scale,
    input  wire [3:0]              dq_offset,

    // Weight loading interface
    input  wire                    mem_write_en,
    input  wire [7:0]              mem_write_val,
    input  wire [ADDR_W-1:0]       mem_write_idx,
    input  wire [3:0]              mem_write_core,    // Target core for write

    // Work dispatch — feeds all cores
    input  wire                    valid_in,
    input  wire [ADDR_W-1:0]       weight_base_addr,
    input  wire [7:0]              activation_in,

    // Round-robin scheduling
    input  wire                    schedule_mode,      // 0=broadcast, 1=round-robin

    // Aggregated output
    output wire                    any_valid_out,
    output reg  [31:0]             total_accumulator,
    output reg  [31:0]             total_products_out,
    output reg  [31:0]             total_zero_skips
);

    // ========================================================================
    // Core Instance Signals
    // ========================================================================
    wire                    core_valid_out  [0:NUM_CORES-1];
    wire [LANES_PER_CORE-1:0] core_zero_mask [0:NUM_CORES-1];
    wire [31:0]             core_accumulator [0:NUM_CORES-1];
    wire [4:0]              core_pipe_active [0:NUM_CORES-1];

    // Round-robin counter
    reg [3:0] rr_counter;
    always @(posedge clk) begin
        if (rst)
            rr_counter <= 4'd0;
        else if (valid_in)
            rr_counter <= (rr_counter == NUM_CORES - 1) ? 4'd0 : rr_counter + 1;
    end

    // ========================================================================
    // Core Instantiation (generate NUM_CORES instances)
    // ========================================================================
    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c = c + 1) begin : core_gen

            // Per-core valid: broadcast or round-robin
            wire core_valid = schedule_mode ?
                (valid_in && (rr_counter == c)) :   // Round-robin
                valid_in;                            // Broadcast

            // Per-core memory write
            wire core_mem_wr = mem_write_en && (mem_write_core == c);

            gpu_core #(
                .LANES(LANES_PER_CORE),
                .MEM_DEPTH(MEM_DEPTH),
                .ADDR_W(ADDR_W)
            ) core_inst (
                .clk(clk),
                .rst(rst),
                .dq_scale(dq_scale),
                .dq_offset(dq_offset),
                .core_id(c[3:0]),
                .mem_write_en(core_mem_wr),
                .mem_write_val(mem_write_val),
                .mem_write_idx(mem_write_idx),
                .valid_in(core_valid),
                .weight_base_addr(weight_base_addr),
                .activation_in(activation_in),
                .valid_out(core_valid_out[c]),
                .zero_skip_mask(core_zero_mask[c]),
                .accumulator(core_accumulator[c]),
                .lane_results(),  // Not aggregated for simplicity
                .pipe_active(core_pipe_active[c]),
                .products_per_cycle()
            );
        end
    endgenerate

    // ========================================================================
    // Output Aggregation
    // ========================================================================
    // Any core producing output?
    reg any_valid_reg;
    integer ai;
    always @(*) begin
        any_valid_reg = 1'b0;
        for (ai = 0; ai < NUM_CORES; ai = ai + 1)
            any_valid_reg = any_valid_reg | core_valid_out[ai];
    end
    assign any_valid_out = any_valid_reg;

    // Combinational: sum all core accumulators (no race condition)
    reg [31:0] accum_sum;
    integer si;
    always @(*) begin
        accum_sum = 32'd0;
        for (si = 0; si < NUM_CORES; si = si + 1)
            accum_sum = accum_sum + core_accumulator[si];  // blocking = correct!
    end

    // Combinational: count products and zero-skips this cycle
    reg [31:0] cycle_products;
    reg [31:0] cycle_zero_skips;
    integer ci, zi;
    always @(*) begin
        cycle_products = 32'd0;
        cycle_zero_skips = 32'd0;
        for (ci = 0; ci < NUM_CORES; ci = ci + 1) begin
            if (core_valid_out[ci]) begin
                cycle_products = cycle_products + LANES_PER_CORE;
                for (zi = 0; zi < LANES_PER_CORE; zi = zi + 1)
                    if (core_zero_mask[ci][zi])
                        cycle_zero_skips = cycle_zero_skips + 1;
            end
        end
    end

    // Sequential: register the aggregated values
    always @(posedge clk) begin
        if (rst) begin
            total_accumulator  <= 32'd0;
            total_products_out <= 32'd0;
            total_zero_skips   <= 32'd0;
        end else begin
            total_accumulator  <= accum_sum;
            if (any_valid_reg) begin
                total_products_out <= total_products_out + cycle_products;
                total_zero_skips   <= total_zero_skips + cycle_zero_skips;
            end
        end
    end

endmodule
