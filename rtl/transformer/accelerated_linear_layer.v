// ============================================================================
// Module: accelerated_linear_layer
// Description: Linear layer (y = x * W + b) accelerated by gpu_core.
//   Uses the parameterized N-lane pipelined core for matrix-vector multiply.
//   This bridges the gap between the transformer engine and the optimized
//   compute pipeline.
//
// How it works:
//   1. Weights are pre-loaded into gpu_core's memory
//   2. For each output element: feed LANES weights at a time
//   3. gpu_core accumulates the dot product across multiple cycles
//   4. Bias is added after accumulation
//
// Parameters: IN_DIM, OUT_DIM, LANES (compute parallelism)
// ============================================================================
module accelerated_linear_layer #(
    parameter IN_DIM    = 8,
    parameter OUT_DIM   = 8,
    parameter LANES     = 4,
    parameter DATA_WIDTH = 16,
    parameter MEM_DEPTH = 256,
    parameter ADDR_W    = 8
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              valid_in,
    input  wire [IN_DIM*DATA_WIDTH-1:0]      x_in,       // Input vector (packed)

    // Weight loading (must be done before inference)
    input  wire                              load_weight,
    input  wire [ADDR_W-1:0]                 load_addr,
    input  wire [7:0]                        load_data,

    // Bias (packed)
    input  wire [OUT_DIM*DATA_WIDTH-1:0]     bias_in,

    // Output
    output reg  [OUT_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                               valid_out,
    output reg  [31:0]                       total_zero_skips
);

    // State machine
    reg [3:0] state;
    localparam IDLE      = 4'd0;
    localparam LOAD_ACT  = 4'd1;
    localparam COMPUTE   = 4'd2;
    localparam BIAS_ADD  = 4'd3;
    localparam OUTPUT    = 4'd4;

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf [0:IN_DIM-1];
    reg signed [DATA_WIDTH-1:0] result [0:OUT_DIM-1];
    reg [$clog2(OUT_DIM):0] out_idx;     // Current output element
    reg [$clog2(IN_DIM):0]  in_idx;      // Current input offset within dot product

    // GPU core interface
    reg         core_valid;
    reg [ADDR_W-1:0] core_addr;
    reg [7:0]   core_activation;
    wire        core_valid_out;
    wire [31:0] core_accumulator;
    wire [LANES-1:0] core_zero_mask;

    // Instantiate the pipelined GPU core
    gpu_core #(
        .LANES(LANES),
        .MEM_DEPTH(MEM_DEPTH),
        .ADDR_W(ADDR_W)
    ) compute_core (
        .clk(clk),
        .rst(rst),
        .dq_scale(4'd1),          // Scale = 1 (already quantized)
        .dq_offset(4'd0),         // No offset
        .core_id(4'd0),
        .mem_write_en(load_weight),
        .mem_write_val(load_data),
        .mem_write_idx(load_addr),
        .valid_in(core_valid),
        .weight_base_addr(core_addr),
        .activation_in(core_activation),
        .valid_out(core_valid_out),
        .zero_skip_mask(core_zero_mask),
        .accumulator(core_accumulator),
        .lane_results(),
        .pipe_active(),
        .products_per_cycle()
    );

    integer i;
    integer steps_per_dot;

    always @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            valid_out  <= 1'b0;
            core_valid <= 1'b0;
            out_idx    <= 0;
            in_idx     <= 0;
            y_out      <= 0;
            total_zero_skips <= 0;
        end else begin
            core_valid <= 1'b0;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Unpack input vector
                        for (i = 0; i < IN_DIM; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        out_idx <= 0;
                        in_idx  <= 0;
                        state   <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Feed activation values to gpu_core, LANES at a time
                    // Weight address = out_idx * IN_DIM + in_idx
                    if (in_idx < IN_DIM) begin
                        core_valid     <= 1'b1;
                        core_addr      <= out_idx * IN_DIM + in_idx;
                        core_activation <= x_buf[in_idx][7:0];  // Lower 8 bits
                        in_idx         <= in_idx + LANES;
                    end else begin
                        // Dot product for this output element complete
                        // Wait for pipeline to drain
                        if (core_valid_out) begin
                            result[out_idx] <= core_accumulator[DATA_WIDTH-1:0];
                            out_idx <= out_idx + 1;
                            in_idx  <= 0;
                            if (out_idx + 1 >= OUT_DIM)
                                state <= BIAS_ADD;
                        end
                    end

                    // Count zero skips
                    if (core_valid_out) begin
                        for (i = 0; i < LANES; i = i + 1)
                            if (core_zero_mask[i])
                                total_zero_skips <= total_zero_skips + 1;
                    end
                end

                BIAS_ADD: begin
                    // Add bias to each output element
                    for (i = 0; i < OUT_DIM; i = i + 1) begin
                        y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                            result[i] + bias_in[i*DATA_WIDTH +: DATA_WIDTH];
                    end
                    state <= OUTPUT;
                end

                OUTPUT: begin
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end
            endcase
        end
    end

endmodule
