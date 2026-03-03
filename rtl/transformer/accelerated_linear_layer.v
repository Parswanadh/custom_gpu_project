// ============================================================================
// Module: accelerated_linear_layer
// Description: Linear layer (y = x * W + b) accelerated by gpu_core.
//   FIXES: Updated for new gpu_core interface (signed, acc_clear, per-lane act,
//          stall, parity) — Issues #1, #2, #4, #11
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
    input  wire [IN_DIM*DATA_WIDTH-1:0]      x_in,

    // Weight loading
    input  wire                              load_weight,
    input  wire [ADDR_W-1:0]                 load_addr,
    input  wire signed [7:0]                 load_data,

    // Bias
    input  wire [OUT_DIM*DATA_WIDTH-1:0]     bias_in,

    // Output
    output reg  [OUT_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                               valid_out,
    output reg  [31:0]                       total_zero_skips
);

    // State machine
    reg [3:0] state;
    localparam IDLE      = 4'd0;
    localparam COMPUTE   = 4'd1;
    localparam DRAIN     = 4'd2;
    localparam BIAS_ADD  = 4'd3;
    localparam OUTPUT    = 4'd4;

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf [0:IN_DIM-1];
    reg signed [DATA_WIDTH-1:0] result [0:OUT_DIM-1];
    reg [$clog2(OUT_DIM):0] out_idx;
    reg [$clog2(IN_DIM):0]  in_idx;
    reg [3:0] drain_cnt;

    // GPU core interface — updated for new interface
    reg         core_valid;
    reg [ADDR_W-1:0] core_addr;
    reg [8*LANES-1:0] core_activation;  // Per-lane activation vector
    reg         core_acc_clear;
    wire        core_valid_out;
    wire signed [31:0] core_accumulator;
    wire [LANES-1:0] core_zero_mask;
    wire        core_ready;
    wire        core_parity_error;

    gpu_core #(
        .LANES(LANES),
        .MEM_DEPTH(MEM_DEPTH),
        .ADDR_W(ADDR_W)
    ) compute_core (
        .clk(clk),
        .rst(rst),
        .dq_scale(4'd1),
        .dq_offset(4'd0),
        .core_id(4'd0),
        .mem_write_en(load_weight),
        .mem_write_val(load_data),
        .mem_write_idx(load_addr),
        .valid_in(core_valid),
        .weight_base_addr(core_addr),
        .activation_in(core_activation),
        .acc_clear(core_acc_clear),
        .downstream_ready(1'b1),
        .ready(core_ready),
        .valid_out(core_valid_out),
        .zero_skip_mask(core_zero_mask),
        .accumulator(core_accumulator),
        .lane_results(),
        .pipe_active(),
        .products_per_cycle(),
        .parity_error(core_parity_error)
    );

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            valid_out  <= 1'b0;
            core_valid <= 1'b0;
            core_acc_clear <= 1'b0;
            out_idx    <= 0;
            in_idx     <= 0;
            drain_cnt  <= 0;
            y_out      <= 0;
            total_zero_skips <= 0;
            core_activation <= 0;
        end else begin
            core_valid <= 1'b0;
            core_acc_clear <= 1'b0;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < IN_DIM; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        out_idx <= 0;
                        in_idx  <= 0;
                        core_acc_clear <= 1'b1;
                        state   <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (in_idx < IN_DIM && core_ready) begin
                        core_valid <= 1'b1;
                        core_addr  <= out_idx * IN_DIM + in_idx;
                        // Pack activation into per-lane vector (broadcast same value)
                        begin : pack_act
                            integer li;
                            for (li = 0; li < LANES; li = li + 1)
                                core_activation[li*8 +: 8] <= x_buf[in_idx][7:0];
                        end
                        in_idx <= in_idx + LANES;
                    end else if (in_idx >= IN_DIM) begin
                        drain_cnt <= 0;
                        state <= DRAIN;
                    end
                end

                DRAIN: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt >= 4'd6) begin
                        result[out_idx] <= core_accumulator[DATA_WIDTH-1:0];
                        // Count zero skips
                        if (core_valid_out) begin
                            begin : count_zs
                                integer zi;
                                for (zi = 0; zi < LANES; zi = zi + 1)
                                    if (core_zero_mask[zi])
                                        total_zero_skips <= total_zero_skips + 1;
                            end
                        end

                        out_idx <= out_idx + 1;
                        in_idx  <= 0;
                        core_acc_clear <= 1'b1;
                        if (out_idx + 1 >= OUT_DIM)
                            state <= BIAS_ADD;
                        else
                            state <= COMPUTE;
                    end
                end

                BIAS_ADD: begin
                    for (i = 0; i < OUT_DIM; i = i + 1)
                        y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                            result[i] + $signed(bias_in[i*DATA_WIDTH +: DATA_WIDTH]);
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
