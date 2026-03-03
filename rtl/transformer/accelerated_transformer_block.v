// ============================================================================
// Module: accelerated_transformer_block
// Description: Complete transformer block using GPU CORE pipeline.
//   Architecture: Pre-LayerNorm Transformer
//     x → LN1 → Attention(+KV cache) → + residual → LN2 → FFN(gpu_core) → + residual → out
//
//   KEY IMPROVEMENT: The FFN now uses gpu_core instances for matrix multiply,
//   meaning the pipelined N-lane hardware is ACTUALLY USED during inference.
//   This was previously an inline for-loop (fake pipeline usage).
//
//   FFN uses two pipeline stages:
//     Layer 1: hidden[j] = ReLU( sum_i(LN2_out[i] * W1[i][j]) + b1[j] )
//     Layer 2: out[j]    = sum_i(hidden[i] * W2[i][j]) + b2[j]
//   Both layers exploit zero-skipping via gpu_core's sparsity detection.
// ============================================================================
module accelerated_transformer_block #(
    parameter EMBED_DIM   = 8,
    parameter NUM_HEADS   = 2,
    parameter HEAD_DIM    = 4,
    parameter FFN_DIM     = 16,
    parameter MAX_SEQ_LEN = 32,
    parameter DATA_WIDTH  = 16,
    parameter NUM_LANES   = 4     // gpu_core lane count
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]      seq_pos,

    // LayerNorm params
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     ln1_gamma, ln1_beta,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     ln2_gamma, ln2_beta,

    // Attention weights
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat, wk_flat, wv_flat, wo_flat,

    // FFN weights + biases
    input  wire [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0]   ffn_w1_flat,
    input  wire [FFN_DIM*DATA_WIDTH-1:0]              ffn_b1_flat,
    input  wire [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0]    ffn_w2_flat,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]            ffn_b2_flat,

    // Output
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out,
    output reg  [31:0]                         block_zero_skips
);

    // Internal signals
    wire [EMBED_DIM*DATA_WIDTH-1:0] ln1_out, ln2_out;
    wire                            ln1_valid, ln2_valid;
    wire [EMBED_DIM*DATA_WIDTH-1:0] attn_out;
    wire                            attn_valid;
    wire [31:0]                     attn_zero_skips;

    reg  [EMBED_DIM*DATA_WIDTH-1:0] residual1;
    reg  [EMBED_DIM*DATA_WIDTH-1:0] after_attn;

    // GPU core for FFN — instantiated for pipelined MAC
    // Updated for new gpu_core interface (Issues #1, #2, #4, #11, #16)
    reg         core_rst_r, core_we, core_feed_valid;
    reg  [7:0]  core_w_addr;
    reg signed [7:0]  core_w_data;
    reg  [8*NUM_LANES-1:0] core_activation_vec;
    reg         core_acc_clear;
    wire        core_valid_out;
    wire signed [31:0] core_accumulator;
    wire [NUM_LANES-1:0] core_zero_mask;
    wire [4:0]  core_pipe_active;
    wire [31:0] core_products_per_cycle;
    wire [16*NUM_LANES-1:0] core_lane_results;
    wire        core_ready;
    wire        core_parity_error;

    gpu_core #(.LANES(NUM_LANES), .MEM_DEPTH(256)) u_gpu_core (
        .clk(clk), .rst(core_rst_r),
        .dq_scale(4'd1), .dq_offset(4'd0),
        .core_id(4'd0),
        .mem_write_en(core_we), .mem_write_val(core_w_data), .mem_write_idx(core_w_addr),
        .valid_in(core_feed_valid), .weight_base_addr(8'd0),
        .activation_in(core_activation_vec),
        .acc_clear(core_acc_clear),
        .downstream_ready(1'b1),
        .ready(core_ready),
        .valid_out(core_valid_out), .zero_skip_mask(core_zero_mask),
        .accumulator(core_accumulator), .lane_results(core_lane_results),
        .pipe_active(core_pipe_active), .products_per_cycle(core_products_per_cycle),
        .parity_error(core_parity_error)
    );

    // State machine
    reg [4:0] state;
    localparam IDLE         = 5'd0;
    localparam LN1          = 5'd1;
    localparam ATTN         = 5'd2;
    localparam RESID1       = 5'd3;
    localparam LN2          = 5'd4;
    localparam FFN1_LOAD    = 5'd5;
    localparam FFN1_COMPUTE = 5'd6;
    localparam FFN1_DRAIN   = 5'd7;
    localparam FFN1_ACCUM   = 5'd8;
    localparam FFN2_LOAD    = 5'd9;
    localparam FFN2_COMPUTE = 5'd10;
    localparam FFN2_DRAIN   = 5'd11;
    localparam FFN2_ACCUM   = 5'd12;
    localparam RESID2       = 5'd13;
    localparam DONE         = 5'd14;

    reg ln1_en, ln2_en, attn_en;
    integer i;

    // FFN working storage
    reg signed [DATA_WIDTH-1:0] ln2_buf [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] ffn_hidden [0:FFN_DIM-1];
    reg signed [DATA_WIDTH-1:0] ffn_out_buf [0:EMBED_DIM-1];

    reg [$clog2(FFN_DIM):0] ffn_col;
    reg [7:0] ffn_row;
    reg [31:0] ffn_accum;
    reg [3:0]  drain_cnt;
    integer fi;

    // Sub-module: Layer Norm 1
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln1 (
        .clk(clk), .rst(rst), .valid_in(ln1_en),
        .x_in(x_in),
        .gamma_in(ln1_gamma), .beta_in(ln1_beta),
        .y_out(ln1_out), .valid_out(ln1_valid)
    );

    // Sub-module: Accelerated Attention (with KV cache!)
    accelerated_attention #(
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_attn (
        .clk(clk), .rst(rst), .valid_in(attn_en),
        .x_in(ln1_out), .seq_pos(seq_pos),
        .wq_flat(wq_flat), .wk_flat(wk_flat),
        .wv_flat(wv_flat), .wo_flat(wo_flat),
        .y_out(attn_out), .valid_out(attn_valid),
        .zero_skip_count(attn_zero_skips)
    );

    // Sub-module: Layer Norm 2
    layer_norm #(.DIM(EMBED_DIM), .DATA_WIDTH(DATA_WIDTH)) u_ln2 (
        .clk(clk), .rst(rst), .valid_in(ln2_en),
        .x_in(after_attn),
        .gamma_in(ln2_gamma), .beta_in(ln2_beta),
        .y_out(ln2_out), .valid_out(ln2_valid)
    );

    always @(posedge clk) begin
        if (rst) begin
            state            <= IDLE;
            valid_out        <= 1'b0;
            y_out            <= 0;
            ln1_en           <= 0;
            ln2_en           <= 0;
            attn_en          <= 0;
            block_zero_skips <= 0;
            core_rst_r         <= 1;
            core_we          <= 0;
            core_feed_valid  <= 0;
            core_acc_clear   <= 0;
            core_activation_vec <= 0;
            ffn_col          <= 0;
            ffn_row          <= 0;
        end else begin
            ln1_en         <= 0;
            ln2_en         <= 0;
            attn_en        <= 0;
            core_we        <= 0;
            core_feed_valid <= 0;
            core_acc_clear  <= 0;

            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        residual1 <= x_in;
                        ln1_en    <= 1'b1;
                        state     <= LN1;
                    end
                end

                LN1: begin
                    if (ln1_valid) begin
                        attn_en <= 1'b1;
                        state   <= ATTN;
                    end
                end

                ATTN: begin
                    if (attn_valid) begin
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            after_attn[i*DATA_WIDTH +: DATA_WIDTH] <=
                                $signed(residual1[i*DATA_WIDTH +: DATA_WIDTH]) +
                                $signed(attn_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        block_zero_skips <= attn_zero_skips;
                        state <= RESID1;
                    end
                end

                RESID1: begin
                    ln2_en <= 1'b1;
                    state  <= LN2;
                end

                LN2: begin
                    if (ln2_valid) begin
                        // Buffer LN2 output for FFN
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            ln2_buf[i] <= $signed(ln2_out[i*DATA_WIDTH +: DATA_WIDTH]);
                        ffn_col  <= 0;
                        core_acc_clear <= 1;  // Clear accumulator for FFN use
                        state    <= FFN1_LOAD;
                    end
                end

                // ============================================================
                // FFN Layer 1: hidden[j] = ReLU(sum_i(x[i] * W1[i][j]) + b1[j])
                // Uses gpu_core pipeline for the matrix multiply!
                // ============================================================
                FFN1_LOAD: begin
                    core_rst_r <= 0;
                    // Load weights for current column into gpu_core's weight memory
                    if (ffn_row < EMBED_DIM) begin
                        core_we     <= 1;
                        core_w_addr <= ffn_row[7:0];
                        // Extract weight W1[ffn_row][ffn_col] — truncate to 8-bit for gpu_core
                        core_w_data <= ffn_w1_flat[(ffn_row*FFN_DIM+ffn_col)*DATA_WIDTH +: 8];
                        ffn_row     <= ffn_row + 1;
                    end else begin
                        ffn_row <= 0;
                        state   <= FFN1_COMPUTE;
                    end
                end

                FFN1_COMPUTE: begin
                    if (ffn_row < EMBED_DIM) begin
                        core_feed_valid <= 1;
                        // Broadcast activation to all lanes
                        begin : ffn1_pack
                            integer pi;
                            for (pi = 0; pi < NUM_LANES; pi = pi + 1)
                                core_activation_vec[pi*8 +: 8] <= ln2_buf[ffn_row][7:0];
                        end
                        ffn_row         <= ffn_row + 1;
                    end else begin
                        drain_cnt <= 0;
                        state <= FFN1_DRAIN;
                    end
                end

                FFN1_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt >= 4'd6)
                        state <= FFN1_ACCUM;
                end

                FFN1_ACCUM: begin
                    // Read gpu_core accumulator — this is the dot product result
                    // Add bias and apply ReLU
                    begin : ffn1_bias_relu
                        reg signed [31:0] val;
                        val = $signed(core_accumulator) +
                              $signed(ffn_b1_flat[ffn_col*DATA_WIDTH +: DATA_WIDTH]);
                        if (val < 0) begin
                            ffn_hidden[ffn_col] = 0;  // ReLU — creates sparsity!
                            block_zero_skips <= block_zero_skips + 1;
                        end else
                            ffn_hidden[ffn_col] = val[DATA_WIDTH-1:0];
                    end

                    // Count zero-skips from gpu_core
                    for (fi = 0; fi < NUM_LANES; fi = fi + 1)
                        if (core_zero_mask[fi])
                            block_zero_skips <= block_zero_skips + 1;

                    // Move to next output column or to FFN Layer 2
                    if (ffn_col + 1 < FFN_DIM) begin
                        ffn_col  <= ffn_col + 1;
                        ffn_row  <= 0;
                        core_acc_clear <= 1;
                        state    <= FFN1_LOAD;
                    end else begin
                        ffn_col  <= 0;
                        ffn_row  <= 0;
                        core_acc_clear <= 1;
                        state    <= FFN2_LOAD;
                    end
                end

                // ============================================================
                // FFN Layer 2: out[j] = sum_i(hidden[i] * W2[i][j]) + b2[j]
                // Also uses gpu_core pipeline!
                // ============================================================
                FFN2_LOAD: begin
                    core_rst_r <= 0;
                    if (ffn_row < FFN_DIM) begin
                        core_we     <= 1;
                        core_w_addr <= ffn_row[7:0];
                        core_w_data <= ffn_w2_flat[(ffn_row*EMBED_DIM+ffn_col)*DATA_WIDTH +: 8];
                        ffn_row     <= ffn_row + 1;
                    end else begin
                        ffn_row <= 0;
                        state   <= FFN2_COMPUTE;
                    end
                end

                FFN2_COMPUTE: begin
                    if (ffn_row < FFN_DIM) begin
                        core_feed_valid <= 1;
                        begin : ffn2_pack
                            integer pi;
                            for (pi = 0; pi < NUM_LANES; pi = pi + 1)
                                core_activation_vec[pi*8 +: 8] <= ffn_hidden[ffn_row][7:0];
                        end
                        ffn_row         <= ffn_row + 1;
                    end else begin
                        drain_cnt <= 0;
                        state <= FFN2_DRAIN;
                    end
                end

                FFN2_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt >= 6)
                        state <= FFN2_ACCUM;
                end

                FFN2_ACCUM: begin
                    begin : ffn2_bias
                        reg signed [31:0] val;
                        val = $signed(core_accumulator) +
                              $signed(ffn_b2_flat[ffn_col*DATA_WIDTH +: DATA_WIDTH]);
                        ffn_out_buf[ffn_col] = val[DATA_WIDTH-1:0];
                    end

                    for (fi = 0; fi < NUM_LANES; fi = fi + 1)
                        if (core_zero_mask[fi])
                            block_zero_skips <= block_zero_skips + 1;

                    if (ffn_col + 1 < EMBED_DIM) begin
                        ffn_col  <= ffn_col + 1;
                        ffn_row  <= 0;
                        core_acc_clear <= 1;
                        state    <= FFN2_LOAD;
                    end else begin
                        state <= RESID2;
                    end
                end

                RESID2: begin
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        y_out[i*DATA_WIDTH +: DATA_WIDTH] <=
                            $signed(after_attn[i*DATA_WIDTH +: DATA_WIDTH]) +
                            ffn_out_buf[i];
                    state <= DONE;
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end
            endcase
        end
    end

endmodule
