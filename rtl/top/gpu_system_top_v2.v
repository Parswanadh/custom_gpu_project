// ============================================================================
// Module: gpu_system_top_v2
// Description: Top-level standalone GPU with integrated optimized layer compute.
//   Provides a single AXI4-Lite slave interface for host CPU interaction.
//
//   Subsystems wired:
//     - reset_synchronizer: Safe async-to-sync reset
//     - gpu_config_regs:    AXI4-Lite config register file
//     - command_processor:  FIFO command queue with 8 opcodes
//     - perf_counters:      8 hardware performance counters (PMU)
//     - scratchpad:         Dual-port SRAM for intermediate activations
//     - dma_engine:         AXI4 master for bulk weight transfers
//     - axi_weight_memory:  AXI4-Lite slave weight SRAM
//
//   Host CPU pushes command descriptors via AXI write to command_processor.
//   command_processor dispatches compute ops and DMA transfers.
//   perf_counters track pipeline events for monitoring.
//   gpu_config_regs exposes runtime configuration via AXI.
//
// Parameters: Configurable embedding dimension, heads, etc.
// ============================================================================
`timescale 1ns / 1ps

module gpu_system_top_v2 #(
    parameter AXI_ADDR_W    = 16,
    parameter AXI_DATA_W    = 32,
    parameter CMD_WIDTH     = 64,
    parameter FIFO_DEPTH    = 16,
    parameter LOCAL_ADDR_W  = 16,
    parameter SP_DEPTH      = 4096,
    parameter SP_DATA_W     = 16,
    parameter WEIGHT_DEPTH  = 4096,
    parameter WEIGHT_DATA_W = 16,
    parameter [LOCAL_ADDR_W-1:0] PREFETCH_BASE_ADDR = 16'd768,
    parameter [6*8-1:0] SCHED_STAGE_CYCLES_PACKED = {8'd1, 8'd2, 8'd3, 8'd4, 8'd3, 8'd2}
)(
    input  wire                     clk,
    input  wire                     rst_async_n,    // Active-low async reset from board

    // ---- Host AXI4-Lite Slave (Config Registers) ----
    input  wire                     s_axi_awvalid,
    output wire                     s_axi_awready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_awaddr,
    input  wire                     s_axi_wvalid,
    output wire                     s_axi_wready,
    input  wire [AXI_DATA_W-1:0]   s_axi_wdata,
    output wire                     s_axi_bvalid,
    input  wire                     s_axi_bready,
    output wire [1:0]               s_axi_bresp,
    input  wire                     s_axi_arvalid,
    output wire                     s_axi_arready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_araddr,
    output wire                     s_axi_rvalid,
    input  wire                     s_axi_rready,
    output wire [AXI_DATA_W-1:0]   s_axi_rdata,
    output wire [1:0]               s_axi_rresp,

    // ---- Command Input (from host) ----
    input  wire                     cmd_valid,
    input  wire [CMD_WIDTH-1:0]     cmd_data,
    output wire                     cmd_ready,

    // ---- DMA AXI4 Master (to external memory) ----
    output wire                     m_axi_arvalid,
    input  wire                     m_axi_arready,
    output wire [31:0]              m_axi_araddr,
    output wire [7:0]               m_axi_arlen,
    output wire [2:0]               m_axi_arsize,
    output wire [1:0]               m_axi_arburst,
    input  wire                     m_axi_rvalid,
    output wire                     m_axi_rready,
    input  wire [AXI_DATA_W-1:0]   m_axi_rdata,
    input  wire                     m_axi_rlast,
    input  wire [1:0]               m_axi_rresp,
    output wire                     m_axi_awvalid,
    input  wire                     m_axi_awready,
    output wire [31:0]              m_axi_awaddr,
    output wire [7:0]               m_axi_awlen,
    output wire [2:0]               m_axi_awsize,
    output wire [1:0]               m_axi_awburst,
    output wire                     m_axi_wvalid,
    input  wire                     m_axi_wready,
    output wire [AXI_DATA_W-1:0]   m_axi_wdata,
    output wire                     m_axi_wlast,
    output wire [3:0]               m_axi_wstrb,
    input  wire                     m_axi_bvalid,
    output wire                     m_axi_bready,

    // ---- Interrupt to host ----
    output wire                     irq_out,

    // ---- Performance counter direct outputs ----
    output wire [31:0]              cycle_count,
    output wire [31:0]              zero_skip_total,
    output wire [31:0]              mac_total,

    // ---- Config register outputs for compute path ----
    output wire [1:0]               cfg_precision_mode,
    output wire [7:0]               cfg_dq_scale,
    output wire [3:0]               cfg_dq_offset
);

    // ================================================================
    // Internal wires
    // ================================================================

    // Synchronized reset
    wire rst_sync;

    // Config register outputs
    wire [15:0] cfg_embed_dim, cfg_ffn_dim, cfg_max_seq_len, cfg_vocab_size;
    wire [7:0]  cfg_num_heads, cfg_num_layers;
    wire        cfg_activation_type, cfg_infer_start;
    wire [15:0] cfg_token_in, cfg_position_in;
    wire [7:0]  cfg_irq_enable;
    wire        cfg_irq_out;

    // Command processor signals
    wire        cp_sp_read_en, cp_sp_write_en;
    wire [LOCAL_ADDR_W-1:0] cp_sp_read_addr, cp_sp_write_addr;
    wire [15:0] cp_sp_write_data;
    wire [15:0] cp_sp_read_data;
    wire        cp_dma_start;
    wire [31:0] cp_dma_ext_addr;
    wire [LOCAL_ADDR_W-1:0] cp_dma_local_addr;
    wire [15:0] cp_dma_length;
    wire        cp_dma_done;
    wire        cp_compute_start;
    wire [7:0]  cp_compute_opcode, cp_compute_flags;
    wire [LOCAL_ADDR_W-1:0] cp_compute_src_addr, cp_compute_dst_addr;
    wire [15:0] cp_compute_size;
    wire        cp_compute_done;
    wire        cp_busy, cp_idle;
    wire [31:0] cp_cmds_executed;
    wire        cp_interrupt;
    wire        cp_error;

    // Optional advanced modes from compute flags (feature-gated)
    wire cp_prefetch_enable            = cp_compute_flags[4];
    wire cp_scheduler_enable           = cp_compute_flags[6];
    wire cp_synthetic_embedding_enable = cp_compute_flags[7];

    // Prefetch engine control/data signals
    reg         prefetch_start_pulse;
    reg         prefetch_session_active;
    reg  [7:0]  prefetch_total_layers_cfg;
    wire        prefetch_dma_request;
    wire [31:0] prefetch_dma_src_addr;
    wire [15:0] prefetch_dma_length;
    wire        prefetch_compute_ready;
    wire        prefetch_active;
    wire [7:0]  prefetch_current_layer;
    wire [7:0]  prefetch_layer;
    wire        prefetch_all_done;
    wire        prefetch_error;
    wire [31:0] prefetch_buf_read_data;
    wire [LOCAL_ADDR_W-1:0] prefetch_dma_local_addr = PREFETCH_BASE_ADDR;
    wire [15:0] prefetch_dma_length_aligned = {prefetch_dma_length[15:2], 2'b00};

    // Token-level scheduling controller integration
    reg         scheduler_enqueue_pending;
    reg  [7:0]  scheduler_pending_token;
    reg         scheduler_token_valid;
    reg  [7:0]  scheduler_token_in;
    wire        scheduler_token_ready;
    wire [5:0]  scheduler_stage_active;
    wire [6*8-1:0] scheduler_stage_tokens;
    wire [6*8-1:0] scheduler_stage_progress;
    wire        scheduler_token_out_valid;
    wire [7:0]  scheduler_token_out;
    wire [31:0] scheduler_tokens_processed;
    wire [31:0] scheduler_total_cycles;
    wire [31:0] scheduler_pipeline_stalls;
    wire        scheduler_pipeline_full;
    wire [6*8-1:0] scheduler_stage_cycles_packed = SCHED_STAGE_CYCLES_PACKED;

    // DMA signals
    wire        dma_done, dma_busy, dma_interrupt;
    wire        dma_error;
    wire        dma_local_write_en, dma_local_read_en;
    wire [LOCAL_ADDR_W-1:0] dma_local_write_addr, dma_local_read_addr;
    wire [AXI_DATA_W-1:0] dma_local_write_data;
    wire [AXI_DATA_W-1:0] dma_local_read_data_wire;
    reg         dma_reject_pending;
    reg         dma_reject_pulse;
    reg         cp_dma_pending;
    reg [31:0]  cp_dma_ext_addr_pending;
    reg [LOCAL_ADDR_W-1:0] cp_dma_local_addr_pending;
    reg [15:0]  cp_dma_length_pending;
    reg         dma_owner_prefetch;
    reg         prefetch_dma_pending;
    wire [LOCAL_ADDR_W-1:0] dma_req_word_start = cp_dma_local_addr >> 1;
    wire [LOCAL_ADDR_W:0] dma_req_halfwords = {1'b0, cp_dma_length} >> 1;
    wire [LOCAL_ADDR_W:0] dma_req_word_end = {1'b0, dma_req_word_start} + dma_req_halfwords - 1'b1;
    wire dma_start_addr_aligned = (cp_dma_local_addr[1:0] == 2'b00);
    wire dma_start_len_aligned = (cp_dma_length[1:0] == 2'b00);
    wire dma_start_range_ok = (cp_dma_length == 16'd0) ||
                              ((dma_req_word_start < SP_DEPTH) && (dma_req_word_end < SP_DEPTH));
    wire dma_start_valid = dma_start_addr_aligned && dma_start_len_aligned && dma_start_range_ok;
    wire cp_dma_start_valid = cp_dma_start && dma_start_valid;
    wire dma_start_reject = cp_dma_start && !dma_start_valid;
    wire cp_dma_issue_req = cp_dma_pending || cp_dma_start_valid;
    wire prefetch_dma_issue_req = prefetch_dma_pending || (prefetch_session_active && prefetch_dma_request);
    wire dma_issue_cp = !dma_busy && cp_dma_issue_req;
    wire dma_issue_prefetch = !dma_busy && !cp_dma_issue_req && prefetch_dma_issue_req;
    wire dma_start_to_engine = dma_issue_cp || dma_issue_prefetch;
    wire [31:0] dma_ext_addr_mux =
        dma_issue_cp ? (cp_dma_pending ? cp_dma_ext_addr_pending : cp_dma_ext_addr) : prefetch_dma_src_addr;
    wire [LOCAL_ADDR_W-1:0] dma_local_addr_mux =
        dma_issue_cp ? (cp_dma_pending ? cp_dma_local_addr_pending : cp_dma_local_addr) : prefetch_dma_local_addr;
    wire [15:0] dma_length_mux =
        dma_issue_cp ? (cp_dma_pending ? cp_dma_length_pending : cp_dma_length) : prefetch_dma_length_aligned;
    wire dma_done_to_prefetch = dma_done && dma_owner_prefetch;
    wire dma_done_to_cp = (dma_done && !dma_owner_prefetch) || dma_reject_pulse;

    // Scratchpad data path wires
    wire [15:0]          sp_a_read_data;
    wire [SP_DATA_W-1:0] sp_b_read_data;
    wire                 sp_b_read_valid;

    // Scratchpad bounds guards (prevent aliasing from address truncation)
    wire cp_sp_read_addr_in_range  = (cp_sp_read_addr < SP_DEPTH);
    wire cp_sp_write_addr_in_range = (cp_sp_write_addr < SP_DEPTH);
    wire cp_sp_read_en_safe        = cp_sp_read_en && cp_sp_read_addr_in_range;
    wire cp_sp_write_en_safe       = cp_sp_write_en && cp_sp_write_addr_in_range;

    // DMA 32-bit to scratchpad 16-bit adaptation for ext->local transfers.
    wire [LOCAL_ADDR_W-1:0] dma_word_addr_lo = dma_local_write_addr >> 1;
    wire [LOCAL_ADDR_W-1:0] dma_word_addr_hi = (dma_local_write_addr >> 1) + {{(LOCAL_ADDR_W-1){1'b0}}, 1'b1};
    wire dma_split_addr_in_range = (dma_word_addr_lo < SP_DEPTH) && (dma_word_addr_hi < SP_DEPTH);
    wire dma_split_addr_aligned  = (dma_local_write_addr[1:0] == 2'b00);

    // Perf counter event wires
    wire        evt_active = !cp_idle;
    wire        evt_stall  = cp_busy && !cp_compute_start;
    wire [15:0] evt_macs = 16'd0;
    wire [15:0] evt_zero_skips = 16'd0;
    wire        evt_mem_read  = cp_sp_read_en || dma_local_read_en;
    wire        evt_mem_write = cp_sp_write_en || (dma_local_write_en && !dma_owner_prefetch);
    wire        evt_parity_error = 1'b0;

    // Perf counter read interface
    wire [2:0]  perf_read_idx = 3'd0;  // Expose via config regs in future
    wire [31:0] perf_read_data;

    // IRQ aggregation
    wire [7:0] irq_pending = {6'd0, dma_interrupt, cp_interrupt};

    // Compute integration: command processor -> optimized transformer layer
    localparam CMD_MATMUL       = 8'h02;
    localparam CMD_ACTIVATION   = 8'h03;
    localparam CMD_LAYERNORM    = 8'h04;
    localparam CMD_SOFTMAX      = 8'h05;
    localparam CMD_RESIDUAL_ADD = 8'h06;
    localparam CMD_EMBEDDING    = 8'h07;

    reg                 opt_start;
    reg [8*16-1:0]      opt_token_embedding;
    reg [5:0]           opt_position;
    reg                 opt_active;
    reg                 imprint_fast_start;
    reg                 imprint_fast_active;
    reg [15:0]          opt_total_cycles_d;
    reg [8*16-1:0]      opt_layer_output_d;
    reg [LOCAL_ADDR_W-1:0] opt_dst_addr;
    reg                 writeback_active;
    reg [2:0]           writeback_idx;
    reg [2:0]           writeback_last_idx;
    reg                 writeback_done_pending;
    reg                 compute_sp_write_en;
    reg [LOCAL_ADDR_W-1:0] compute_sp_write_addr;
    reg [15:0]          compute_sp_write_data;
    reg                 compute_done_reg;
    reg                 status_error_reg;
    reg [3:0]           write_words_clamped;
    reg [LOCAL_ADDR_W:0] write_end_addr;

    wire                opt_done;
    wire [8*16-1:0]     opt_layer_output;
    wire                opt_rope_done;
    wire                opt_gqa_done;
    wire                opt_softmax_done;
    wire                opt_gelu_done;
    wire                opt_kv_done;
    wire                opt_comp_done;
    wire [15:0]         opt_rope_cycles;
    wire [15:0]         opt_gqa_cycles;
    wire [15:0]         opt_softmax_cycles;
    wire [15:0]         opt_gelu_cycles;
    wire [15:0]         opt_kv_cycles;
    wire [15:0]         opt_comp_cycles;
    wire [15:0]         opt_total_cycles;
    // Optional imprint mode from compute flags:
    //   bit0   : imprint enable
    //   bits2:1: profile select (01 mini-gpt-hc1-v1, 10 gemma3 export-v1)
    wire                imprint_enable = cp_compute_flags[0];
    wire [1:0]          imprint_profile_sel = cp_compute_flags[2:1];
    wire [1:0]          embedding_profile_sel = imprint_enable ? imprint_profile_sel : 2'b01;
    wire [8*16-1:0]     imprint_embedding;
    wire                imprint_profile_supported;
    wire                imprint_fast_done;
    wire [8*16-1:0]     imprint_fast_output;
    wire [15:0]         imprint_fast_cycles;
    wire                use_hardwired_mini =
        imprint_enable && imprint_profile_supported && (imprint_profile_sel == 2'b01);

    function [3:0] calc_write_words;
        input [15:0] size_bytes;
        reg [15:0] words16;
        begin
            words16 = size_bytes[15:1];
            if (words16 == 0)
                calc_write_words = 4'd1;
            else if (words16 >= 16'd8)
                calc_write_words = 4'd8;
            else
                calc_write_words = words16[3:0];
        end
    endfunction
    // Supported writeback payload is 1..8 words (2..16 bytes), 16-bit aligned.
    wire                compute_size_supported =
        (cp_compute_size >= 16'd2) &&
        (cp_compute_size <= 16'd16) &&
        !cp_compute_size[0];
    wire [3:0] requested_write_words = calc_write_words(cp_compute_size);
    wire [LOCAL_ADDR_W:0] requested_write_end_addr =
        {1'b0, cp_compute_dst_addr} + requested_write_words - 1'b1;

    wire                opt_supported_opcode =
        (cp_compute_opcode == CMD_MATMUL)       ||
        (cp_compute_opcode == CMD_ACTIVATION)   ||
        (cp_compute_opcode == CMD_LAYERNORM)    ||
        (cp_compute_opcode == CMD_SOFTMAX)      ||
        (cp_compute_opcode == CMD_RESIDUAL_ADD) ||
        (cp_compute_opcode == CMD_EMBEDDING);

    imprinted_embedding_rom #(
        .DIM(8),
        .DATA_WIDTH(16)
    ) u_imprint_embedding (
        .token_id(cfg_token_in),
        .position(cfg_position_in),
        .profile_sel(embedding_profile_sel),
        .embedding_out(imprint_embedding),
        .profile_supported(imprint_profile_supported)
    );

    imprinted_mini_transformer_core #(
        .DIM(8),
        .DATA_W(16),
        .LATENCY(8)
    ) u_imprint_fast_core (
        .clk(clk),
        .rst(rst_sync),
        .start(imprint_fast_start),
        .token_embedding(imprint_embedding),
        .position(cfg_position_in[5:0]),
        .done(imprint_fast_done),
        .output_vector(imprint_fast_output),
        .cycles_used(imprint_fast_cycles)
    );

    optimized_transformer_layer #(
        .DIM(8),
        .NUM_Q_HEADS(4),
        .NUM_KV_HEADS(2),
        .HEAD_DIM(4)
    ) u_opt_layer (
        .clk(clk),
        .rst(rst_sync),
        .start(opt_start),
        .token_embedding(opt_token_embedding),
        .position(opt_position),
        .done(opt_done),
        .layer_output(opt_layer_output),
        .rope_complete(opt_rope_done),
        .gqa_complete(opt_gqa_done),
        .softmax_complete(opt_softmax_done),
        .gelu_complete(opt_gelu_done),
        .kv_quant_complete(opt_kv_done),
        .compress_complete(opt_comp_done),
        .rope_cycles(opt_rope_cycles),
        .gqa_cycles(opt_gqa_cycles),
        .softmax_cycles(opt_softmax_cycles),
        .gelu_cycles(opt_gelu_cycles),
        .kv_quant_cycles(opt_kv_cycles),
        .compress_cycles(opt_comp_cycles),
        .total_cycles(opt_total_cycles)
    );

    always @(posedge clk) begin
        if (rst_sync) begin
            opt_start         <= 1'b0;
            opt_token_embedding <= {8*16{1'b0}};
            opt_position      <= 6'd0;
            opt_active        <= 1'b0;
            imprint_fast_start <= 1'b0;
            imprint_fast_active <= 1'b0;
            opt_total_cycles_d <= 16'd0;
            opt_layer_output_d <= {8*16{1'b0}};
            opt_dst_addr      <= {LOCAL_ADDR_W{1'b0}};
            writeback_active  <= 1'b0;
            writeback_idx     <= 3'd0;
            writeback_last_idx <= 3'd7;
            writeback_done_pending <= 1'b0;
            compute_sp_write_en <= 1'b0;
            compute_sp_write_addr <= {LOCAL_ADDR_W{1'b0}};
            compute_sp_write_data <= 16'd0;
            compute_done_reg <= 1'b0;
            status_error_reg <= 1'b0;
            write_words_clamped <= 4'd8;
            write_end_addr <= {(LOCAL_ADDR_W+1){1'b0}};
            dma_reject_pending <= 1'b0;
            dma_reject_pulse <= 1'b0;
            cp_dma_pending <= 1'b0;
            cp_dma_ext_addr_pending <= 32'd0;
            cp_dma_local_addr_pending <= {LOCAL_ADDR_W{1'b0}};
            cp_dma_length_pending <= 16'd0;
            dma_owner_prefetch <= 1'b0;
            prefetch_dma_pending <= 1'b0;
            prefetch_start_pulse <= 1'b0;
            prefetch_session_active <= 1'b0;
            prefetch_total_layers_cfg <= 8'd2;
            scheduler_enqueue_pending <= 1'b0;
            scheduler_pending_token <= 8'd0;
            scheduler_token_valid <= 1'b0;
            scheduler_token_in <= 8'd0;
        end else begin
            opt_start         <= 1'b0;
            imprint_fast_start <= 1'b0;
            compute_done_reg  <= 1'b0;
            compute_sp_write_en <= 1'b0;
            prefetch_start_pulse <= 1'b0;
            scheduler_token_valid <= 1'b0;
            dma_reject_pulse <= dma_reject_pending;
            dma_reject_pending <= 1'b0;
            if (cp_dma_start_valid && dma_busy && !cp_dma_pending) begin
                cp_dma_pending <= 1'b1;
                cp_dma_ext_addr_pending <= cp_dma_ext_addr;
                cp_dma_local_addr_pending <= cp_dma_local_addr;
                cp_dma_length_pending <= cp_dma_length;
            end
            if (prefetch_session_active && prefetch_dma_request)
                prefetch_dma_pending <= 1'b1;
            if (dma_issue_cp && cp_dma_pending)
                cp_dma_pending <= 1'b0;
            if (dma_issue_prefetch)
                prefetch_dma_pending <= 1'b0;
            if (dma_start_to_engine)
                dma_owner_prefetch <= dma_issue_prefetch;
            if (prefetch_all_done) begin
                prefetch_session_active <= 1'b0;
                prefetch_dma_pending <= 1'b0;
            end
            if (scheduler_enqueue_pending && scheduler_token_ready) begin
                scheduler_token_valid <= 1'b1;
                scheduler_token_in <= scheduler_pending_token;
                scheduler_enqueue_pending <= 1'b0;
            end
            if (writeback_done_pending) begin
                compute_done_reg <= 1'b1;
                writeback_done_pending <= 1'b0;
            end

            if (cp_sp_read_en && !cp_sp_read_addr_in_range)
                status_error_reg <= 1'b1;
            if (cp_sp_write_en && !cp_sp_write_addr_in_range)
                status_error_reg <= 1'b1;
            if (cp_error || dma_error || prefetch_error)
                status_error_reg <= 1'b1;
            if ((dma_local_write_en && !dma_owner_prefetch) &&
                (!dma_split_addr_in_range || !dma_split_addr_aligned))
                status_error_reg <= 1'b1;
            if (dma_high_conflict)
                status_error_reg <= 1'b1;
            if (dma_start_reject) begin
                dma_reject_pending <= 1'b1;
                status_error_reg   <= 1'b1;
            end

            if (writeback_active) begin
                compute_sp_write_en   <= 1'b1;
                compute_sp_write_addr <= opt_dst_addr + writeback_idx;
                compute_sp_write_data <= opt_layer_output_d[writeback_idx*16 +: 16];
                if (writeback_idx == writeback_last_idx) begin
                    writeback_active <= 1'b0;
                    writeback_idx    <= 3'd0;
                    writeback_done_pending <= 1'b1;
                end else begin
                    writeback_idx <= writeback_idx + 1'b1;
                end
            end else begin
                if (cp_compute_start && !opt_supported_opcode &&
                    !opt_active && !imprint_fast_active) begin
                    status_error_reg <= 1'b1;
                    compute_done_reg <= 1'b1;
                end

                if (cp_compute_start && opt_supported_opcode &&
                    !opt_active && !imprint_fast_active) begin
                    status_error_reg <= 1'b0;
                    if (!compute_size_supported) begin
                        // Compute path currently supports exactly 2..16 aligned bytes.
                        status_error_reg <= 1'b1;
                        compute_done_reg <= 1'b1;
                    end else if ((cp_compute_dst_addr < SP_DEPTH) && (requested_write_end_addr < SP_DEPTH)) begin
                        if (imprint_enable && !imprint_profile_supported) begin
                            status_error_reg <= 1'b1;
                            compute_done_reg <= 1'b1;
                        end else begin
                            write_words_clamped <= requested_write_words;
                            write_end_addr <= requested_write_end_addr;
                            opt_dst_addr <= cp_compute_dst_addr;
                            opt_position <= cfg_position_in[5:0];
                            writeback_last_idx <= requested_write_words[2:0] - 1'b1;
                            if (cp_scheduler_enable && scheduler_pipeline_full) begin
                                status_error_reg <= 1'b1;
                                compute_done_reg <= 1'b1;
                            end else begin
                                if (cp_scheduler_enable) begin
                                    scheduler_enqueue_pending <= 1'b1;
                                    scheduler_pending_token <= cfg_token_in[7:0];
                                end

                                if (cp_prefetch_enable && !prefetch_session_active) begin
                                    prefetch_start_pulse <= 1'b1;
                                    prefetch_session_active <= 1'b1;
                                    prefetch_total_layers_cfg <=
                                        (cfg_num_layers == 8'd0) ? 8'd2 : cfg_num_layers;
                                end

                                if (cp_synthetic_embedding_enable) begin
                                    opt_token_embedding[15:0]    <= cfg_token_in + cp_compute_src_addr + 16'd0;
                                    opt_token_embedding[31:16]   <= cfg_token_in + cp_compute_src_addr + 16'd1;
                                    opt_token_embedding[47:32]   <= cfg_token_in + cp_compute_src_addr + 16'd2;
                                    opt_token_embedding[63:48]   <= cfg_token_in + cp_compute_src_addr + 16'd3;
                                    opt_token_embedding[79:64]   <= cfg_token_in + cp_compute_src_addr + 16'd4;
                                    opt_token_embedding[95:80]   <= cfg_token_in + cp_compute_src_addr + 16'd5;
                                    opt_token_embedding[111:96]  <= cfg_token_in + cp_compute_src_addr + 16'd6;
                                    opt_token_embedding[127:112] <= cfg_token_in + cp_compute_src_addr + 16'd7;
                                end else begin
                                    opt_token_embedding <= imprint_embedding;
                                end

                                if (use_hardwired_mini) begin
                                    imprint_fast_start  <= 1'b1;
                                    imprint_fast_active <= 1'b1;
                                    opt_active          <= 1'b0;
                                end else begin
                                    opt_start    <= 1'b1;
                                    opt_active   <= 1'b1;
                                end
                            end
                        end
                    end else begin
                        status_error_reg <= 1'b1;
                        compute_done_reg <= 1'b1;
                    end
                end

                if (opt_done && opt_active) begin
                    opt_active         <= 1'b0;
                    opt_total_cycles_d <= opt_total_cycles;
                    opt_layer_output_d <= opt_layer_output;
                    writeback_active   <= 1'b1;
                    writeback_idx      <= 3'd0;
                end

                if (imprint_fast_done && imprint_fast_active) begin
                    imprint_fast_active <= 1'b0;
                    opt_total_cycles_d  <= imprint_fast_cycles;
                    opt_layer_output_d  <= imprint_fast_output;
                    writeback_active    <= 1'b1;
                    writeback_idx       <= 3'd0;
                end
            end
        end
    end
    assign cp_compute_done = compute_done_reg;

    // DMA local read data: scratchpad is narrower than AXI data width.
    assign dma_local_read_data_wire = {{(AXI_DATA_W-SP_DATA_W){1'b0}}, sp_b_read_data};
    assign cp_sp_read_data = cp_sp_read_addr_in_range ? sp_a_read_data : 16'd0;

    wire cp_or_compute_sp_write_en = cp_sp_write_en_safe || compute_sp_write_en;
    wire dma_local_write_to_sp = dma_local_write_en && !dma_owner_prefetch;
    wire dma_high_conflict = dma_local_write_to_sp && cp_or_compute_sp_write_en;
    wire dma_split_write_ok = dma_local_write_to_sp &&
                               dma_split_addr_aligned &&
                               dma_split_addr_in_range &&
                               !dma_high_conflict;
    wire sp_a_write_en = dma_split_write_ok || cp_or_compute_sp_write_en;
    wire [LOCAL_ADDR_W-1:0] sp_a_write_addr =
        dma_split_write_ok ?
            dma_word_addr_hi :
            (compute_sp_write_en ? compute_sp_write_addr : cp_sp_write_addr);
    wire [15:0] sp_a_write_data =
        dma_split_write_ok ?
            dma_local_write_data[31:16] :
            (compute_sp_write_en ? compute_sp_write_data : cp_sp_write_data);

    // ================================================================
    // Module instantiations
    // ================================================================

    // 1. Reset Synchronizer
    reset_synchronizer u_reset_sync (
        .clk(clk),
        .rst_async_n(rst_async_n),
        .rst_sync(rst_sync)
    );

    // 2. GPU Config Registers (AXI4-Lite)
    gpu_config_regs #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W)
    ) u_config_regs (
        .aclk(clk),
        .aresetn(~rst_sync),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .cfg_embed_dim(cfg_embed_dim), .cfg_num_heads(cfg_num_heads),
        .cfg_num_layers(cfg_num_layers), .cfg_ffn_dim(cfg_ffn_dim),
        .cfg_max_seq_len(cfg_max_seq_len), .cfg_vocab_size(cfg_vocab_size),
        .cfg_precision_mode(cfg_precision_mode), .cfg_activation_type(cfg_activation_type),
        .cfg_dq_scale(cfg_dq_scale), .cfg_dq_offset(cfg_dq_offset),
        .cfg_infer_start(cfg_infer_start), .cfg_token_in(cfg_token_in),
        .cfg_position_in(cfg_position_in), .cfg_irq_enable(cfg_irq_enable),
        .status_busy(cp_busy), .status_idle(cp_idle), .status_error(status_error_reg),
        .status_token_out(16'd0),
        .irq_pending(irq_pending),
        .irq_out(irq_out)
    );

    // 3. Command Processor
    command_processor #(
        .CMD_WIDTH(CMD_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH),
        .ADDR_WIDTH(LOCAL_ADDR_W)
    ) u_cmd_proc (
        .clk(clk), .rst(rst_sync),
        .cmd_valid(cmd_valid), .cmd_data(cmd_data), .cmd_ready(cmd_ready),
        .sp_read_en(cp_sp_read_en), .sp_read_addr(cp_sp_read_addr),
        .sp_read_data(cp_sp_read_data),
        .sp_write_en(cp_sp_write_en), .sp_write_addr(cp_sp_write_addr),
        .sp_write_data(cp_sp_write_data),
        .dma_start(cp_dma_start), .dma_ext_addr(cp_dma_ext_addr),
        .dma_local_addr(cp_dma_local_addr), .dma_length(cp_dma_length),
        .dma_done(dma_done_to_cp),
        .compute_start(cp_compute_start), .compute_opcode(cp_compute_opcode),
        .compute_src_addr(cp_compute_src_addr), .compute_dst_addr(cp_compute_dst_addr),
        .compute_size(cp_compute_size), .compute_flags(cp_compute_flags),
        .compute_done(cp_compute_done),
        .busy(cp_busy), .idle(cp_idle), .cmds_executed(cp_cmds_executed),
        .error_out(cp_error),
        .interrupt_out(cp_interrupt)
    );

    // 4. Performance Counters
    perf_counters #(
        .NUM_COUNTERS(8),
        .COUNTER_W(32)
    ) u_perf (
        .clk(clk), .rst(rst_sync),
        .counter_enable(1'b1),
        .counter_clear(1'b0),
        .evt_active(evt_active), .evt_stall(evt_stall),
        .evt_macs(evt_macs), .evt_zero_skips(evt_zero_skips),
        .evt_mem_read(evt_mem_read), .evt_mem_write(evt_mem_write),
        .evt_parity_error(evt_parity_error),
        .read_idx(perf_read_idx), .read_data(perf_read_data),
        .cycle_count(cycle_count), .zero_skip_total(zero_skip_total),
        .mac_total(mac_total)
    );

    // 5. Scratchpad (Dual-Port SRAM)
    //    Port A: Command processor (compute pipeline side)
    //    Port B: DMA engine (data transfer side)
    scratchpad #(
        .DEPTH(SP_DEPTH),
        .DATA_W(SP_DATA_W)
    ) u_scratchpad (
        .clk(clk), .rst(rst_sync),
        // Port A: Command processor
        .a_read_en(cp_sp_read_en_safe),
        .a_read_addr(cp_sp_read_addr[$clog2(SP_DEPTH)-1:0]),
        .a_read_data(sp_a_read_data),
        .a_read_valid(),
        .a_write_en(sp_a_write_en),
        .a_write_addr(sp_a_write_addr[$clog2(SP_DEPTH)-1:0]),
        .a_write_data(sp_a_write_data),
        // Port B: DMA engine
        .b_read_en(dma_local_read_en),
        .b_read_addr(dma_local_read_addr[$clog2(SP_DEPTH)-1:0]),
        .b_read_data(sp_b_read_data),
        .b_read_valid(sp_b_read_valid),
        .b_write_en(dma_split_write_ok),
        .b_write_addr(dma_word_addr_lo[$clog2(SP_DEPTH)-1:0]),
        .b_write_data(dma_local_write_data[15:0]),
        .usage_count()
    );

    // 6. Optional prefetch engine (feature-gated via compute_flags[4])
    prefetch_engine #(
        .BUFFER_DEPTH(64),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(6)
    ) u_prefetch (
        .clk(clk),
        .rst(rst_sync),
        .start(prefetch_start_pulse),
        .layer_done(cp_compute_done),
        .total_layers(prefetch_total_layers_cfg),
        .dma_request(prefetch_dma_request),
        .dma_src_addr(prefetch_dma_src_addr),
        .dma_length(prefetch_dma_length),
        .dma_done(dma_done_to_prefetch),
        .buf_read_en(1'b0),
        .buf_read_addr(6'd0),
        .buf_read_data(prefetch_buf_read_data),
        .buf_write_en(dma_local_write_en && dma_owner_prefetch),
        .buf_write_addr(dma_local_write_addr[7:2]),
        .buf_write_data(dma_local_write_data),
        .compute_ready(prefetch_compute_ready),
        .prefetch_active(prefetch_active),
        .current_layer(prefetch_current_layer),
        .prefetch_layer(prefetch_layer),
        .all_done(prefetch_all_done),
        .error(prefetch_error)
    );

    // 7. Optional layer scheduler telemetry (feature-gated enqueue via compute_flags[6])
    layer_pipeline_controller #(
        .NUM_STAGES(6),
        .TOKEN_WIDTH(8)
    ) u_layer_scheduler (
        .clk(clk),
        .rst(rst_sync),
        .token_valid(scheduler_token_valid),
        .token_in(scheduler_token_in),
        .token_ready(scheduler_token_ready),
        .stage_cycles_packed(scheduler_stage_cycles_packed),
        .stage_active(scheduler_stage_active),
        .stage_tokens(scheduler_stage_tokens),
        .stage_progress_packed(scheduler_stage_progress),
        .token_out_valid(scheduler_token_out_valid),
        .token_out(scheduler_token_out),
        .tokens_processed(scheduler_tokens_processed),
        .total_cycles(scheduler_total_cycles),
        .pipeline_stalls(scheduler_pipeline_stalls),
        .pipeline_full(scheduler_pipeline_full)
    );

    // 8. DMA Engine (shared by command DMA and optional prefetch DMA)
    dma_engine #(
        .AXI_ADDR_W(32),
        .AXI_DATA_W(AXI_DATA_W),
        .LOCAL_ADDR_W(LOCAL_ADDR_W),
        .MAX_BURST(16)
    ) u_dma (
        .clk(clk), .rst(rst_sync),
        .start(dma_start_to_engine),
        .ext_addr(dma_ext_addr_mux),
        .local_addr(dma_local_addr_mux),
        .transfer_len(dma_length_mux),
        .direction(1'b0),  // Default: ext→local
        .done(dma_done), .busy(dma_busy), .error(dma_error),
        // AXI4 Master
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast), .m_axi_rresp(m_axi_rresp),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        // Local memory
        .local_write_en(dma_local_write_en),
        .local_write_addr(dma_local_write_addr),
        .local_write_data(dma_local_write_data),
        .local_read_en(dma_local_read_en),
        .local_read_addr(dma_local_read_addr),
        .local_read_data(dma_local_read_data_wire),
        .interrupt(dma_interrupt)
    );

endmodule
