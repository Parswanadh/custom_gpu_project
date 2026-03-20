// ============================================================================
// Module: gpu_system_top
// Description: Top-level standalone GPU integrating all subsystems.
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

module gpu_system_top #(
    parameter AXI_ADDR_W    = 16,
    parameter AXI_DATA_W    = 32,
    parameter CMD_WIDTH     = 64,
    parameter FIFO_DEPTH    = 16,
    parameter LOCAL_ADDR_W  = 16,
    parameter SP_DEPTH      = 4096,
    parameter SP_DATA_W     = 16,
    parameter WEIGHT_DEPTH  = 4096,
    parameter WEIGHT_DATA_W = 16
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

    // DMA signals
    wire        dma_done, dma_busy, dma_interrupt;
    wire        dma_error;
    wire        dma_local_write_en, dma_local_read_en;
    wire [LOCAL_ADDR_W-1:0] dma_local_write_addr, dma_local_read_addr;
    wire [AXI_DATA_W-1:0] dma_local_write_data;
    wire [AXI_DATA_W-1:0] dma_local_read_data_wire;
    wire                  dma_split_write_ok;
    wire                  dma_split_write_drop;

    // Scratchpad port B read data
    wire [SP_DATA_W-1:0] sp_b_read_data;
    wire                 sp_b_read_valid;

    // Perf counter event wires
    wire        evt_active = !cp_idle;
    wire        evt_stall  = cp_busy && !cp_compute_start;
    wire [15:0] evt_macs = 16'd0;      // Connected when compute pipeline is integrated
    wire [15:0] evt_zero_skips = 16'd0; // Connected when compute pipeline is integrated
    wire        evt_mem_read  = cp_sp_read_en || dma_local_read_en;
    wire        evt_mem_write = cp_sp_write_en || dma_split_write_ok;
    wire        evt_parity_error = 1'b0;

    // Perf counter read interface
    wire [2:0]  perf_read_idx = 3'd0;  // Expose via config regs in future
    wire [31:0] perf_read_data;

    // IRQ aggregation
    wire [7:0] irq_pending = {5'd0, internal_error, dma_interrupt, cp_interrupt};

    // Compute done stub (connect to actual compute units when integrated)
    // For now, auto-ack compute immediately so command processor doesn't hang
    reg compute_done_reg;
    reg err_compute_stub;
    reg err_dma_adapt;
    wire internal_error = err_compute_stub | err_dma_adapt;
    always @(posedge clk) begin
        if (rst_sync) begin
            compute_done_reg <= 1'b0;
            err_compute_stub <= 1'b0;
            err_dma_adapt <= 1'b0;
        end else begin
            compute_done_reg <= cp_compute_start;
            if (cp_compute_start)
                err_compute_stub <= 1'b1;
            if (dma_split_write_drop)
                err_dma_adapt <= 1'b1;
        end
    end
    assign cp_compute_done = compute_done_reg;

    // DMA local read data: scratchpad is narrower than AXI read width.
    assign dma_local_read_data_wire = {{(AXI_DATA_W-SP_DATA_W){1'b0}}, sp_b_read_data};

    // DMA 32-bit to scratchpad 16-bit adaptation for ext->local transfers.
    wire [LOCAL_ADDR_W-1:0] dma_word_addr_lo = dma_local_write_addr >> 1;
    wire [LOCAL_ADDR_W-1:0] dma_word_addr_hi = (dma_local_write_addr >> 1) + {{(LOCAL_ADDR_W-1){1'b0}}, 1'b1};
    wire dma_split_addr_in_range = (dma_word_addr_lo < SP_DEPTH) && (dma_word_addr_hi < SP_DEPTH);
    wire dma_split_addr_aligned  = (dma_local_write_addr[1:0] == 2'b00);
    assign dma_split_write_ok = dma_local_write_en && dma_split_addr_aligned && dma_split_addr_in_range;
    assign dma_split_write_drop = dma_local_write_en && !dma_split_write_ok;

    wire sp_a_write_en = cp_sp_write_en || dma_split_write_ok;
    wire [LOCAL_ADDR_W-1:0] sp_a_write_addr = dma_split_write_ok ? dma_word_addr_hi : cp_sp_write_addr;
    wire [SP_DATA_W-1:0] sp_a_write_data = dma_split_write_ok ?
        dma_local_write_data[(SP_DATA_W*2)-1:SP_DATA_W] :
        cp_sp_write_data;

    wire dma_b_write_en = dma_split_write_ok;
    wire [LOCAL_ADDR_W-1:0] dma_b_write_addr = dma_word_addr_lo;
    wire [SP_DATA_W-1:0] dma_b_write_data = dma_local_write_data[SP_DATA_W-1:0];

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
        .status_busy(cp_busy), .status_idle(cp_idle), .status_error(cp_error | dma_error | internal_error),
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
        .dma_done(dma_done),
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
        .a_read_en(cp_sp_read_en),
        .a_read_addr(cp_sp_read_addr[$clog2(SP_DEPTH)-1:0]),
        .a_read_data(cp_sp_read_data),
        .a_read_valid(),
        .a_write_en(sp_a_write_en),
        .a_write_addr(sp_a_write_addr[$clog2(SP_DEPTH)-1:0]),
        .a_write_data(sp_a_write_data),
        // Port B: DMA engine
        .b_read_en(dma_local_read_en),
        .b_read_addr(dma_local_read_addr[$clog2(SP_DEPTH)-1:0]),
        .b_read_data(sp_b_read_data),
        .b_read_valid(sp_b_read_valid),
        .b_write_en(dma_b_write_en),
        .b_write_addr(dma_b_write_addr[$clog2(SP_DEPTH)-1:0]),
        .b_write_data(dma_b_write_data),
        .usage_count()
    );

    // 6. DMA Engine
    dma_engine #(
        .AXI_ADDR_W(32),
        .AXI_DATA_W(AXI_DATA_W),
        .LOCAL_ADDR_W(LOCAL_ADDR_W),
        .MAX_BURST(16)
    ) u_dma (
        .clk(clk), .rst(rst_sync),
        .start(cp_dma_start),
        .ext_addr(cp_dma_ext_addr),
        .local_addr(cp_dma_local_addr),
        .transfer_len(cp_dma_length),
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
