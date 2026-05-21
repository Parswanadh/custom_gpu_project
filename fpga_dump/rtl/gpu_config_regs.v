// ============================================================================
// Module: gpu_config_regs
// Description: Memory-mapped configuration register file (Issue #20).
//   AXI4-Lite slave for runtime GPU configuration.
//
//   Address Map (32-bit aligned):
//     0x0000: GPU_ID         (RO) — Hardware ID and version
//     0x0004: GPU_STATUS     (RO) — Status flags (busy, idle, error)
//     0x0008: EMBED_DIM      (RW) — Embedding dimension
//     0x000C: NUM_HEADS      (RW) — Number of attention heads
//     0x0010: NUM_LAYERS     (RW) — Number of transformer layers
//     0x0014: FFN_DIM        (RW) — FFN hidden dimension
//     0x0018: MAX_SEQ_LEN    (RW) — Maximum sequence length
//     0x001C: VOCAB_SIZE     (RW) — Vocabulary size
//     0x0020: PRECISION_MODE (RW) — 0=Q8.8, 1=BF16, 2=INT4
//     0x0024: ACTIVATION_TYPE(RW) — 0=GELU, 1=ReLU
//     0x0028: DQ_SCALE       (RW) — Dequantizer scale factor
//     0x002C: DQ_OFFSET      (RW) — Dequantizer zero-point
//     0x0030: INFER_START    (WO) — Write 1 to start inference
//     0x0034: INFER_TOKEN_IN (RW) — Input token ID
//     0x0038: INFER_POS_IN   (RW) — Input position
//     0x003C: INFER_TOKEN_OUT(RO) — Output predicted token
//     0x0040: IRQ_ENABLE     (RW) — Interrupt enable mask
//     0x0044: IRQ_STATUS     (RW1C) — Interrupt status (write 1 to clear)
// ============================================================================
`timescale 1ns / 1ps

module gpu_config_regs #(
    parameter AXI_ADDR_W = 16,
    parameter AXI_DATA_W = 32
)(
    input  wire                     aclk,
    input  wire                     aresetn,

    // AXI4-Lite Slave Interface
    input  wire                     s_axi_awvalid,
    output reg                      s_axi_awready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_awaddr,
    input  wire                     s_axi_wvalid,
    output reg                      s_axi_wready,
    input  wire [AXI_DATA_W-1:0]   s_axi_wdata,
    output reg                      s_axi_bvalid,
    input  wire                     s_axi_bready,
    output wire [1:0]               s_axi_bresp,
    input  wire                     s_axi_arvalid,
    output reg                      s_axi_arready,
    input  wire [AXI_ADDR_W-1:0]   s_axi_araddr,
    output reg                      s_axi_rvalid,
    input  wire                     s_axi_rready,
    output reg  [AXI_DATA_W-1:0]   s_axi_rdata,
    output wire [1:0]               s_axi_rresp,

    // Configuration outputs
    output reg  [15:0]              cfg_embed_dim,
    output reg  [7:0]               cfg_num_heads,
    output reg  [7:0]               cfg_num_layers,
    output reg  [15:0]              cfg_ffn_dim,
    output reg  [15:0]              cfg_max_seq_len,
    output reg  [15:0]              cfg_vocab_size,
    output reg  [1:0]               cfg_precision_mode,
    output reg                      cfg_activation_type,
    output reg  [7:0]               cfg_dq_scale,
    output reg  [3:0]               cfg_dq_offset,
    output reg                      cfg_infer_start,
    output reg  [15:0]              cfg_token_in,
    output reg  [15:0]              cfg_position_in,
    output reg  [7:0]               cfg_irq_enable,

    // Status inputs
    input  wire                     status_busy,
    input  wire                     status_idle,
    input  wire                     status_error,
    input  wire [15:0]              status_token_out,
    input  wire [7:0]               irq_pending,

    // Interrupt output
    output wire                     irq_out
);

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    localparam GPU_ID_VAL = 32'hB17B_0001;  // BitbyBit v0.01

    // IRQ status register
    reg [7:0] irq_status;
    assign irq_out = |(irq_status & cfg_irq_enable);

    // AXI write state
    reg [AXI_ADDR_W-1:0] wr_addr;
    reg                   wr_addr_valid;
    reg [AXI_DATA_W-1:0] wr_data;
    reg                   wr_data_valid;
    wire                  wr_commit;
    wire                  irq_clear_we;
    wire [7:0]            irq_clear_mask;
    wire [7:0]            irq_status_merged;

    assign wr_commit       = wr_addr_valid && wr_data_valid;
    assign irq_clear_we    = wr_commit && (wr_addr[7:0] == 8'h44);
    assign irq_clear_mask  = irq_clear_we ? wr_data[7:0] : 8'd0;
    // Deterministic IRQ merge: W1C clears previously latched bits, new hardware
    // events in irq_pending are applied in the same cycle and win on collisions.
    assign irq_status_merged = (irq_status & ~irq_clear_mask) | irq_pending;

    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready    <= 1'b0;
            s_axi_wready     <= 1'b0;
            s_axi_bvalid     <= 1'b0;
            s_axi_arready    <= 1'b0;
            s_axi_rvalid     <= 1'b0;
            s_axi_rdata      <= 32'd0;
            wr_addr_valid    <= 1'b0;
            wr_data_valid    <= 1'b0;
            cfg_embed_dim    <= 16'd64;
            cfg_num_heads    <= 8'd4;
            cfg_num_layers   <= 8'd2;
            cfg_ffn_dim      <= 16'd256;
            cfg_max_seq_len  <= 16'd128;
            cfg_vocab_size   <= 16'd50257;
            cfg_precision_mode <= 2'd0;
            cfg_activation_type <= 1'b0;
            cfg_dq_scale     <= 8'd1;
            cfg_dq_offset    <= 4'd0;
            cfg_infer_start  <= 1'b0;
            cfg_token_in     <= 16'd0;
            cfg_position_in  <= 16'd0;
            cfg_irq_enable   <= 8'd0;
            irq_status       <= 8'd0;
        end else begin
            cfg_infer_start <= 1'b0;  // Auto-clear
            irq_status <= irq_status_merged;

            // Write address (AXI4-Lite permits one outstanding write response)
            if (s_axi_awvalid && !wr_addr_valid && !s_axi_bvalid) begin
                wr_addr <= s_axi_awaddr;
                wr_addr_valid <= 1'b1;
                s_axi_awready <= 1'b1;
            end else s_axi_awready <= 1'b0;

            // Write data
            if (s_axi_wvalid && !wr_data_valid && !s_axi_bvalid) begin
                wr_data <= s_axi_wdata;
                wr_data_valid <= 1'b1;
                s_axi_wready <= 1'b1;
            end else s_axi_wready <= 1'b0;

            // Complete write
            if (wr_commit) begin
                case (wr_addr[7:0])
                    8'h08: cfg_embed_dim      <= wr_data[15:0];
                    8'h0C: cfg_num_heads      <= wr_data[7:0];
                    8'h10: cfg_num_layers     <= wr_data[7:0];
                    8'h14: cfg_ffn_dim        <= wr_data[15:0];
                    8'h18: cfg_max_seq_len    <= wr_data[15:0];
                    8'h1C: cfg_vocab_size     <= wr_data[15:0];
                    8'h20: cfg_precision_mode <= wr_data[1:0];
                    8'h24: cfg_activation_type <= wr_data[0];
                    8'h28: cfg_dq_scale       <= wr_data[7:0];
                    8'h2C: cfg_dq_offset      <= wr_data[3:0];
                    8'h30: cfg_infer_start    <= wr_data[0];
                    8'h34: cfg_token_in       <= wr_data[15:0];
                    8'h38: cfg_position_in    <= wr_data[15:0];
                    8'h40: cfg_irq_enable     <= wr_data[7:0];
                    8'h44: ;  // W1C handled by irq_status_merged
                endcase
                wr_addr_valid <= 1'b0;
                wr_data_valid <= 1'b0;
                s_axi_bvalid  <= 1'b1;
            end

            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            // Read
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= GPU_ID_VAL;
                    8'h04: s_axi_rdata <= {29'd0, status_error, status_idle, status_busy};
                    8'h08: s_axi_rdata <= {16'd0, cfg_embed_dim};
                    8'h0C: s_axi_rdata <= {24'd0, cfg_num_heads};
                    8'h10: s_axi_rdata <= {24'd0, cfg_num_layers};
                    8'h14: s_axi_rdata <= {16'd0, cfg_ffn_dim};
                    8'h18: s_axi_rdata <= {16'd0, cfg_max_seq_len};
                    8'h1C: s_axi_rdata <= {16'd0, cfg_vocab_size};
                    8'h20: s_axi_rdata <= {30'd0, cfg_precision_mode};
                    8'h24: s_axi_rdata <= {31'd0, cfg_activation_type};
                    8'h28: s_axi_rdata <= {24'd0, cfg_dq_scale};
                    8'h2C: s_axi_rdata <= {28'd0, cfg_dq_offset};
                    8'h34: s_axi_rdata <= {16'd0, cfg_token_in};
                    8'h38: s_axi_rdata <= {16'd0, cfg_position_in};
                    8'h3C: s_axi_rdata <= {16'd0, status_token_out};
                    8'h40: s_axi_rdata <= {24'd0, cfg_irq_enable};
                    8'h44: s_axi_rdata <= {24'd0, irq_status};
                    default: s_axi_rdata <= 32'hDEAD_BEEF;
                endcase
            end else s_axi_arready <= 1'b0;

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;
        end
    end

endmodule

