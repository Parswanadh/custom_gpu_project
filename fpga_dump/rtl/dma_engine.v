// ============================================================================
// Module: dma_engine
// Description: Simple DMA engine for bulk weight/data transfers (Issue #17).
//   AXI4 master interface for reading from external memory (DRAM).
//   Writes into local weight SRAM or scratchpad.
//
//   Supports:
//     - Burst reads from external memory
//     - Sequential writes to local SRAM
//     - Configurable transfer length
//     - Interrupt on completion
//
// Parameters: AXI_ADDR_W, AXI_DATA_W, LOCAL_ADDR_W
// ============================================================================
`timescale 1ns / 1ps

module dma_engine #(
    parameter AXI_ADDR_W   = 32,
    parameter AXI_DATA_W   = 32,
    parameter LOCAL_ADDR_W = 16,
    parameter MAX_BURST    = 16,    // Max AXI burst length
    parameter WAIT_TIMEOUT_CYCLES = 16'd4096
)(
    input  wire                     clk,
    input  wire                     rst,

    // Control interface
    input  wire                     start,
    input  wire [AXI_ADDR_W-1:0]   ext_addr,       // External memory start address
    input  wire [LOCAL_ADDR_W-1:0]  local_addr,     // Local SRAM start address
    input  wire [15:0]              transfer_len,   // Number of bytes to transfer
    input  wire                     direction,      // 0 = ext→local (read), 1 = local→ext (write)
    output reg                      done,
    output reg                      busy,
    output reg                      error,

    // AXI4 Master Read Address Channel
    output reg                      m_axi_arvalid,
    input  wire                     m_axi_arready,
    output reg  [AXI_ADDR_W-1:0]   m_axi_araddr,
    output reg  [7:0]              m_axi_arlen,     // Burst length - 1
    output wire [2:0]              m_axi_arsize,
    output wire [1:0]              m_axi_arburst,

    // AXI4 Master Read Data Channel
    input  wire                     m_axi_rvalid,
    output reg                      m_axi_rready,
    input  wire [AXI_DATA_W-1:0]   m_axi_rdata,
    input  wire                     m_axi_rlast,
    input  wire [1:0]              m_axi_rresp,

    // AXI4 Master Write Address Channel
    output reg                      m_axi_awvalid,
    input  wire                     m_axi_awready,
    output reg  [AXI_ADDR_W-1:0]   m_axi_awaddr,
    output reg  [7:0]              m_axi_awlen,
    output wire [2:0]              m_axi_awsize,
    output wire [1:0]              m_axi_awburst,

    // AXI4 Master Write Data Channel
    output reg                      m_axi_wvalid,
    input  wire                     m_axi_wready,
    output reg  [AXI_DATA_W-1:0]   m_axi_wdata,
    output reg                      m_axi_wlast,
    output wire [3:0]              m_axi_wstrb,

    // AXI4 Write Response
    input  wire                     m_axi_bvalid,
    output reg                      m_axi_bready,

    // Local memory write interface (for ext→local transfers)
    output reg                      local_write_en,
    output reg  [LOCAL_ADDR_W-1:0]  local_write_addr,
    output reg  [AXI_DATA_W-1:0]   local_write_data,   // Full AXI-width write (was 8-bit, lost 3 bytes/beat)

    // Local memory read interface (for local→ext transfers)
    output reg                      local_read_en,
    output reg  [LOCAL_ADDR_W-1:0]  local_read_addr,
    input  wire [AXI_DATA_W-1:0]   local_read_data,    // Full AXI-width read

    // Interrupt
    output reg                      interrupt
);

    // Fixed AXI parameters
    assign m_axi_arsize  = 3'b010;  // 4 bytes per beat
    assign m_axi_arburst = 2'b01;   // INCR burst
    assign m_axi_awsize  = 3'b010;
    assign m_axi_awburst = 2'b01;

    reg [3:0] m_axi_wstrb_reg;
    assign m_axi_wstrb = m_axi_wstrb_reg;

    function [3:0] tail_wstrb;
        input [2:0] byte_count;
        begin
            case (byte_count)
                3'd0: tail_wstrb = 4'b0000;
                3'd1: tail_wstrb = 4'b0001;
                3'd2: tail_wstrb = 4'b0011;
                3'd3: tail_wstrb = 4'b0111;
                default: tail_wstrb = 4'b1111;
            endcase
        end
    endfunction

    function [AXI_DATA_W-1:0] mask_tail_data;
        input [AXI_DATA_W-1:0] data_in;
        input [2:0]            byte_count;
        integer                byte_idx;
        begin
            mask_tail_data = {AXI_DATA_W{1'b0}};
            for (byte_idx = 0; byte_idx < AXI_DATA_W/8; byte_idx = byte_idx + 1) begin
                if (byte_idx < byte_count)
                    mask_tail_data[(byte_idx*8) +: 8] = data_in[(byte_idx*8) +: 8];
            end
        end
    endfunction

    // State machine
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_RD_ADDR    = 4'd1;
    localparam S_RD_DATA    = 4'd2;
    localparam S_WR_ADDR    = 4'd3;
    localparam S_WR_DATA    = 4'd4;
    localparam S_WR_RESP    = 4'd5;
    localparam S_DONE       = 4'd6;
    localparam [1:0] RESP_OKAY = 2'b00;
    reg [AXI_ADDR_W-1:0]   cur_ext_addr;
    reg [LOCAL_ADDR_W-1:0]  cur_local_addr;
    reg [15:0]              remaining;
    reg [7:0]               burst_count;
    reg [7:0]               beat_count;
    reg [15:0]              wait_counter;
    reg                     error_latched;

    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            busy            <= 1'b0;
            error           <= 1'b0;
            interrupt       <= 1'b0;
            m_axi_arvalid   <= 1'b0;
            m_axi_rready    <= 1'b0;
            m_axi_awvalid   <= 1'b0;
            m_axi_wvalid    <= 1'b0;
            m_axi_wlast     <= 1'b0;
            m_axi_wstrb_reg <= 4'b0000;
            m_axi_bready    <= 1'b0;
            local_write_en  <= 1'b0;
            local_read_en   <= 1'b0;
            wait_counter    <= 16'd0;
            error_latched   <= 1'b0;
        end else begin
            done           <= 1'b0;
            error          <= 1'b0;
            interrupt      <= 1'b0;
            local_write_en <= 1'b0;
            local_read_en  <= 1'b0;
            m_axi_wstrb_reg <= 4'b0000;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        error_latched <= 1'b0;
                        wait_counter  <= 16'd0;
                        if (transfer_len == 0) begin
                            // Hard guard: zero-length transfers complete immediately
                            // and must not emit any AXI traffic.
                            m_axi_arvalid <= 1'b0;
                            m_axi_rready  <= 1'b0;
                            m_axi_awvalid <= 1'b0;
                            m_axi_wvalid  <= 1'b0;
                            m_axi_wlast   <= 1'b0;
                            m_axi_bready  <= 1'b0;
                            done          <= 1'b1;
                            interrupt     <= 1'b1;
                            busy          <= 1'b0;
                            state         <= S_IDLE;
                        end else begin
                            cur_ext_addr   <= ext_addr;
                            cur_local_addr <= local_addr;
                            remaining      <= transfer_len;
                            busy           <= 1'b1;
                            if (!direction)
                                state <= S_RD_ADDR;
                            else
                                state <= S_WR_ADDR;
                        end
                    end
                end

                // ---- Read from external, write to local ----
                S_RD_ADDR: begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr  <= cur_ext_addr;
                    // Compute burst length (synthesizable — shift instead of divide)
                    burst_count <= (remaining > MAX_BURST*4) ?
                                   (MAX_BURST - 1) : (((remaining + 3) >> 2) - 1);
                    m_axi_arlen <= (remaining > MAX_BURST*4) ?
                                   (MAX_BURST - 1) : (((remaining + 3) >> 2) - 1);
                    if (m_axi_arready && m_axi_arvalid) begin
                        wait_counter  <= 16'd0;
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        beat_count    <= 0;
                        state <= S_RD_DATA;
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b0;
                        busy          <= 1'b0;
                        error_latched <= 1'b1;
                        wait_counter  <= 16'd0;
                        state         <= S_DONE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end

                S_RD_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        wait_counter <= 16'd0;
                        if (m_axi_rresp != RESP_OKAY) begin
                            m_axi_rready <= 1'b0;
                            busy         <= 1'b0;
                            error_latched <= 1'b1;
                            state        <= S_DONE;
                        end else begin
                        // Tail policy: preserve only requested bytes on final beat.
                        local_write_en   <= 1'b1;
                        local_write_addr <= cur_local_addr;
                        if (remaining >= 4)
                            local_write_data <= m_axi_rdata;
                        else
                            local_write_data <= mask_tail_data(m_axi_rdata, {1'b0, remaining[1:0]});
                        cur_local_addr   <= cur_local_addr + 4;
                        cur_ext_addr     <= cur_ext_addr + 4;
                        remaining        <= (remaining >= 4) ? remaining - 4 : 0;
                        beat_count       <= beat_count + 1;

                        if (m_axi_rlast || remaining <= 4) begin
                            m_axi_rready <= 1'b0;
                            if (remaining <= 4)
                                state <= S_DONE;
                            else
                                state <= S_RD_ADDR;  // Next burst
                        end
                        end
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b0;
                        busy          <= 1'b0;
                        error_latched <= 1'b1;
                        wait_counter  <= 16'd0;
                        state         <= S_DONE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end

                // ---- Write from local to external ----
                S_WR_ADDR: begin
                    m_axi_awvalid <= 1'b1;
                    m_axi_awaddr  <= cur_ext_addr;
                    m_axi_awlen   <= (remaining > MAX_BURST*4) ?
                                     (MAX_BURST - 1) : (((remaining + 3) >> 2) - 1);
                    if (m_axi_awready && m_axi_awvalid) begin
                        wait_counter  <= 16'd0;
                        m_axi_awvalid <= 1'b0;
                        beat_count    <= 0;
                        burst_count   <= (remaining > MAX_BURST*4) ?
                                         MAX_BURST : ((remaining + 3) >> 2);
                        state <= S_WR_DATA;
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b0;
                        busy          <= 1'b0;
                        error_latched <= 1'b1;
                        wait_counter  <= 16'd0;
                        state         <= S_DONE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end

                S_WR_DATA: begin
                    local_read_en   <= 1'b1;
                    local_read_addr <= cur_local_addr;
                    m_axi_wvalid    <= 1'b1;
                    if (remaining >= 4) begin
                        m_axi_wdata     <= local_read_data;
                        m_axi_wstrb_reg <= 4'b1111;
                    end else begin
                        m_axi_wdata     <= mask_tail_data(local_read_data, {1'b0, remaining[1:0]});
                        m_axi_wstrb_reg <= tail_wstrb({1'b0, remaining[1:0]});
                    end
                    m_axi_wlast     <= (beat_count + 1 >= burst_count);

                    if (m_axi_wready && m_axi_wvalid) begin
                        wait_counter <= 16'd0;
                        cur_local_addr <= cur_local_addr + 4;
                        cur_ext_addr   <= cur_ext_addr + 4;
                        remaining      <= (remaining >= 4) ? remaining - 4 : 0;
                        beat_count     <= beat_count + 1;

                        if (m_axi_wlast) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state <= S_WR_RESP;
                        end
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b0;
                        busy          <= 1'b0;
                        error_latched <= 1'b1;
                        wait_counter  <= 16'd0;
                        state         <= S_DONE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end

                S_WR_RESP: begin
                    if (m_axi_bvalid) begin
                        wait_counter <= 16'd0;
                        m_axi_bready <= 1'b0;
                        if (remaining == 0)
                            state <= S_DONE;
                        else
                            state <= S_WR_ADDR;
                    end else if (wait_counter >= WAIT_TIMEOUT_CYCLES) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b0;
                        m_axi_awvalid <= 1'b0;
                        m_axi_wvalid  <= 1'b0;
                        m_axi_bready  <= 1'b0;
                        busy          <= 1'b0;
                        error_latched <= 1'b1;
                        wait_counter  <= 16'd0;
                        state         <= S_DONE;
                    end else begin
                        wait_counter <= wait_counter + 1'b1;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    error     <= error_latched;
                    interrupt <= 1'b1;
                    state     <= S_IDLE;
                end
            endcase
        end
    end

endmodule

