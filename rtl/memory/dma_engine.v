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
module dma_engine #(
    parameter AXI_ADDR_W   = 32,
    parameter AXI_DATA_W   = 32,
    parameter LOCAL_ADDR_W = 16,
    parameter MAX_BURST    = 16     // Max AXI burst length
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
    assign m_axi_wstrb   = 4'b1111;

    // State machine
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_RD_ADDR    = 4'd1;
    localparam S_RD_DATA    = 4'd2;
    localparam S_WR_ADDR    = 4'd3;
    localparam S_WR_DATA    = 4'd4;
    localparam S_WR_RESP    = 4'd5;
    localparam S_DONE       = 4'd6;

    reg [AXI_ADDR_W-1:0]   cur_ext_addr;
    reg [LOCAL_ADDR_W-1:0]  cur_local_addr;
    reg [15:0]              remaining;
    reg [7:0]               burst_count;
    reg [7:0]               beat_count;

    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            busy            <= 1'b0;
            interrupt       <= 1'b0;
            m_axi_arvalid   <= 1'b0;
            m_axi_rready    <= 1'b0;
            m_axi_awvalid   <= 1'b0;
            m_axi_wvalid    <= 1'b0;
            m_axi_wlast     <= 1'b0;
            m_axi_bready    <= 1'b0;
            local_write_en  <= 1'b0;
            local_read_en   <= 1'b0;
        end else begin
            done           <= 1'b0;
            interrupt      <= 1'b0;
            local_write_en <= 1'b0;
            local_read_en  <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
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
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready  <= 1'b1;
                        beat_count    <= 0;
                        state <= S_RD_DATA;
                    end
                end

                S_RD_DATA: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        // Write full 32-bit word to local memory (all 4 bytes)
                        local_write_en   <= 1'b1;
                        local_write_addr <= cur_local_addr;
                        local_write_data <= m_axi_rdata;   // Full word, not just [7:0]
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
                end

                // ---- Write from local to external ----
                S_WR_ADDR: begin
                    m_axi_awvalid <= 1'b1;
                    m_axi_awaddr  <= cur_ext_addr;
                    m_axi_awlen   <= (remaining > MAX_BURST*4) ?
                                     (MAX_BURST - 1) : (((remaining + 3) >> 2) - 1);
                    if (m_axi_awready && m_axi_awvalid) begin
                        m_axi_awvalid <= 1'b0;
                        beat_count    <= 0;
                        burst_count   <= (remaining > MAX_BURST*4) ?
                                         MAX_BURST : ((remaining + 3) >> 2);
                        state <= S_WR_DATA;
                    end
                end

                S_WR_DATA: begin
                    local_read_en   <= 1'b1;
                    local_read_addr <= cur_local_addr;
                    m_axi_wvalid    <= 1'b1;
                    m_axi_wdata     <= local_read_data;   // Full word, not zero-padded byte
                    m_axi_wlast     <= (beat_count + 1 >= burst_count);

                    if (m_axi_wready && m_axi_wvalid) begin
                        cur_local_addr <= cur_local_addr + 4;
                        cur_ext_addr   <= cur_ext_addr + 4;
                        remaining      <= (remaining >= 4) ? remaining - 4 : 0;
                        beat_count     <= beat_count + 1;

                        if (m_axi_wlast) begin
                            m_axi_wvalid <= 1'b0;
                            m_axi_bready <= 1'b1;
                            state <= S_WR_RESP;
                        end
                    end
                end

                S_WR_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                        if (remaining == 0)
                            state <= S_DONE;
                        else
                            state <= S_WR_ADDR;
                    end
                end

                S_DONE: begin
                    done      <= 1'b1;
                    interrupt <= 1'b1;
                    state     <= S_IDLE;
                end
            endcase
        end
    end

endmodule

