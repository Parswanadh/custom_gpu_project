// ============================================================================
// Module: axi_weight_memory
// Description: AXI4-Lite slave interface for loading weights from external
//   memory (CPU, DMA, or SoC bus) into the GPU's weight storage.
//
//   This enables the GPU to be used as a real peripheral:
//     1. CPU writes weights via AXI bus to this module
//     2. Module stores weights in internal SRAM
//     3. GPU cores read weights from this SRAM during inference
//
//   Address Map:
//     0x0000 - 0x0FFF: Weight memory (4096 x 8-bit = 4KB)
//     0x1000:          Control register (bit 0 = start inference)
//     0x1004:          Status register (bit 0 = busy, bit 1 = done)
//     0x1008:          Weight count register
//     0x100C:          Zero-skip count (read-only)
//
// AXI4-Lite: 32-bit address, 32-bit data, no bursts
// ============================================================================
module axi_weight_memory #(
    parameter MEM_DEPTH    = 4096,  // Weight memory depth
    parameter DATA_WIDTH   = 8,     // Weight precision
    parameter AXI_ADDR_W   = 16,    // AXI address width
    parameter AXI_DATA_W   = 32     // AXI data width
)(
    // AXI4-Lite clock + reset
    input  wire                     aclk,
    input  wire                     aresetn,    // Active-low reset

    // AXI4-Lite Write Address Channel
    input  wire                     s_axi_awvalid,
    output reg                      s_axi_awready,
    input  wire [AXI_ADDR_W-1:0]    s_axi_awaddr,

    // AXI4-Lite Write Data Channel
    input  wire                     s_axi_wvalid,
    output reg                      s_axi_wready,
    input  wire [AXI_DATA_W-1:0]    s_axi_wdata,
    input  wire [3:0]               s_axi_wstrb,

    // AXI4-Lite Write Response Channel
    output reg                      s_axi_bvalid,
    input  wire                     s_axi_bready,
    output wire [1:0]               s_axi_bresp,

    // AXI4-Lite Read Address Channel
    input  wire                     s_axi_arvalid,
    output reg                      s_axi_arready,
    input  wire [AXI_ADDR_W-1:0]    s_axi_araddr,

    // AXI4-Lite Read Data Channel
    output reg                      s_axi_rvalid,
    input  wire                     s_axi_rready,
    output reg  [AXI_DATA_W-1:0]    s_axi_rdata,
    output wire [1:0]               s_axi_rresp,

    // GPU-side interface: weight read port
    input  wire                     gpu_read_en,
    input  wire [$clog2(MEM_DEPTH)-1:0] gpu_read_addr,
    output reg  [DATA_WIDTH-1:0]    gpu_read_data,
    output reg                      gpu_read_valid,

    // GPU-side interface: bulk write port (for DMA-like transfers)
    output reg                      weight_load_valid,
    output reg  [$clog2(MEM_DEPTH)-1:0] weight_load_addr,
    output reg  [DATA_WIDTH-1:0]    weight_load_data,

    // Control/Status
    output reg                      start_inference,
    input  wire                     inference_busy,
    input  wire                     inference_done,
    input  wire [31:0]              zero_skip_count
);

    // OKAY response
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    // Weight memory
    reg [DATA_WIDTH-1:0] weight_mem [0:MEM_DEPTH-1];
    reg [31:0]           weight_count;

    // AXI write handling
    reg [AXI_ADDR_W-1:0] wr_addr;
    reg                   wr_addr_valid;
    reg                   wr_data_valid;
    reg [AXI_DATA_W-1:0]  wr_data;

    integer i;

    always @(posedge aclk) begin
        if (!aresetn) begin
            s_axi_awready    <= 1'b0;
            s_axi_wready     <= 1'b0;
            s_axi_bvalid     <= 1'b0;
            s_axi_arready    <= 1'b0;
            s_axi_rvalid     <= 1'b0;
            s_axi_rdata      <= 0;
            wr_addr_valid    <= 1'b0;
            wr_data_valid    <= 1'b0;
            start_inference  <= 1'b0;
            weight_count     <= 0;
            weight_load_valid <= 1'b0;
            gpu_read_valid   <= 1'b0;

            for (i = 0; i < MEM_DEPTH; i = i + 1)
                weight_mem[i] <= 0;

        end else begin
            // Defaults
            weight_load_valid <= 1'b0;
            gpu_read_valid    <= 1'b0;
            start_inference   <= 1'b0;

            // ---- AXI Write Address ----
            if (s_axi_awvalid && !wr_addr_valid) begin
                wr_addr       <= s_axi_awaddr;
                wr_addr_valid <= 1'b1;
                s_axi_awready <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
            end

            // ---- AXI Write Data ----
            if (s_axi_wvalid && !wr_data_valid) begin
                wr_data       <= s_axi_wdata;
                wr_data_valid <= 1'b1;
                s_axi_wready  <= 1'b1;
            end else begin
                s_axi_wready <= 1'b0;
            end

            // ---- Complete Write ----
            if (wr_addr_valid && wr_data_valid) begin
                if (wr_addr < MEM_DEPTH) begin
                    // Weight memory write (pack 4 bytes per AXI word)
                    if (s_axi_wstrb[0]) weight_mem[wr_addr]   <= wr_data[7:0];
                    if (s_axi_wstrb[1] && wr_addr+1 < MEM_DEPTH)
                        weight_mem[wr_addr+1] <= wr_data[15:8];
                    if (s_axi_wstrb[2] && wr_addr+2 < MEM_DEPTH)
                        weight_mem[wr_addr+2] <= wr_data[23:16];
                    if (s_axi_wstrb[3] && wr_addr+3 < MEM_DEPTH)
                        weight_mem[wr_addr+3] <= wr_data[31:24];

                    // Signal to GPU
                    weight_load_valid <= 1'b1;
                    weight_load_addr  <= wr_addr;
                    weight_load_data  <= wr_data[DATA_WIDTH-1:0];
                    weight_count      <= weight_count + 1;

                end else if (wr_addr == 16'h1000) begin
                    // Control register
                    start_inference <= wr_data[0];
                end
                // Clear and set response
                wr_addr_valid <= 1'b0;
                wr_data_valid <= 1'b0;
                s_axi_bvalid  <= 1'b1;
            end

            // ---- Write Response Handshake ----
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            // ---- AXI Read ----
            if (s_axi_arvalid && !s_axi_rvalid) begin
                s_axi_arready <= 1'b1;
                s_axi_rvalid  <= 1'b1;

                if (s_axi_araddr < MEM_DEPTH) begin
                    // Read weight memory (4 bytes packed)
                    s_axi_rdata <= {
                        (s_axi_araddr+3 < MEM_DEPTH) ?
                            weight_mem[s_axi_araddr+3] : 8'd0,
                        (s_axi_araddr+2 < MEM_DEPTH) ?
                            weight_mem[s_axi_araddr+2] : 8'd0,
                        (s_axi_araddr+1 < MEM_DEPTH) ?
                            weight_mem[s_axi_araddr+1] : 8'd0,
                        weight_mem[s_axi_araddr]
                    };
                end else begin
                    case (s_axi_araddr)
                        16'h1004: s_axi_rdata <= {30'd0, inference_done, inference_busy};
                        16'h1008: s_axi_rdata <= weight_count;
                        16'h100C: s_axi_rdata <= zero_skip_count;
                        default:  s_axi_rdata <= 32'hDEAD_BEEF;
                    endcase
                end
            end else begin
                s_axi_arready <= 1'b0;
            end

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;

            // ---- GPU read port ----
            if (gpu_read_en) begin
                gpu_read_data  <= weight_mem[gpu_read_addr];
                gpu_read_valid <= 1'b1;
            end
        end
    end

endmodule
