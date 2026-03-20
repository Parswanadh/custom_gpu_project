// ============================================================================
// Module: axi_weight_memory
// Description: AXI4-Lite slave interface for loading weights from external
//   memory (CPU, DMA, or SoC bus) into the GPU's weight storage.
//
//   FIXES APPLIED:
//     - Issue #16: Parity bits on weight memory with error flag
//     - Issue #18: Parity error exposed via status register at 0x1010
//
//   Address Map:
//     0x0000 - 0x0FFF: Weight memory (4096 x 8-bit = 4KB)
//     0x1000:          Control register (bit 0 = start inference)
//     0x1004:          Status register (bit 0 = busy, bit 1 = done, bit 2 = parity_error)
//     0x1008:          Weight count register
//     0x100C:          Zero-skip count (read-only)
//     0x1010:          Parity error count (read-only)
//
// AXI4-Lite: 32-bit address, 32-bit data, no bursts
// ============================================================================
module axi_weight_memory #(
    parameter MEM_DEPTH    = 4096,
    parameter DATA_WIDTH   = 8,
    parameter AXI_ADDR_W   = 16,
    parameter AXI_DATA_W   = 32
)(
    input  wire                     aclk,
    input  wire                     aresetn,

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

    // GPU-side interface: bulk write port
    output reg                      weight_load_valid,
    output reg  [$clog2(MEM_DEPTH)-1:0] weight_load_addr,
    output reg  [DATA_WIDTH-1:0]    weight_load_data,

    // Control/Status
    output reg                      start_inference,
    input  wire                     inference_busy,
    input  wire                     inference_done,
    input  wire [31:0]              zero_skip_count,

    // Error flag (Issue #16)
    output reg                      parity_error_out
);

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    // Weight memory with parity (Issue #16)
    reg [DATA_WIDTH-1:0] weight_mem [0:MEM_DEPTH-1];
    reg                  weight_par [0:MEM_DEPTH-1];
    reg [31:0]           weight_count;
    reg [31:0]           parity_error_count;

    // AXI write handling
    reg [AXI_ADDR_W-1:0] wr_addr;
    reg                   wr_addr_valid;
    reg                   wr_data_valid;
    reg [AXI_DATA_W-1:0]  wr_data;
    reg [3:0]             wr_strb;

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
            wr_strb          <= 4'b0;
            start_inference  <= 1'b0;
            weight_count     <= 0;
            weight_load_valid <= 1'b0;
            gpu_read_valid   <= 1'b0;
            parity_error_out <= 1'b0;
            parity_error_count <= 0;

            for (i = 0; i < MEM_DEPTH; i = i + 1) begin
                weight_mem[i] <= 0;
                weight_par[i] <= 1'b0;
            end

        end else begin
            weight_load_valid <= 1'b0;
            gpu_read_valid    <= 1'b0;
            start_inference   <= 1'b0;

            // ---- AXI Write Address ----
            // AXI-Lite allows only one outstanding write response. Do not
            // accept a new write transaction while BVALID is pending.
            if (s_axi_awvalid && !wr_addr_valid && !s_axi_bvalid) begin
                wr_addr       <= s_axi_awaddr;
                wr_addr_valid <= 1'b1;
                s_axi_awready <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
            end

            // ---- AXI Write Data ----
            if (s_axi_wvalid && !wr_data_valid && !s_axi_bvalid) begin
                wr_data       <= s_axi_wdata;
                wr_strb       <= s_axi_wstrb;
                wr_data_valid <= 1'b1;
                s_axi_wready  <= 1'b1;
            end else begin
                s_axi_wready <= 1'b0;
            end

            // ---- Complete Write ----
            if (wr_addr_valid && wr_data_valid) begin
                if (wr_addr < MEM_DEPTH) begin
                    // Weight memory write with parity (Issue #16)
                    if (wr_strb[0]) begin
                        weight_mem[wr_addr]   <= wr_data[7:0];
                        weight_par[wr_addr]   <= ^wr_data[7:0];
                    end
                    if (wr_strb[1] && wr_addr+1 < MEM_DEPTH) begin
                        weight_mem[wr_addr+1] <= wr_data[15:8];
                        weight_par[wr_addr+1] <= ^wr_data[15:8];
                    end
                    if (wr_strb[2] && wr_addr+2 < MEM_DEPTH) begin
                        weight_mem[wr_addr+2] <= wr_data[23:16];
                        weight_par[wr_addr+2] <= ^wr_data[23:16];
                    end
                    if (wr_strb[3] && wr_addr+3 < MEM_DEPTH) begin
                        weight_mem[wr_addr+3] <= wr_data[31:24];
                        weight_par[wr_addr+3] <= ^wr_data[31:24];
                    end

                    weight_load_valid <= 1'b1;
                    weight_load_addr  <= wr_addr;
                    weight_load_data  <= wr_data[DATA_WIDTH-1:0];
                    weight_count      <= weight_count + 1;

                end else if (wr_addr == 16'h1000) begin
                    start_inference <= wr_data[0];
                end
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
                        16'h1004: s_axi_rdata <= {29'd0, parity_error_out, inference_done, inference_busy};
                        16'h1008: s_axi_rdata <= weight_count;
                        16'h100C: s_axi_rdata <= zero_skip_count;
                        16'h1010: s_axi_rdata <= parity_error_count;
                        default:  s_axi_rdata <= 32'hDEAD_BEEF;
                    endcase
                end
            end else begin
                s_axi_arready <= 1'b0;
            end

            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;

            // ---- GPU read port with parity check (Issue #16) ----
            if (gpu_read_en) begin
                gpu_read_data  <= weight_mem[gpu_read_addr];
                gpu_read_valid <= 1'b1;
                // Check parity on read
                if (^weight_mem[gpu_read_addr] != weight_par[gpu_read_addr]) begin
                    parity_error_out   <= 1'b1;
                    parity_error_count <= parity_error_count + 1;
                end
            end
        end
    end

endmodule
