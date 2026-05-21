// Dual-Port LUT for RoPE Sine/Cosine Lookups
// Supports two independent read ports and one write port for initialization.
// Optimized for SOTA Edge AI high-throughput parallel access.

module dual_port_lut #(
    parameter ADDR_WIDTH = 10,
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024
)(
    input  wire                   clk,
    
    // Write Port (Initialization)
    input  wire                   we,
    input  wire [ADDR_WIDTH-1:0]  waddr,
    input  wire [DATA_WIDTH-1:0]  din,
    
    // Read Port 1
    input  wire [ADDR_WIDTH-1:0]  raddr1,
    output reg  [DATA_WIDTH-1:0]  dout1,
    
    // Read Port 2
    input  wire [ADDR_WIDTH-1:0]  raddr2,
    output reg  [DATA_WIDTH-1:0]  dout2
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Initialization for simulation
    integer k;
    initial begin
        for (k = 0; k < DEPTH; k = k + 1) begin
            mem[k] = 0;
        end
    end

    // Synchronous Logic
    always @(posedge clk) begin
        // Write access
        if (we) begin
            mem[waddr] <= din;
        end
        
        // Simultaneous Dual Read access
        // Sample addresses and output data on next edge
        dout1 <= mem[raddr1];
        dout2 <= mem[raddr2];
    end

endmodule
