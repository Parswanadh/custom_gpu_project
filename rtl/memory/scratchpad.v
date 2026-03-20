// ============================================================================
// Module: scratchpad
// Description: Dual-port SRAM scratchpad for intermediate activations (Issue #19).
//   Replaces wide wire buses between transformer stages.
//   Port A: Read/Write from compute pipeline
//   Port B: Read/Write from DMA or command processor
//
// Parameters: DEPTH (number of 16-bit words), ADDR_W
// ============================================================================
`timescale 1ns / 1ps

module scratchpad #(
    parameter DEPTH  = 4096,        // 4K words × 16 bits = 8KB
    parameter DATA_W = 16,
    parameter ADDR_W = $clog2(DEPTH)
)(
    input  wire                clk,
    input  wire                rst,

    // Port A: Compute pipeline
    input  wire                a_read_en,
    input  wire [ADDR_W-1:0]  a_read_addr,
    output reg  [DATA_W-1:0]  a_read_data,
    output reg                a_read_valid,
    input  wire                a_write_en,
    input  wire [ADDR_W-1:0]  a_write_addr,
    input  wire [DATA_W-1:0]  a_write_data,

    // Port B: DMA / Command processor
    input  wire                b_read_en,
    input  wire [ADDR_W-1:0]  b_read_addr,
    output reg  [DATA_W-1:0]  b_read_data,
    output reg                b_read_valid,
    input  wire                b_write_en,
    input  wire [ADDR_W-1:0]  b_write_addr,
    input  wire [DATA_W-1:0]  b_write_data,

    // Status
    output wire [ADDR_W:0]     usage_count  // Not tracked in simple SRAM
);

    // SRAM storage
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    integer i;

    assign usage_count = DEPTH;  // Always "full" capacity available

    // Dual-port behavior with deterministic same-address write policy:
    // if both ports write same address in the same cycle, Port B wins.
    always @(posedge clk) begin
        if (rst) begin
            a_read_data  <= {DATA_W{1'b0}};
            a_read_valid <= 1'b0;
            b_read_data  <= {DATA_W{1'b0}};
            b_read_valid <= 1'b0;
        end else begin
            a_read_valid <= 1'b0;
            b_read_valid <= 1'b0;

            if (a_write_en && !(b_write_en && (b_write_addr == a_write_addr))) begin
                mem[a_write_addr] <= a_write_data;
            end
            if (b_write_en) begin
                mem[b_write_addr] <= b_write_data;
            end

            if (a_read_en) begin
                a_read_data  <= mem[a_read_addr];
                a_read_valid <= 1'b1;
            end
            if (b_read_en) begin
                b_read_data  <= mem[b_read_addr];
                b_read_valid <= 1'b1;
            end
        end
    end

endmodule

