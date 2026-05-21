// ============================================================================
// Module: memory_bridge
// Description: Maps linear addresses to 128-banked structure using bit-shuffling.
// ============================================================================
`timescale 1ns / 1ps

module memory_bridge #(
    parameter ADDR_WIDTH = 16,
    parameter BANK_WIDTH = 7   // 128 banks = 2^7
)(
    input  wire                  clk,
    input  wire                  rst_n,
    
    // Linear interface
    input  wire [ADDR_WIDTH-1:0] addr_in,
    input  wire                  en_in,
    
    // Banked interface
    output wire [BANK_WIDTH-1:0] bank_out,
    output wire [ADDR_WIDTH-BANK_WIDTH-1:0] offset_out,
    output wire                  en_out
);

    // Bit-shuffle logic: use middle bits for bank selection
    assign bank_out   = addr_in[BANK_WIDTH:1];
    assign offset_out = {addr_in[ADDR_WIDTH-1:BANK_WIDTH+1], addr_in[0]};
    assign en_out     = en_in;

endmodule
