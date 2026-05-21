`timescale 1ns/1ps

module inv_sqrt_lut_256 (
    input  wire        clk,
    input  wire [7:0]  addr,
    output reg  [15:0] dout
);
    // Simple mock LUT for Reciprocal Square Root
    always @(posedge clk) begin
        if (addr == 8'd0) 
            dout <= 16'h7FFF;
        else
            dout <= 16'h7FFF / addr; // Mock computation
    end
endmodule