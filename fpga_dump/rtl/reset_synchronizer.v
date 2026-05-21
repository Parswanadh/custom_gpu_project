// ============================================================================
// Module: reset_synchronizer
// Description: 2-FF reset synchronizer for safe async-to-sync reset (Issue #6).
//   Takes an asynchronous reset input and produces a clean synchronous reset
//   output that is asserted asynchronously but de-asserted synchronously.
//   This prevents metastability on reset release.
//
// Usage: One instance per clock domain, placed at the top level.
// ============================================================================
`timescale 1ns / 1ps

module reset_synchronizer (
    input  wire clk,
    input  wire rst_async_n,    // Active-low asynchronous reset
    output wire rst_sync        // Active-high synchronous reset
);

    reg rst_ff1, rst_ff2;

    always @(posedge clk or negedge rst_async_n) begin
        if (!rst_async_n) begin
            rst_ff1 <= 1'b1;
            rst_ff2 <= 1'b1;
        end else begin
            rst_ff1 <= 1'b0;
            rst_ff2 <= rst_ff1;
        end
    end

    assign rst_sync = rst_ff2;

endmodule

