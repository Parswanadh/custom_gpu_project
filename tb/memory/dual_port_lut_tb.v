`timescale 1ns/1ps

module dual_port_lut_tb;

    parameter ADDR_WIDTH = 4;
    parameter DATA_WIDTH = 16;
    parameter DEPTH = 16;

    reg clk;
    reg we;
    reg [ADDR_WIDTH-1:0] waddr;
    reg [DATA_WIDTH-1:0] din;
    reg [ADDR_WIDTH-1:0] raddr1;
    reg [ADDR_WIDTH-1:0] raddr2;
    wire [DATA_WIDTH-1:0] dout1;
    wire [DATA_WIDTH-1:0] dout2;

    // Instantiate UUT
    dual_port_lut #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) uut (
        .clk(clk),
        .we(we),
        .waddr(waddr),
        .din(din),
        .raddr1(raddr1),
        .dout1(dout1),
        .raddr2(raddr2),
        .dout2(dout2)
    );

    // Clock Generation
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;
    initial begin
        // Initialize signals
        we = 0;
        waddr = 0;
        din = 0;
        raddr1 = 0;
        raddr2 = 0;

        #(50);
        @(posedge clk);

        // Write sequence
        $display("Starting memory write sequence...");
        for (i = 0; i < DEPTH; i = i + 1) begin
            @(negedge clk); // Drive on negedge for max setup time
            we = 1;
            waddr = i;
            din = 16'hA000 + i;
            $display("Writing: addr=%h, data=%h", waddr, din);
        end
        
        @(negedge clk);
        we = 0;
        $display("Initialization complete.");

        repeat(5) @(posedge clk);

        // Check 1
        @(negedge clk);
        raddr1 = 4'h0;
        raddr2 = 4'hF;
        
        repeat(2) @(posedge clk); // 1st edge: sample, 2nd edge: data out
        #1;
        $display("Check 1: raddr1=0 (val=%h), raddr2=F (val=%h)", dout1, dout2);
        if (dout1 === 16'hA000 && dout2 === 16'hA00F) begin
            $display("PASS: First read check.");
        end else begin
            $display("FAIL: First read check. Got %h and %h", dout1, dout2);
        end

        // Check 2
        @(negedge clk);
        raddr1 = 4'h5;
        raddr2 = 4'hA;
        
        repeat(2) @(posedge clk);
        #1;
        $display("Check 2: raddr1=5 (val=%h), raddr2=A (val=%h)", dout1, dout2);
        if (dout1 === 16'hA005 && dout2 === 16'hA00A) begin
            $display("PASS: Second read check.");
        end else begin
            $display("FAIL: Second read check. Got %h and %h", dout1, dout2);
        end

        $finish;
    end

endmodule
