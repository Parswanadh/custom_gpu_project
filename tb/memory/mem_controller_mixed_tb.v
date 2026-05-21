`timescale 1ns / 1ps

module mem_controller_mixed_tb;

    parameter ADDR_WIDTH = 12;
    parameter DATA_WIDTH = 32;
    parameter DEPTH      = 4096;

    reg                   clk;
    reg                   rst;
    reg                   req_valid;
    reg [ADDR_WIDTH-1:0]  req_addr;
    reg [1:0]             precision;
    wire                  req_ready;
    wire [15:0]           resp_data;
    wire                  resp_valid;

    // Instantiate DUT
    mem_controller_mixed #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .req_valid(req_valid),
        .req_addr(req_addr),
        .precision(precision),
        .req_ready(req_ready),
        .resp_data(resp_data),
        .resp_valid(resp_valid)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;
    
    initial begin
        $display("Starting Mixed-Precision Memory Controller Testbench...");
        
        // Reset
        rst = 1;
        req_valid = 0;
        req_addr = 0;
        precision = 0;
        #50;
        @(negedge clk);
        rst = 0;
        #20;

        // 1. Preload memory
        dut.preload_mem(0, 32'h87654321);
        dut.preload_mem(1, 32'hFEDCBA98);
        
        // 2. Test 4-bit precision
        $display("Testing 4-bit precision...");
        precision = 2'b00;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk);
            req_valid = 1;
            req_addr = i;
            
            // Wait for acceptance
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            
            @(negedge clk);
            req_valid = 0;
            
            // Wait for response
            while (!resp_valid) @(posedge clk);
            
            #1;
            $display("Addr %0d (4b): Expected %x, Got %x", i, (32'h87654321 >> (i*4)) & 4'hF, resp_data[3:0]);
            if (resp_data[3:0] !== ((32'h87654321 >> (i*4)) & 4'hF)) begin
                $display("FAIL: 4-bit precision mismatch at addr %0d", i);
                $finish;
            end
        end

        // 3. Test 8-bit precision
        $display("Testing 8-bit precision...");
        precision = 2'b01;
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            req_valid = 1;
            req_addr = i;
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            @(negedge clk);
            req_valid = 0;
            while (!resp_valid) @(posedge clk);
            #1;
            $display("Addr %0d (8b): Expected %x, Got %x", i, (32'h87654321 >> (i*8)) & 8'hFF, resp_data[7:0]);
            if (resp_data[7:0] !== ((32'h87654321 >> (i*8)) & 8'hFF)) begin
                $display("FAIL: 8-bit precision mismatch at addr %0d", i);
                $finish;
            end
        end

        // 4. Test 16-bit precision
        $display("Testing 16-bit precision...");
        precision = 2'b10;
        for (i = 0; i < 2; i = i + 1) begin
            @(negedge clk);
            req_valid = 1;
            req_addr = i + 2; 
            @(posedge clk);
            while (!req_ready) @(posedge clk);
            @(negedge clk);
            req_valid = 0;
            while (!resp_valid) @(posedge clk);
            #1;
            $display("Addr %0d (16b): Expected %x, Got %x", i+2, (32'hFEDCBA98 >> (i*16)) & 16'hFFFF, resp_data);
            if (resp_data !== ((32'hFEDCBA98 >> (i*16)) & 16'hFFFF)) begin
                $display("FAIL: 16-bit precision mismatch at addr %0d", i+2);
                $finish;
            end
        end

        $display("PASS: All mixed-precision fetch tests completed.");
        $finish;
    end

endmodule
