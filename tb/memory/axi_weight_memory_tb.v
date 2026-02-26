// ============================================================================
// Testbench: axi_weight_memory_tb
// Tests the AXI4-Lite interface for weight storage
// ============================================================================
`timescale 1ns/1ps

module axi_weight_memory_tb;

    parameter MEM_DEPTH  = 256;
    parameter DATA_WIDTH = 8;
    parameter AXI_ADDR_W = 16;
    parameter AXI_DATA_W = 32;

    reg clk, rstn;

    // AXI signals
    reg         awvalid, wvalid, bready, arvalid, rready;
    wire        awready, wready, bvalid, arready, rvalid;
    reg  [AXI_ADDR_W-1:0] awaddr, araddr;
    reg  [AXI_DATA_W-1:0] wdata;
    reg  [3:0]  wstrb;
    wire [AXI_DATA_W-1:0] rdata;
    wire [1:0]  bresp, rresp;

    // GPU side
    reg         gpu_re;
    reg  [7:0]  gpu_raddr;
    wire [7:0]  gpu_rdata;
    wire        gpu_rvalid;
    wire        wl_valid;
    wire [7:0]  wl_addr;
    wire [7:0]  wl_data;
    wire        start_inf;

    axi_weight_memory #(
        .MEM_DEPTH(MEM_DEPTH), .DATA_WIDTH(DATA_WIDTH),
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)
    ) uut (
        .aclk(clk), .aresetn(rstn),
        .s_axi_awvalid(awvalid), .s_axi_awready(awready), .s_axi_awaddr(awaddr),
        .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb),
        .s_axi_bvalid(bvalid), .s_axi_bready(bready), .s_axi_bresp(bresp),
        .s_axi_arvalid(arvalid), .s_axi_arready(arready), .s_axi_araddr(araddr),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .gpu_read_en(gpu_re), .gpu_read_addr(gpu_raddr),
        .gpu_read_data(gpu_rdata), .gpu_read_valid(gpu_rvalid),
        .weight_load_valid(wl_valid), .weight_load_addr(wl_addr),
        .weight_load_data(wl_data), .start_inference(start_inf),
        .inference_busy(1'b0), .inference_done(1'b0), .zero_skip_count(32'd42)
    );

    always #5 clk = ~clk;

    // AXI write task
    task axi_write;
        input [AXI_ADDR_W-1:0] addr;
        input [AXI_DATA_W-1:0] data;
        input [3:0] strb;
        begin
            @(posedge clk);
            awaddr <= addr; awvalid <= 1;
            wdata <= data; wstrb <= strb; wvalid <= 1;
            bready <= 1;
            // Wait for both channels
            fork
                begin wait(awready); @(posedge clk); awvalid <= 0; end
                begin wait(wready);  @(posedge clk); wvalid <= 0;  end
            join
            wait(bvalid);
            @(posedge clk);
            bready <= 0;
        end
    endtask

    // AXI read task
    task axi_read;
        input  [AXI_ADDR_W-1:0] addr;
        output [AXI_DATA_W-1:0] data;
        begin
            @(posedge clk);
            araddr <= addr; arvalid <= 1; rready <= 1;
            wait(arready); @(posedge clk); arvalid <= 0;
            wait(rvalid);
            data = rdata;
            @(posedge clk); rready <= 0;
        end
    endtask

    reg [31:0] read_val;
    integer test_pass;

    initial begin
        $display("");
        $display("================================================================");
        $display("  AXI4-Lite Weight Memory Testbench");
        $display("================================================================");

        clk = 0; rstn = 0;
        awvalid = 0; wvalid = 0; bready = 0;
        arvalid = 0; rready = 0;
        gpu_re = 0;
        test_pass = 1;

        #30; rstn = 1; #10;

        // --- Write 4 weights via AXI ---
        $display("[1] Writing weights via AXI...");
        axi_write(16'h0000, 32'hDEADBEEF, 4'hF);
        $display("    Wrote 0x%08h to addr 0x0000", 32'hDEADBEEF);

        axi_write(16'h0004, 32'h12345678, 4'hF);
        $display("    Wrote 0x%08h to addr 0x0004", 32'h12345678);

        #20;

        // --- Read back via AXI ---
        $display("[2] Reading weights back via AXI...");
        axi_read(16'h0000, read_val);
        $display("    Read from 0x0000: 0x%08h  (expected 0xDEADBEEF)", read_val);
        if (read_val !== 32'hDEADBEEF) begin
            $display("    [FAIL] Mismatch!"); test_pass = 0;
        end else $display("    [PASS]");

        axi_read(16'h0004, read_val);
        $display("    Read from 0x0004: 0x%08h  (expected 0x12345678)", read_val);
        if (read_val !== 32'h12345678) begin
            $display("    [FAIL] Mismatch!"); test_pass = 0;
        end else $display("    [PASS]");

        // --- Read status registers ---
        $display("[3] Reading control/status registers...");
        axi_read(16'h1008, read_val);
        $display("    Weight count: %0d", read_val);

        axi_read(16'h100C, read_val);
        $display("    Zero-skip count: %0d  (expected 42)", read_val);
        if (read_val !== 32'd42) begin
            $display("    [FAIL]"); test_pass = 0;
        end else $display("    [PASS]");

        // --- GPU-side read ---
        $display("[4] GPU-side weight read...");
        @(posedge clk);
        gpu_re <= 1; gpu_raddr <= 8'h00;
        @(posedge clk);
        gpu_re <= 0;
        @(posedge clk);
        $display("    GPU read addr 0x00: 0x%02h (expected 0xEF)", gpu_rdata);
        if (gpu_rdata !== 8'hEF) begin
            $display("    [FAIL]"); test_pass = 0;
        end else $display("    [PASS]");

        // --- Start inference signal ---
        $display("[5] Testing start inference control...");
        axi_write(16'h1000, 32'h00000001, 4'hF);
        @(posedge clk);
        $display("    start_inference = %b (expected 1)", start_inf);

        // --- Summary ---
        $display("");
        $display("================================================================");
        if (test_pass)
            $display("  === ALL AXI TESTS PASSED ===");
        else
            $display("  === SOME AXI TESTS FAILED ===");
        $display("================================================================");
        $finish;
    end

endmodule
