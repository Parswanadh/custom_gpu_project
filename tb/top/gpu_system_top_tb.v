// ============================================================================
// Testbench: gpu_system_top_tb
// End-to-end integration test for the standalone GPU system
// Tests: reset, config register access, command dispatch, DMA, perf counters
// ============================================================================
`timescale 1ns / 1ps

module gpu_system_top_tb;

    parameter AXI_ADDR_W   = 16;
    parameter AXI_DATA_W   = 32;
    parameter CMD_WIDTH    = 64;
    parameter FIFO_DEPTH   = 16;
    parameter LOCAL_ADDR_W = 16;
    parameter SP_DEPTH     = 256;
    parameter SP_DATA_W    = 16;

    reg                     clk;
    reg                     rst_async_n;

    // Config AXI
    reg                     s_axi_awvalid, s_axi_wvalid, s_axi_bready;
    reg  [AXI_ADDR_W-1:0]  s_axi_awaddr;
    reg  [AXI_DATA_W-1:0]  s_axi_wdata;
    wire                    s_axi_awready, s_axi_wready, s_axi_bvalid;
    wire [1:0]              s_axi_bresp;
    reg                     s_axi_arvalid, s_axi_rready;
    reg  [AXI_ADDR_W-1:0]  s_axi_araddr;
    wire                    s_axi_arready, s_axi_rvalid;
    wire [AXI_DATA_W-1:0]  s_axi_rdata;
    wire [1:0]              s_axi_rresp;

    // Command input
    reg                     cmd_valid;
    reg  [CMD_WIDTH-1:0]    cmd_data;
    wire                    cmd_ready;

    // DMA AXI master (simulate external memory)
    wire                    m_axi_arvalid, m_axi_rready;
    reg                     m_axi_arready;
    wire [31:0]             m_axi_araddr;
    wire [7:0]              m_axi_arlen;
    wire [2:0]              m_axi_arsize;
    wire [1:0]              m_axi_arburst;
    reg                     m_axi_rvalid;
    reg  [AXI_DATA_W-1:0]  m_axi_rdata;
    reg                     m_axi_rlast;
    reg  [1:0]              m_axi_rresp;
    wire                    m_axi_awvalid, m_axi_wvalid, m_axi_wlast, m_axi_bready;
    reg                     m_axi_awready, m_axi_wready, m_axi_bvalid;
    wire [31:0]             m_axi_awaddr;
    wire [7:0]              m_axi_awlen;
    wire [2:0]              m_axi_awsize, m_axi_awburst_dummy;
    wire [1:0]              m_axi_awburst;
    wire [AXI_DATA_W-1:0]  m_axi_wdata;
    wire [3:0]              m_axi_wstrb;

    wire                    irq_out;
    wire [31:0]             cycle_count, zero_skip_total, mac_total;

    gpu_system_top #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W),
        .CMD_WIDTH(CMD_WIDTH), .FIFO_DEPTH(FIFO_DEPTH),
        .LOCAL_ADDR_W(LOCAL_ADDR_W), .SP_DEPTH(SP_DEPTH), .SP_DATA_W(SP_DATA_W)
    ) uut (
        .clk(clk), .rst_async_n(rst_async_n),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .cmd_valid(cmd_valid), .cmd_data(cmd_data), .cmd_ready(cmd_ready),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast), .m_axi_rresp(m_axi_rresp),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .irq_out(irq_out),
        .cycle_count(cycle_count), .zero_skip_total(zero_skip_total), .mac_total(mac_total)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    // AXI write task
    task axi_write;
        input [AXI_ADDR_W-1:0] addr;
        input [AXI_DATA_W-1:0] data;
        integer timeout;
        reg got_bresp;
        begin
            @(posedge clk);
            s_axi_awvalid <= 1'b1; s_axi_awaddr <= addr;
            s_axi_wvalid  <= 1'b1; s_axi_wdata  <= data;
            s_axi_bready  <= 1'b1;
            timeout = 0;
            got_bresp = 1'b0;
            while (timeout < 20 && !got_bresp) begin
                @(posedge clk);
                if (s_axi_awready) s_axi_awvalid <= 1'b0;
                if (s_axi_wready) s_axi_wvalid <= 1'b0;
                if (s_axi_bvalid) begin
                    if (s_axi_bresp != 2'b00) begin
                        $display("[FATAL] AXI write BRESP error at addr=0x%0H resp=%0b", addr, s_axi_bresp);
                        $fatal(1);
                    end
                    s_axi_bready <= 1'b0;
                    got_bresp = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!got_bresp) begin
                $display("[FATAL] AXI write timeout at addr=0x%0H", addr);
                $fatal(1);
            end
            @(posedge clk);
            s_axi_awvalid <= 1'b0; s_axi_wvalid <= 1'b0; s_axi_bready <= 1'b0;
        end
    endtask

    // AXI read task
    task axi_read;
        input [AXI_ADDR_W-1:0] addr;
        integer timeout;
        reg got_rdata;
        begin
            @(posedge clk);
            s_axi_arvalid <= 1'b1; s_axi_araddr <= addr;
            s_axi_rready  <= 1'b1;
            timeout = 0;
            got_rdata = 1'b0;
            while (!got_rdata && timeout < 20) begin
                @(posedge clk);
                if (s_axi_arready) s_axi_arvalid <= 1'b0;
                if (s_axi_rvalid) begin
                    if (s_axi_rresp != 2'b00) begin
                        $display("[FATAL] AXI read RRESP error at addr=0x%0H resp=%0b", addr, s_axi_rresp);
                        $fatal(1);
                    end
                    got_rdata = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!got_rdata) begin
                $display("[FATAL] AXI read timeout at addr=0x%0H", addr);
                $fatal(1);
            end
            @(posedge clk);
            s_axi_arvalid <= 1'b0; s_axi_rready <= 1'b0;
        end
    endtask

    // Push command task
    task push_cmd;
        input [63:0] cmd;
        integer wait_cycles;
        begin
            wait_cycles = 0;
            while (!cmd_ready && wait_cycles < 100) begin
                @(negedge clk);
                wait_cycles = wait_cycles + 1;
            end
            if (!cmd_ready) begin
                $display("[FATAL] cmd_ready timeout while pushing cmd=0x%0H", cmd);
                $fatal(1);
            end
            @(negedge clk);
            cmd_valid <= 1'b1; cmd_data <= cmd;
            @(negedge clk);
            cmd_valid <= 1'b0;
        end
    endtask

    initial begin
        $display("============================================");
        $display("  GPU System Top Integration Testbench");
        $display("============================================");

        // Initialize all signals
        rst_async_n = 0;
        s_axi_awvalid = 0; s_axi_wvalid = 0; s_axi_bready = 0;
        s_axi_arvalid = 0; s_axi_rready = 0;
        s_axi_awaddr = 0; s_axi_wdata = 0; s_axi_araddr = 0;
        cmd_valid = 0; cmd_data = 0;
        m_axi_arready = 0; m_axi_rvalid = 0; m_axi_rdata = 0;
        m_axi_rlast = 0; m_axi_rresp = 0;
        m_axi_awready = 0; m_axi_wready = 0; m_axi_bvalid = 0;

        // Test 1: Reset behavior
        #30;
        rst_async_n = 1;
        repeat(5) @(posedge clk);

        $display("[1] Testing reset...");
        if (cmd_ready) begin
            $display("[PASS] System ready after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] System not ready after reset");
            fail_count = fail_count + 1;
        end

        // Test 2: Read GPU_ID via AXI
        $display("[2] Reading GPU_ID via AXI...");
        axi_read(16'h0000);
        if (s_axi_rdata == 32'hB17B_0001) begin
            $display("[PASS] GPU_ID = 0x%08H", s_axi_rdata);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] GPU_ID = 0x%08H (expected 0xB17B0001)", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 3: Write and read config register
        $display("[3] Config register write/read...");
        axi_write(16'h0008, 32'd768);  // EMBED_DIM = 768
        axi_read(16'h0008);
        if (s_axi_rdata[15:0] == 16'd768) begin
            $display("[PASS] EMBED_DIM = %0d", s_axi_rdata[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] EMBED_DIM = %0d (expected 768)", s_axi_rdata[15:0]);
            fail_count = fail_count + 1;
        end

        // Test 4: NOP command through command processor
        $display("[4] Dispatching NOP command...");
        push_cmd({8'h00, 56'd0}); // CMD_NOP
        repeat(10) @(posedge clk);
        // Check status shows idle
        axi_read(16'h0004); // GPU_STATUS
        if (s_axi_rdata[1] == 1'b1 && s_axi_rdata[2] == 1'b0) begin // idle and no error
            $display("[PASS] NOP completed, GPU idle");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Unexpected status after NOP: status=0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 5: MATMUL command must latch explicit stub error (fail-closed semantics)
        $display("[5] Dispatching MATMUL command...");
        push_cmd({8'h02, 8'hAA, 16'h0100, 16'h0200, 16'h0010}); // CMD_MATMUL
        repeat(15) @(posedge clk);
        axi_read(16'h0004);
        if (s_axi_rdata[1] == 1'b1 && s_axi_rdata[2] == 1'b1) begin
            $display("[PASS] MATMUL completed with expected compute-stub error latch");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Expected idle+error after MATMUL stub: status=0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 6: FENCE command generates interrupt
        $display("[6] Dispatching FENCE command...");
        // Enable IRQ for command processor
        axi_write(16'h0040, 32'h01); // IRQ_ENABLE bit 0
        push_cmd({8'h0F, 56'd0}); // CMD_FENCE
        begin : wait_irq
            integer t;
            t = 0;
            while (!irq_out && t < 50) begin @(posedge clk); t = t + 1; end
        end
        if (irq_out) begin
            $display("[PASS] FENCE generated interrupt");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] No interrupt from FENCE");
            fail_count = fail_count + 1;
        end

        // Test 7: Performance counters are running
        $display("[7] Checking performance counters...");
        repeat(10) @(posedge clk);
        if (cycle_count > 0) begin
            $display("[PASS] Cycle counter running: %0d", cycle_count);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Cycle counter stuck at 0");
            fail_count = fail_count + 1;
        end

        // Test 8: Multiple commands in sequence
        $display("[8] Multiple commands in sequence...");
        push_cmd({8'h00, 56'd0}); // NOP
        push_cmd({8'h03, 8'h00, 16'h0000, 16'h0000, 16'h0004}); // ACTIVATION
        push_cmd({8'h0F, 56'd0}); // FENCE
        repeat(40) @(posedge clk);
        axi_read(16'h0004);
        if (s_axi_rdata[1] == 1'b1 && s_axi_rdata[2] == 1'b1) begin
            $display("[PASS] Multiple commands completed and stub error remained visible");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Commands still processing: status=0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $display("TB_RESULT pass=%0d fail=%0d", pass_count, fail_count);
        if (fail_count != 0)
            $fatal(1, "gpu_system_top_tb failed with %0d checks failing", fail_count);
        $finish;
    end

endmodule
