// ============================================================================
// Testbench: dma_engine_tb
// Tests DMA burst read transfers from external memory to local SRAM
// ============================================================================
`timescale 1ns / 1ps

module dma_engine_tb;

    parameter AXI_ADDR_W   = 32;
    parameter AXI_DATA_W   = 32;
    parameter LOCAL_ADDR_W = 16;
    parameter MAX_BURST    = 16;

    reg                     clk, rst;

    // Control
    reg                     start;
    reg  [AXI_ADDR_W-1:0]  ext_addr;
    reg  [LOCAL_ADDR_W-1:0] local_addr;
    reg  [15:0]             transfer_len;
    reg                     direction;
    wire                    done, busy;
    wire                    error;

    // AXI Read Address
    wire                    m_axi_arvalid;
    reg                     m_axi_arready;
    wire [AXI_ADDR_W-1:0]  m_axi_araddr;
    wire [7:0]              m_axi_arlen;
    wire [2:0]              m_axi_arsize;
    wire [1:0]              m_axi_arburst;

    // AXI Read Data
    reg                     m_axi_rvalid;
    wire                    m_axi_rready;
    reg  [AXI_DATA_W-1:0]  m_axi_rdata;
    reg                     m_axi_rlast;
    reg  [1:0]              m_axi_rresp;

    // AXI Write (not used for read tests)
    wire                    m_axi_awvalid, m_axi_wvalid, m_axi_wlast;
    reg                     m_axi_awready, m_axi_wready, m_axi_bvalid;
    wire [AXI_ADDR_W-1:0]  m_axi_awaddr;
    wire [7:0]              m_axi_awlen;
    wire [2:0]              m_axi_awsize;
    wire [1:0]              m_axi_awburst;
    wire [AXI_DATA_W-1:0]  m_axi_wdata;
    wire [3:0]              m_axi_wstrb;
    wire                    m_axi_bready;

    // Local memory interface
    wire                    local_write_en;
    wire [LOCAL_ADDR_W-1:0] local_write_addr;
    wire [AXI_DATA_W-1:0]  local_write_data;
    wire                    local_read_en;
    wire [LOCAL_ADDR_W-1:0] local_read_addr;
    reg  [AXI_DATA_W-1:0]  local_read_data;

    wire                    interrupt;

    dma_engine #(
        .AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W),
        .LOCAL_ADDR_W(LOCAL_ADDR_W), .MAX_BURST(MAX_BURST)
    ) uut (
        .clk(clk), .rst(rst),
        .start(start), .ext_addr(ext_addr), .local_addr(local_addr),
        .transfer_len(transfer_len), .direction(direction),
        .done(done), .busy(busy), .error(error),
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
        .local_write_en(local_write_en), .local_write_addr(local_write_addr),
        .local_write_data(local_write_data),
        .local_read_en(local_read_en), .local_read_addr(local_read_addr),
        .local_read_data(local_read_data),
        .interrupt(interrupt)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer write_count;

    initial begin
        $display("============================================");
        $display("  DMA Engine Testbench");
        $display("============================================");

        rst = 1; start = 0; direction = 0;
        ext_addr = 0; local_addr = 0; transfer_len = 0;
        m_axi_arready = 0; m_axi_rvalid = 0; m_axi_rdata = 0;
        m_axi_rlast = 0; m_axi_rresp = 0;
        m_axi_awready = 0; m_axi_wready = 0; m_axi_bvalid = 0;
        local_read_data = 0;
        #25; rst = 0; #15;

        // Test 1: Idle state
        if (!busy && !done) begin
            $display("[PASS] DMA idle after reset");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] DMA not idle: busy=%b, done=%b", busy, done);
            fail_count = fail_count + 1;
        end

        // Test 2: Read transfer (ext→local, 16 bytes = 4 beats)
        $display("[2] Starting read transfer: 16 bytes from 0x1000 to local 0x0000...");
        @(posedge clk);
        start        <= 1'b1;
        ext_addr     <= 32'h0000_1000;
        local_addr   <= 16'h0000;
        transfer_len <= 16'd16;
        direction    <= 1'b0; // ext→local
        @(posedge clk);
        start <= 1'b0;

        // Wait for AR channel
        begin : wait_ar
            integer t;
            t = 0;
            while (!m_axi_arvalid && t < 20) begin @(posedge clk); t = t + 1; end
        end

        if (m_axi_arvalid) begin
            $display("    AR: addr=0x%08H, len=%0d", m_axi_araddr, m_axi_arlen);
            // Accept address
            @(posedge clk); m_axi_arready <= 1'b1;
            @(posedge clk); m_axi_arready <= 1'b0;

            // Send 4 data beats
            write_count = 0;
            begin : send_beats
                integer beat;
                for (beat = 0; beat < 4; beat = beat + 1) begin
                    @(posedge clk);
                    m_axi_rvalid <= 1'b1;
                    m_axi_rdata  <= 32'hAA00_0000 + beat;
                    m_axi_rlast  <= (beat == 3);
                    // Wait for rready
                    while (!m_axi_rready) @(posedge clk);
                    if (local_write_en) write_count = write_count + 1;
                end
            end
            @(posedge clk); m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;

            // Wait for done
            begin : wait_done
                integer t;
                t = 0;
                while (!done && t < 50) begin @(posedge clk); t = t + 1; end
            end

            if (done && interrupt) begin
                $display("[PASS] Read transfer complete, interrupt asserted");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Transfer not complete: done=%b, interrupt=%b", done, interrupt);
                fail_count = fail_count + 1;
            end
        end else begin
            $display("[FAIL] No AR request generated");
            fail_count = fail_count + 1;
        end

        // Test 3: Verify busy signal
        $display("[3] Testing busy signal...");
        @(posedge clk);
        start        <= 1'b1;
        ext_addr     <= 32'h0000_2000;
        local_addr   <= 16'h0100;
        transfer_len <= 16'd8;
        direction    <= 1'b0;
        @(posedge clk);
        start <= 1'b0;
        @(posedge clk);
        if (busy) begin
            $display("[PASS] DMA busy during transfer");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] DMA not busy during transfer");
            fail_count = fail_count + 1;
        end

        // Complete the transfer quickly
        begin : complete_xfer
            integer t;
            t = 0;
            while (!m_axi_arvalid && t < 20) begin @(posedge clk); t = t + 1; end
            @(posedge clk); m_axi_arready <= 1'b1;
            @(posedge clk); m_axi_arready <= 1'b0;
            // 2 beats
            begin : send2
                integer beat;
                for (beat = 0; beat < 2; beat = beat + 1) begin
                    @(posedge clk);
                    m_axi_rvalid <= 1'b1;
                    m_axi_rdata  <= 32'hBB000000 + beat;
                    m_axi_rlast  <= (beat == 1);
                    while (!m_axi_rready) @(posedge clk);
                end
            end
            @(posedge clk); m_axi_rvalid <= 1'b0; m_axi_rlast <= 1'b0;
            t = 0;
            while (!done && t < 50) begin @(posedge clk); t = t + 1; end
        end

        // Test 4: Not busy after completion
        repeat(2) @(posedge clk);
        if (!busy) begin
            $display("[PASS] DMA idle after transfer complete");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] DMA still busy after completion");
            fail_count = fail_count + 1;
        end

        // Test 5: Zero-length transfer guard
        $display("[5] Testing zero-length transfer guard...");
        begin : test_zero_length
            integer t;
            integer saw_done;
            integer saw_interrupt;
            integer saw_axi_traffic;

            saw_done = 0;
            saw_interrupt = 0;
            saw_axi_traffic = 0;

            @(negedge clk);
            start        <= 1'b1;
            ext_addr     <= 32'h0000_3000;
            local_addr   <= 16'h0200;
            transfer_len <= 16'd0;
            direction    <= 1'b0;
            @(negedge clk);
            start <= 1'b0;

            if (done) saw_done = 1;
            if (interrupt) saw_interrupt = 1;

            for (t = 0; t < 6; t = t + 1) begin
                @(posedge clk);
                if (done) saw_done = 1;
                if (interrupt) saw_interrupt = 1;
                if (m_axi_arvalid || m_axi_awvalid || m_axi_wvalid || m_axi_rready || m_axi_bready)
                    saw_axi_traffic = 1;
            end

            if (saw_done && saw_interrupt && !saw_axi_traffic && !busy) begin
                $display("[PASS] Zero-length transfer completes with no AXI traffic");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Zero-length guard failed: done=%0d intr=%0d axi=%0d busy=%b",
                         saw_done, saw_interrupt, saw_axi_traffic, busy);
                fail_count = fail_count + 1;
            end
        end

        // Test 6: Tail-byte write uses safe WSTRB policy
        $display("[6] Testing tail-byte write WSTRB policy...");
        begin : test_write_tail
            integer t;
            integer saw_wbeat;
            reg [3:0] sampled_wstrb;
            reg [31:0] sampled_wdata;
            reg sampled_wlast;

            saw_wbeat = 0;
            sampled_wstrb = 4'b0000;
            sampled_wdata = 32'd0;
            sampled_wlast = 1'b0;
            local_read_data = 32'hDDCC_BBAA;

            @(posedge clk);
            start        <= 1'b1;
            ext_addr     <= 32'h0000_4000;
            local_addr   <= 16'h0300;
            transfer_len <= 16'd3;
            direction    <= 1'b1; // local→ext
            @(posedge clk);
            start <= 1'b0;

            t = 0;
            while (!m_axi_awvalid && t < 20) begin @(posedge clk); t = t + 1; end
            if (!m_axi_awvalid || m_axi_awlen != 0) begin
                $display("[FAIL] Tail write did not issue single-beat AW (valid=%b len=%0d)", m_axi_awvalid, m_axi_awlen);
                fail_count = fail_count + 1;
            end else begin
                @(posedge clk); m_axi_awready <= 1'b1;
                @(posedge clk); m_axi_awready <= 1'b0;

                t = 0;
                while (!m_axi_wvalid && t < 20) begin @(posedge clk); t = t + 1; end
                if (m_axi_wvalid) begin
                    saw_wbeat = 1;
                    sampled_wstrb = m_axi_wstrb;
                    sampled_wdata = m_axi_wdata;
                    sampled_wlast = m_axi_wlast;
                    @(posedge clk); m_axi_wready <= 1'b1;
                    @(posedge clk); m_axi_wready <= 1'b0;
                end

                t = 0;
                while (!m_axi_bready && t < 20) begin @(posedge clk); t = t + 1; end
                if (m_axi_bready) begin
                    @(posedge clk); m_axi_bvalid <= 1'b1;
                    @(posedge clk); m_axi_bvalid <= 1'b0;
                end

                t = 0;
                while (!done && t < 40) begin @(posedge clk); t = t + 1; end

                if (saw_wbeat &&
                    sampled_wstrb == 4'b0111 &&
                    sampled_wdata == 32'h00CC_BBAA &&
                    sampled_wlast &&
                    done && interrupt) begin
                    $display("[PASS] Tail write uses masked data + WSTRB=0x7");
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Tail write policy failed: wvalid=%0d wstrb=0x%0h wdata=0x%08h wlast=%b done=%b intr=%b",
                             saw_wbeat, sampled_wstrb, sampled_wdata, sampled_wlast, done, interrupt);
                    fail_count = fail_count + 1;
                end
            end
        end

        // Test 7: Tail-byte read masks local write data
        $display("[7] Testing tail-byte read local write mask...");
        begin : test_read_tail
            integer t;
            reg sampled_local_write;
            reg [31:0] sampled_local_data;

            sampled_local_write = 1'b0;
            sampled_local_data = 32'd0;

            @(posedge clk);
            start        <= 1'b1;
            ext_addr     <= 32'h0000_5000;
            local_addr   <= 16'h0400;
            transfer_len <= 16'd3;
            direction    <= 1'b0; // ext→local
            @(posedge clk);
            start <= 1'b0;

            t = 0;
            while (!m_axi_arvalid && t < 20) begin @(posedge clk); t = t + 1; end
            if (!m_axi_arvalid || m_axi_arlen != 0) begin
                $display("[FAIL] Tail read did not issue single-beat AR (valid=%b len=%0d)", m_axi_arvalid, m_axi_arlen);
                fail_count = fail_count + 1;
            end else begin
                @(posedge clk); m_axi_arready <= 1'b1;
                @(posedge clk); m_axi_arready <= 1'b0;

                t = 0;
                while (!m_axi_rready && t < 20) begin @(posedge clk); t = t + 1; end

                @(negedge clk);
                m_axi_rvalid <= 1'b1;
                m_axi_rdata  <= 32'h1122_3344;
                m_axi_rlast  <= 1'b1;

                t = 0;
                while (!sampled_local_write && t < 20) begin
                    @(negedge clk);
                    if (local_write_en) begin
                        sampled_local_write = 1'b1;
                        sampled_local_data = local_write_data;
                    end
                    t = t + 1;
                end

                m_axi_rvalid <= 1'b0;
                m_axi_rlast  <= 1'b0;

                t = 0;
                while (!done && t < 40) begin @(posedge clk); t = t + 1; end

                if (sampled_local_write &&
                    sampled_local_data == 32'h0022_3344 &&
                    done && interrupt) begin
                    $display("[PASS] Tail read masks upper byte on local write");
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Tail read mask failed: wren=%b wdata=0x%08h done=%b intr=%b",
                             sampled_local_write, sampled_local_data, done, interrupt);
                    fail_count = fail_count + 1;
                end
            end
        end

        // Test 8: AXI read error response should fail-closed and assert error
        $display("[8] Testing AXI read error response handling...");
        begin : test_rresp_error
            integer t;
            @(posedge clk);
            start        <= 1'b1;
            ext_addr     <= 32'h0000_6000;
            local_addr   <= 16'h0500;
            transfer_len <= 16'd4;
            direction    <= 1'b0;
            @(posedge clk);
            start <= 1'b0;

            t = 0;
            while (!m_axi_arvalid && t < 20) begin @(posedge clk); t = t + 1; end
            @(posedge clk); m_axi_arready <= 1'b1;
            @(posedge clk); m_axi_arready <= 1'b0;

            @(posedge clk);
            m_axi_rvalid <= 1'b1;
            m_axi_rdata  <= 32'hCAFE_BABE;
            m_axi_rresp  <= 2'b10; // SLVERR
            m_axi_rlast  <= 1'b1;
            @(posedge clk);
            m_axi_rvalid <= 1'b0;
            m_axi_rresp  <= 2'b00;
            m_axi_rlast  <= 1'b0;

            t = 0;
            while (!done && t < 40) begin @(posedge clk); t = t + 1; end
            if (done && interrupt && error) begin
                $display("[PASS] AXI read error surfaces via dma error pulse");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] AXI read error not surfaced: done=%b intr=%b error=%b", done, interrupt, error);
                fail_count = fail_count + 1;
            end
        end

        // Test 9: Address-channel watchdog should terminate hung transfer
        $display("[9] Testing address-channel watchdog timeout...");
        begin : test_watchdog_timeout
            integer t;
            reg saw_done;
            reg saw_error;
            saw_done = 1'b0;
            saw_error = 1'b0;

            @(posedge clk);
            start        <= 1'b1;
            ext_addr     <= 32'h0000_7000;
            local_addr   <= 16'h0600;
            transfer_len <= 16'd4;
            direction    <= 1'b0;
            @(posedge clk);
            start <= 1'b0;

            for (t = 0; t < 4300; t = t + 1) begin
                @(posedge clk);
                if (done) saw_done = 1'b1;
                if (error) saw_error = 1'b1;
            end

            if (saw_done && saw_error && !busy) begin
                $display("[PASS] DMA watchdog timed out stalled transfer safely");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] DMA watchdog timeout failed: saw_done=%b saw_error=%b busy=%b",
                         saw_done, saw_error, busy);
                fail_count = fail_count + 1;
            end
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        if (fail_count != 0) begin
            $fatal(1, "dma_engine_tb failed");
        end
        $finish;
    end

endmodule
