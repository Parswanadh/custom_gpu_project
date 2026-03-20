// ============================================================================
// Testbench: gpu_system_top_v2_tb
// End-to-end integration test for the standalone GPU system
// Tests: reset, config register access, command dispatch, DMA, perf counters
// ============================================================================
`timescale 1ns / 1ps

module gpu_system_top_v2_tb;

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

    gpu_system_top_v2 #(
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
    integer baseline_matmul_cycles = -1;
    integer mini_imprint_cycles = -1;
    integer gemma_imprint_cycles = -1;

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
                    s_axi_bready <= 1'b0;
                    got_bresp = 1'b1;
                end
                timeout = timeout + 1;
            end
            if (!got_bresp) begin
                $display("[FAIL] AXI write timeout at addr=0x%0H", addr);
                $fatal;
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
            while (timeout < 20 && !got_rdata) begin
                @(posedge clk);
                if (s_axi_arready) s_axi_arvalid <= 1'b0;
                if (s_axi_rvalid) got_rdata = 1'b1;
                timeout = timeout + 1;
            end
            if (!got_rdata) begin
                $display("[FAIL] AXI read timeout at addr=0x%0H", addr);
                $fatal;
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
                $display("[FAIL] cmd_ready timeout while pushing cmd=0x%0H", cmd);
                $fatal;
            end
            @(negedge clk);
            cmd_valid <= 1'b1; cmd_data <= cmd;
            @(negedge clk);
            cmd_valid <= 1'b0;
        end
    endtask

    // Measure command latency from internal compute dispatch to compute done.
    task measure_compute_latency;
        input [63:0] cmd;
        output integer measured_cycles;
        integer t;
        integer start_cycle_local;
        reg started;
        begin
            measured_cycles = -1;
            started = 1'b0;
            start_cycle_local = 0;
            push_cmd(cmd);
            for (t = 0; t < 500; t = t + 1) begin
                @(posedge clk);
                if (!started && uut.cp_compute_start) begin
                    started = 1'b1;
                    start_cycle_local = cycle_count;
                end
                if (started && uut.cp_compute_done) begin
                    measured_cycles = cycle_count - start_cycle_local;
                    t = 500;
                end
            end
            if (measured_cycles < 0) begin
                $display("[FAIL] compute latency measurement timeout for cmd=0x%0H", cmd);
                $fatal;
            end
        end
    endtask

    initial begin
        $display("============================================");
        $display("  GPU System Top V2 Integration Testbench");
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
        if (s_axi_rdata[1] == 1'b1) begin // idle bit
            $display("[PASS] NOP completed, GPU idle");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] GPU not idle after NOP: status=0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 5: MATMUL command goes through optimized transformer layer
        $display("[5] Dispatching MATMUL command via optimized pipeline...");
        axi_write(16'h0034, 32'd21);  // TOKEN_IN seed
        axi_write(16'h0038, 32'd3);   // POSITION_IN
        begin : wait_compute_done
            integer elapsed_cycles;
            reg [15:0] expected_dynamic0;
            reg [15:0] expected_dynamic1;
            reg completed;
            measure_compute_latency({8'h02, 8'hAA, 16'h0010, 16'h0020, 16'h0010}, elapsed_cycles); // CMD_MATMUL
            expected_dynamic0 = 16'd21 + 16'h0010 + 16'd0;
            expected_dynamic1 = 16'd21 + 16'h0010 + 16'd1;
            axi_read(16'h0004); // GPU_STATUS
            completed = s_axi_rdata[1];
            baseline_matmul_cycles = elapsed_cycles;
            if (completed && elapsed_cycles > 6 &&
                uut.opt_token_embedding[15:0] == expected_dynamic0 &&
                uut.opt_token_embedding[31:16] == expected_dynamic1) begin
                $display("[PASS] MATMUL completed via optimized path in %0d cycles", elapsed_cycles);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] MATMUL path invalid: completed=%0d elapsed=%0d status=0x%08H emb0=%0d emb1=%0d",
                         completed, elapsed_cycles, s_axi_rdata,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                fail_count = fail_count + 1;
            end
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
        repeat(120) @(posedge clk);
        axi_read(16'h0004);
        if (s_axi_rdata[1] == 1'b1) begin
            $display("[PASS] Multiple commands completed successfully");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Commands still processing: status=0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 9: Reject odd-byte DMA length (fail-closed, no AXI traffic side-effects)
        $display("[9] Verifying odd-byte DMA length is rejected...");
        begin : odd_dma_reject
            integer t;
            reg [15:0] pre_dma_dst0;
            reg [15:0] pre_dma_dst1;
            reg saw_arvalid;
            uut.u_scratchpad.mem[16'h0001] = 16'h1357;
            uut.u_scratchpad.mem[16'h0002] = 16'h2468;
            pre_dma_dst0 = uut.u_scratchpad.mem[16'h0001];
            pre_dma_dst1 = uut.u_scratchpad.mem[16'h0002];
            saw_arvalid = 1'b0;

            axi_read(16'h0004); // GPU_STATUS
            if (s_axi_rdata[2]) begin
                $display("[FAIL] status_error already set before odd DMA test: status=0x%08H", s_axi_rdata);
                fail_count = fail_count + 1;
            end else begin
                // CMD_LOAD_WEIGHTS, src=0x1000, dst=0x0002, len=3 (odd bytes => reject)
                push_cmd({8'h01, 8'h00, 16'h1000, 16'h0002, 16'h0003});
                for (t = 0; t < 30; t = t + 1) begin
                    @(posedge clk);
                    if (m_axi_arvalid) saw_arvalid = 1'b1;
                end
                axi_read(16'h0004); // GPU_STATUS
                if (s_axi_rdata[2] && s_axi_rdata[1] &&
                    !uut.dma_busy &&
                    !saw_arvalid &&
                    uut.u_scratchpad.mem[16'h0001] == pre_dma_dst0 &&
                    uut.u_scratchpad.mem[16'h0002] == pre_dma_dst1) begin
                    $display("[PASS] Odd-byte DMA length rejected with status_error and no DMA side-effects");
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Odd-byte DMA rejection invalid: status=0x%08H dma_busy=%0d saw_arvalid=%0d dst1=%0d->%0d dst2=%0d->%0d",
                             s_axi_rdata, uut.dma_busy, saw_arvalid,
                             $signed(pre_dma_dst0), $signed(uut.u_scratchpad.mem[16'h0001]),
                             $signed(pre_dma_dst1), $signed(uut.u_scratchpad.mem[16'h0002]));
                    fail_count = fail_count + 1;
                end
            end
        end

        // Test 10: Optional imprint mode - mini-gpt-hc1-v1 profile (flags[2:1]=01, flags[0]=1)
        $display("[10] Dispatching MATMUL with MINI imprint profile...");
        axi_write(16'h0034, 32'd9);  // TOKEN_IN
        axi_write(16'h0038, 32'd4);  // POSITION_IN
        begin : wait_compute_done_mini
            integer elapsed_cycles;
            reg [15:0] expected_mini0;
            reg [15:0] expected_mini1;
            reg completed;
            measure_compute_latency({8'h02, 8'h03, 16'h0001, 16'h0040, 16'h0010}, elapsed_cycles); // CMD_MATMUL, imprint mini
            expected_mini0 = 16'd540; // 9*50 + 4*20 + 0*40 + 10
            expected_mini1 = 16'd580; // +40 per dim
            axi_read(16'h0004);
            completed = s_axi_rdata[1];
            mini_imprint_cycles = elapsed_cycles;
            if (completed && elapsed_cycles > 6 &&
                uut.opt_token_embedding[15:0] == expected_mini0 &&
                uut.opt_token_embedding[31:16] == expected_mini1) begin
                $display("[PASS] MINI imprint profile engaged in %0d cycles (emb0=%0d emb1=%0d)",
                         elapsed_cycles,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] MINI imprint mismatch: completed=%0d cycles=%0d emb0=%0d emb1=%0d",
                         completed, elapsed_cycles,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                fail_count = fail_count + 1;
            end
        end

        // Test 11: Optional imprint mode - gemma3 exported profile (flags[2:1]=10, flags[0]=1)
        $display("[11] Dispatching MATMUL with GEMMA exported imprint profile...");
        axi_write(16'h0034, 32'd12); // TOKEN_IN
        axi_write(16'h0038, 32'd5);  // POSITION_IN
        begin : wait_compute_done_gemma
            integer elapsed_cycles;
            reg [15:0] expected_gemma0;
            reg [15:0] expected_gemma1;
            reg profile_data_loaded;
            reg completed;
            measure_compute_latency({8'h02, 8'h05, 16'h0002, 16'h0060, 16'h0010}, elapsed_cycles); // CMD_MATMUL, imprint gemma exported profile
            expected_gemma0 = $signed(uut.u_imprint_embedding.gemma_token_rom[12*8 + 0]) +
                              $signed(uut.u_imprint_embedding.gemma_pos_rom[5*8 + 0]);
            expected_gemma1 = $signed(uut.u_imprint_embedding.gemma_token_rom[12*8 + 1]) +
                              $signed(uut.u_imprint_embedding.gemma_pos_rom[5*8 + 1]);
            profile_data_loaded = ($signed(expected_gemma0) != 16'sd0) || ($signed(expected_gemma1) != 16'sd0);
            axi_read(16'h0004);
            completed = s_axi_rdata[1];
            gemma_imprint_cycles = elapsed_cycles;
            if (completed && elapsed_cycles > 6 && profile_data_loaded &&
                uut.opt_token_embedding[15:0] == expected_gemma0 &&
                uut.opt_token_embedding[31:16] == expected_gemma1) begin
                $display("[PASS] GEMMA exported profile engaged in %0d cycles (emb0=%0d emb1=%0d)",
                         elapsed_cycles,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] GEMMA exported profile mismatch: completed=%0d cycles=%0d emb0=%0d emb1=%0d",
                         completed, elapsed_cycles,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                fail_count = fail_count + 1;
            end
        end

        // Test 12: Unsupported imprint profile must fail-closed with no writeback.
        $display("[12] Verifying unsupported imprint profile is rejected...");
        begin : unsupported_imprint_reject
            integer t;
            reg saw_done;
            reg saw_writeback;
            reg [15:0] pre_dst0;
            reg [15:0] pre_dst1;
            uut.u_scratchpad.mem[16'h0070] = 16'h55AA;
            uut.u_scratchpad.mem[16'h0071] = 16'hAA55;
            pre_dst0 = uut.u_scratchpad.mem[16'h0070];
            pre_dst1 = uut.u_scratchpad.mem[16'h0071];
            saw_done = 1'b0;
            saw_writeback = 1'b0;

            push_cmd({8'h02, 8'h07, 16'h0003, 16'h0070, 16'h0010}); // imprint_enable=1, profile=2'b11 (unsupported)
            for (t = 0; t < 120; t = t + 1) begin
                @(posedge clk);
                if (uut.cp_compute_done) saw_done = 1'b1;
                if (uut.compute_sp_write_en) saw_writeback = 1'b1;
                if (saw_done) t = 120;
            end
            axi_read(16'h0004); // GPU_STATUS
            if (saw_done &&
                s_axi_rdata[2] &&
                !saw_writeback &&
                uut.u_scratchpad.mem[16'h0070] == pre_dst0 &&
                uut.u_scratchpad.mem[16'h0071] == pre_dst1) begin
                $display("[PASS] Unsupported imprint profile rejected with status_error and no writeback");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Unsupported imprint rejection invalid: done=%0d status=0x%08H saw_writeback=%0d dst0=%0d->%0d dst1=%0d->%0d",
                         saw_done, s_axi_rdata, saw_writeback,
                         $signed(pre_dst0), $signed(uut.u_scratchpad.mem[16'h0070]),
                         $signed(pre_dst1), $signed(uut.u_scratchpad.mem[16'h0071]));
                fail_count = fail_count + 1;
            end
        end

        // Test 13: Multi-run speed comparison with interleaved sampling and stability margin.
        $display("[13] Comparing baseline vs MINI imprint speed (interleaved 5-run average)...");
        begin : compare_speed_multi
            integer r;
            integer base_sum;
            integer mini_sum;
            integer base_avg;
            integer mini_avg;
            integer speedup_x100;
            integer base_min;
            integer base_max;
            integer mini_min;
            integer mini_max;
            integer base_spread;
            integer mini_spread;
            integer base_ctmp;
            integer mini_ctmp;
            base_sum = 0;
            mini_sum = 0;
            base_min = 32'h7fffffff;
            base_max = 0;
            mini_min = 32'h7fffffff;
            mini_max = 0;

            for (r = 0; r < 5; r = r + 1) begin
                axi_write(16'h0034, 32'd21);
                axi_write(16'h0038, 32'd3);
                measure_compute_latency({8'h02, 8'hAA, 16'h0010, 16'h0020, 16'h0010}, base_ctmp);
                base_sum = base_sum + base_ctmp;
                if (base_ctmp < base_min) base_min = base_ctmp;
                if (base_ctmp > base_max) base_max = base_ctmp;

                axi_write(16'h0034, 32'd9);
                axi_write(16'h0038, 32'd4);
                measure_compute_latency({8'h02, 8'h03, 16'h0001, 16'h0040, 16'h0010}, mini_ctmp);
                mini_sum = mini_sum + mini_ctmp;
                if (mini_ctmp < mini_min) mini_min = mini_ctmp;
                if (mini_ctmp > mini_max) mini_max = mini_ctmp;

                $display("    run%0d baseline=%0d mini=%0d", r + 1, base_ctmp, mini_ctmp);
            end

            base_avg = base_sum / 5;
            mini_avg = mini_sum / 5;
            base_spread = base_max - base_min;
            mini_spread = mini_max - mini_min;
            speedup_x100 = (base_avg * 100) / mini_avg;

            if ((speedup_x100 >= 170) &&
                ((base_avg - mini_avg) >= 12) &&
                (base_avg >= 30) && (base_avg <= 45) &&
                (mini_avg >= 15) && (mini_avg <= 25) &&
                (base_spread <= 1) &&
                (mini_spread <= 1)) begin
                $display("[PASS] MINI imprint faster/stable: baseline=%0d cycles (spread=%0d), mini=%0d cycles (spread=%0d), speedup=%0d.%02dx",
                         base_avg, base_spread, mini_avg, mini_spread, speedup_x100 / 100, speedup_x100 % 100);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] MINI imprint speed/stability check failed: baseline=%0d (spread=%0d), mini=%0d (spread=%0d), speedup=%0d.%02dx",
                         base_avg, base_spread, mini_avg, mini_spread, speedup_x100 / 100, speedup_x100 % 100);
                fail_count = fail_count + 1;
            end
        end

        // Test 14: Reject odd-byte compute sizes explicitly.
        $display("[14] Verifying odd-byte compute size is rejected...");
        begin : odd_size_reject
            integer t;
            reg saw_writeback;
            reg [15:0] pre_dst0;
            reg [15:0] pre_dst1;
            pre_dst0 = uut.u_scratchpad.mem[16'h0020];
            pre_dst1 = uut.u_scratchpad.mem[16'h0021];
            axi_read(16'h0004); // GPU_STATUS
            if (s_axi_rdata[2]) begin
                $display("[FAIL] status_error already set before odd-size test: status=0x%08H", s_axi_rdata);
                fail_count = fail_count + 1;
            end else begin
                push_cmd({8'h02, 8'h00, 16'h0010, 16'h0020, 16'h0003}); // CMD_MATMUL, odd size bytes
                saw_writeback = 1'b0;
                for (t = 0; t < 20; t = t + 1) begin
                    @(posedge clk);
                    if (uut.compute_sp_write_en)
                        saw_writeback = 1'b1;
                end
                axi_read(16'h0004); // GPU_STATUS
                if (s_axi_rdata[2] &&
                    !saw_writeback &&
                    uut.u_scratchpad.mem[16'h0020] == pre_dst0 &&
                    uut.u_scratchpad.mem[16'h0021] == pre_dst1) begin
                    $display("[PASS] Odd-byte compute size rejected with status_error and no writeback side-effects");
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Odd-byte rejection invalid: status=0x%08H saw_writeback=%0d dst0=%0d->%0d dst1=%0d->%0d",
                             s_axi_rdata,
                             saw_writeback,
                             $signed(pre_dst0), $signed(uut.u_scratchpad.mem[16'h0020]),
                             $signed(pre_dst1), $signed(uut.u_scratchpad.mem[16'h0021]));
                    fail_count = fail_count + 1;
                end
            end
        end

        // Test 15: Verify valid compute clears prior status_error and recovers operation.
        $display("[15] Verifying recovery after odd-size rejection...");
        begin : recovery_after_error
            integer elapsed_cycles;
            reg [15:0] expected_dynamic0;
            reg [15:0] expected_dynamic1;
            reg completed;
            axi_write(16'h0034, 32'd21);  // TOKEN_IN seed
            axi_write(16'h0038, 32'd3);   // POSITION_IN
            measure_compute_latency({8'h02, 8'hAA, 16'h0010, 16'h0020, 16'h0010}, elapsed_cycles); // CMD_MATMUL
            expected_dynamic0 = 16'd21 + 16'h0010 + 16'd0;
            expected_dynamic1 = 16'd21 + 16'h0010 + 16'd1;
            axi_read(16'h0004); // GPU_STATUS
            completed = s_axi_rdata[1];
            if (completed &&
                !s_axi_rdata[2] &&
                elapsed_cycles > 6 &&
                uut.opt_token_embedding[15:0] == expected_dynamic0 &&
                uut.opt_token_embedding[31:16] == expected_dynamic1) begin
                $display("[PASS] Recovery compute succeeded and status_error cleared");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Recovery after error failed: completed=%0d elapsed=%0d status=0x%08H emb0=%0d emb1=%0d",
                         completed, elapsed_cycles, s_axi_rdata,
                         $signed(uut.opt_token_embedding[15:0]),
                         $signed(uut.opt_token_embedding[31:16]));
                fail_count = fail_count + 1;
            end
        end

        // Test 16: Prefetch DMA request must survive DMA-busy backpressure.
        $display("[16] Verifying prefetch request retention under DMA backpressure...");
        begin : prefetch_backpressure_retention
            integer t;
            reg saw_compute_done;
            reg saw_prefetch_req;
            reg saw_pending_while_busy;
            reg saw_issue_after_release;
            saw_compute_done = 1'b0;
            saw_prefetch_req = 1'b0;
            saw_pending_while_busy = 1'b0;
            saw_issue_after_release = 1'b0;

            axi_write(16'h0034, 32'd9);  // TOKEN_IN seed
            axi_write(16'h0038, 32'd1);  // POSITION_IN

            force uut.dma_busy = 1'b1;
            push_cmd({8'h02, 8'h90, 16'h0010, 16'h0020, 16'h0010}); // CMD_MATMUL + prefetch(bit4) + synthetic embedding(bit7)
            for (t = 0; t < 120; t = t + 1) begin
                @(posedge clk);
                if (uut.cp_compute_done)
                    saw_compute_done = 1'b1;
                if (uut.prefetch_dma_request)
                    saw_prefetch_req = 1'b1;
                if (uut.prefetch_dma_pending)
                    saw_pending_while_busy = 1'b1;
            end

            release uut.dma_busy;
            for (t = 0; t < 120; t = t + 1) begin
                @(posedge clk);
                if (uut.dma_issue_prefetch)
                    saw_issue_after_release = 1'b1;
            end

            if (saw_compute_done &&
                saw_prefetch_req &&
                saw_pending_while_busy &&
                saw_issue_after_release &&
                !uut.prefetch_error) begin
                $display("[PASS] Prefetch DMA request retained and issued after DMA backpressure release");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Prefetch backpressure handling failed: compute_done=%0d req=%0d pending=%0d issued=%0d prefetch_error=%0d",
                         saw_compute_done, saw_prefetch_req, saw_pending_while_busy, saw_issue_after_release, uut.prefetch_error);
                fail_count = fail_count + 1;
            end
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $display("TB_RESULT pass=%0d fail=%0d", pass_count, fail_count);
        if (fail_count != 0)
            $fatal(1, "gpu_system_top_v2_tb failed with %0d checks failing", fail_count);
        $finish;
    end

endmodule
