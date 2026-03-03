// ============================================================================
// Testbench: gpu_config_regs_tb
// Tests AXI4-Lite register read/write for GPU configuration
// ============================================================================
`timescale 1ns / 1ps

module gpu_config_regs_tb;

    parameter AXI_ADDR_W = 16;
    parameter AXI_DATA_W = 32;

    reg                     aclk, aresetn;

    // AXI4-Lite signals
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

    // Config outputs
    wire [15:0]             cfg_embed_dim;
    wire [7:0]              cfg_num_heads, cfg_num_layers;
    wire [15:0]             cfg_ffn_dim, cfg_max_seq_len, cfg_vocab_size;
    wire [1:0]              cfg_precision_mode;
    wire                    cfg_activation_type, cfg_infer_start;
    wire [7:0]              cfg_dq_scale;
    wire [3:0]              cfg_dq_offset;
    wire [15:0]             cfg_token_in, cfg_position_in;
    wire [7:0]              cfg_irq_enable;

    // Status inputs
    reg                     status_busy, status_idle, status_error;
    reg  [15:0]             status_token_out;
    reg  [7:0]              irq_pending;
    wire                    irq_out;

    gpu_config_regs #(.AXI_ADDR_W(AXI_ADDR_W), .AXI_DATA_W(AXI_DATA_W)) uut (
        .aclk(aclk), .aresetn(aresetn),
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
        .cfg_embed_dim(cfg_embed_dim), .cfg_num_heads(cfg_num_heads),
        .cfg_num_layers(cfg_num_layers), .cfg_ffn_dim(cfg_ffn_dim),
        .cfg_max_seq_len(cfg_max_seq_len), .cfg_vocab_size(cfg_vocab_size),
        .cfg_precision_mode(cfg_precision_mode), .cfg_activation_type(cfg_activation_type),
        .cfg_dq_scale(cfg_dq_scale), .cfg_dq_offset(cfg_dq_offset),
        .cfg_infer_start(cfg_infer_start), .cfg_token_in(cfg_token_in),
        .cfg_position_in(cfg_position_in), .cfg_irq_enable(cfg_irq_enable),
        .status_busy(status_busy), .status_idle(status_idle), .status_error(status_error),
        .status_token_out(status_token_out), .irq_pending(irq_pending),
        .irq_out(irq_out)
    );

    initial aclk = 0;
    always #5 aclk = ~aclk;

    integer pass_count = 0;
    integer fail_count = 0;

    // AXI write transaction
    task axi_write;
        input [AXI_ADDR_W-1:0] addr;
        input [AXI_DATA_W-1:0] data;
        integer timeout;
        begin
            @(posedge aclk);
            s_axi_awvalid <= 1'b1;
            s_axi_awaddr  <= addr;
            s_axi_wvalid  <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_bready  <= 1'b1;
            // Wait for both ready
            timeout = 0;
            while (timeout < 20) begin
                @(posedge aclk);
                if (s_axi_awready) s_axi_awvalid <= 1'b0;
                if (s_axi_wready) s_axi_wvalid <= 1'b0;
                if (s_axi_bvalid) begin
                    s_axi_bready <= 1'b0;
                    timeout = 20; // exit
                end
                timeout = timeout + 1;
            end
            @(posedge aclk);
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid <= 1'b0;
            s_axi_bready <= 1'b0;
        end
    endtask

    // AXI read transaction
    task axi_read;
        input [AXI_ADDR_W-1:0] addr;
        integer timeout;
        begin
            @(posedge aclk);
            s_axi_arvalid <= 1'b1;
            s_axi_araddr  <= addr;
            s_axi_rready  <= 1'b1;
            timeout = 0;
            while (!s_axi_rvalid && timeout < 20) begin
                @(posedge aclk);
                if (s_axi_arready) s_axi_arvalid <= 1'b0;
                timeout = timeout + 1;
            end
            @(posedge aclk);
            s_axi_arvalid <= 1'b0;
            s_axi_rready  <= 1'b0;
        end
    endtask

    initial begin
        $display("============================================");
        $display("  GPU Config Registers Testbench (AXI4-Lite)");
        $display("============================================");

        aresetn = 0;
        s_axi_awvalid = 0; s_axi_wvalid = 0; s_axi_bready = 0;
        s_axi_arvalid = 0; s_axi_rready = 0;
        s_axi_awaddr = 0; s_axi_wdata = 0; s_axi_araddr = 0;
        status_busy = 0; status_idle = 1; status_error = 0;
        status_token_out = 16'h0042; irq_pending = 0;
        #25; aresetn = 1; #15;

        // Test 1: Read GPU_ID (0x0000)
        $display("[1] Reading GPU_ID...");
        axi_read(16'h0000);
        if (s_axi_rdata == 32'hB17B_0001) begin
            $display("[PASS] GPU_ID = 0x%08H", s_axi_rdata);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] GPU_ID = 0x%08H (expected 0xB17B0001)", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 2: Read default EMBED_DIM
        $display("[2] Reading default EMBED_DIM...");
        axi_read(16'h0008);
        if (s_axi_rdata[15:0] == 16'd64) begin
            $display("[PASS] Default EMBED_DIM = %0d", s_axi_rdata[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Default EMBED_DIM = %0d (expected 64)", s_axi_rdata[15:0]);
            fail_count = fail_count + 1;
        end

        // Test 3: Write and read back EMBED_DIM
        $display("[3] Write EMBED_DIM = 768...");
        axi_write(16'h0008, 32'd768);
        axi_read(16'h0008);
        if (s_axi_rdata[15:0] == 16'd768 && cfg_embed_dim == 16'd768) begin
            $display("[PASS] EMBED_DIM written and read back: %0d", cfg_embed_dim);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] EMBED_DIM: rdata=%0d, cfg=%0d", s_axi_rdata[15:0], cfg_embed_dim);
            fail_count = fail_count + 1;
        end

        // Test 4: Write precision mode
        $display("[4] Write PRECISION_MODE = BF16 (1)...");
        axi_write(16'h0020, 32'd1);
        if (cfg_precision_mode == 2'd1) begin
            $display("[PASS] PRECISION_MODE = %0d", cfg_precision_mode);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] PRECISION_MODE = %0d (expected 1)", cfg_precision_mode);
            fail_count = fail_count + 1;
        end

        // Test 5: Read GPU_STATUS
        $display("[5] Reading GPU_STATUS...");
        status_busy = 1; status_idle = 0; status_error = 0;
        @(posedge aclk);
        axi_read(16'h0004);
        if (s_axi_rdata[0] == 1'b1 && s_axi_rdata[1] == 1'b0) begin
            $display("[PASS] STATUS: busy=%b, idle=%b", s_axi_rdata[0], s_axi_rdata[1]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] STATUS = 0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        // Test 6: INFER_START auto-clear
        $display("[6] Testing INFER_START auto-clear...");
        axi_write(16'h0030, 32'd1);
        // cfg_infer_start should pulse for 1 cycle then auto-clear
        repeat(3) @(posedge aclk);
        if (cfg_infer_start == 1'b0) begin
            $display("[PASS] INFER_START auto-cleared");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] INFER_START still asserted: %b", cfg_infer_start);
            fail_count = fail_count + 1;
        end

        // Test 7: Read token out
        $display("[7] Reading INFER_TOKEN_OUT...");
        status_token_out = 16'h002A;
        @(posedge aclk);
        axi_read(16'h003C);
        if (s_axi_rdata[15:0] == 16'h002A) begin
            $display("[PASS] TOKEN_OUT = 0x%04H", s_axi_rdata[15:0]);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] TOKEN_OUT = 0x%04H (expected 0x002A)", s_axi_rdata[15:0]);
            fail_count = fail_count + 1;
        end

        // Test 8: Invalid address returns DEAD_BEEF
        $display("[8] Reading invalid address...");
        axi_read(16'h00FF);
        if (s_axi_rdata == 32'hDEAD_BEEF) begin
            $display("[PASS] Invalid addr returns 0xDEADBEEF");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Invalid addr returns 0x%08H", s_axi_rdata);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
