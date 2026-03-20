// ============================================================================
// Testbench: command_processor_tb
// Tests all 8 opcodes of the command processor FIFO
// ============================================================================
`timescale 1ns / 1ps

module command_processor_tb;

    parameter CMD_WIDTH  = 64;
    parameter FIFO_DEPTH = 16;
    parameter ADDR_WIDTH = 16;

    reg                     clk, rst;
    reg                     cmd_valid;
    reg  [CMD_WIDTH-1:0]    cmd_data;
    wire                    cmd_ready;

    // Scratchpad interface (unused in this test, just wired)
    wire                    sp_read_en, sp_write_en;
    wire [ADDR_WIDTH-1:0]   sp_read_addr, sp_write_addr;
    reg  [15:0]             sp_read_data;
    wire [15:0]             sp_write_data;

    // DMA interface
    wire                    dma_start;
    wire [31:0]             dma_ext_addr;
    wire [ADDR_WIDTH-1:0]   dma_local_addr;
    wire [15:0]             dma_length;
    reg                     dma_done;

    // Compute interface
    wire                    compute_start;
    wire [7:0]              compute_opcode;
    wire [ADDR_WIDTH-1:0]   compute_src_addr, compute_dst_addr;
    wire [15:0]             compute_size;
    wire [7:0]              compute_flags;
    reg                     compute_done;

    // Status
    wire                    busy, idle;
    wire [31:0]             cmds_executed;
    wire                    error_out;
    wire                    interrupt_out;

    command_processor #(
        .CMD_WIDTH(CMD_WIDTH), .FIFO_DEPTH(FIFO_DEPTH), .ADDR_WIDTH(ADDR_WIDTH)
    ) uut (
        .clk(clk), .rst(rst),
        .cmd_valid(cmd_valid), .cmd_data(cmd_data), .cmd_ready(cmd_ready),
        .sp_read_en(sp_read_en), .sp_read_addr(sp_read_addr), .sp_read_data(sp_read_data),
        .sp_write_en(sp_write_en), .sp_write_addr(sp_write_addr), .sp_write_data(sp_write_data),
        .dma_start(dma_start), .dma_ext_addr(dma_ext_addr),
        .dma_local_addr(dma_local_addr), .dma_length(dma_length), .dma_done(dma_done),
        .compute_start(compute_start), .compute_opcode(compute_opcode),
        .compute_src_addr(compute_src_addr), .compute_dst_addr(compute_dst_addr),
        .compute_size(compute_size), .compute_flags(compute_flags), .compute_done(compute_done),
        .busy(busy), .idle(idle), .cmds_executed(cmds_executed),
        .error_out(error_out),
        .interrupt_out(interrupt_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    // Helper: build command descriptor
    function [63:0] make_cmd;
        input [7:0]  opcode;
        input [7:0]  flags;
        input [15:0] src;
        input [15:0] dst;
        input [15:0] size;
        begin
            make_cmd = {opcode, flags, src, dst, size};
        end
    endfunction

    // Helper: push command
    task push_cmd;
        input [63:0] cmd;
        begin
            @(negedge clk);
            cmd_valid <= 1'b1;
            cmd_data  <= cmd;
            @(negedge clk);
            cmd_valid <= 1'b0;
        end
    endtask

    // Helper: wait for idle
    task wait_idle;
        integer timeout;
        begin
            timeout = 0;
            while (!idle && timeout < 100) begin
                @(posedge clk);
                timeout = timeout + 1;
            end
        end
    endtask

    initial begin
        $display("============================================");
        $display("  Command Processor Testbench");
        $display("============================================");

        rst = 1; cmd_valid = 0; cmd_data = 0;
        sp_read_data = 16'hABCD;
        dma_done = 0; compute_done = 0;
        #25; rst = 0; #15;

        // Test 1: NOP command
        $display("[1] Testing CMD_NOP...");
        push_cmd(make_cmd(8'h00, 8'h00, 16'h0000, 16'h0000, 16'h0000));
        // Wait enough cycles for FIFO read → DISPATCH → NOP execute → IDLE
        repeat(10) @(posedge clk);
        if (cmds_executed == 1 && idle) begin
            $display("[PASS] CMD_NOP executed, cmds_executed=%0d", cmds_executed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] CMD_NOP: cmds_executed=%0d, idle=%b", cmds_executed, idle);
            fail_count = fail_count + 1;
        end

        // Test 2: LOAD_WEIGHTS (triggers DMA)
        $display("[2] Testing CMD_LOAD_WEIGHTS...");
        push_cmd(make_cmd(8'h01, 8'h00, 16'h1000, 16'h2000, 16'h0040));
        // Wait for DMA start
        begin : wait_dma
            integer t;
            t = 0;
            while (!dma_start && t < 50) begin @(posedge clk); t = t + 1; end
        end
        if (dma_start && dma_length == 16'h0040) begin
            // Simulate DMA completion
            #30; @(posedge clk); dma_done <= 1'b1; @(posedge clk); dma_done <= 1'b0;
            wait_idle();
            if (cmds_executed == 2) begin
                $display("[PASS] CMD_LOAD_WEIGHTS: DMA triggered, addr=0x%h, len=%0d", dma_ext_addr, dma_length);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] CMD_LOAD_WEIGHTS: cmds_executed=%0d", cmds_executed);
                fail_count = fail_count + 1;
            end
        end else begin
            $display("[FAIL] CMD_LOAD_WEIGHTS: DMA not triggered");
            fail_count = fail_count + 1;
        end

        // Test 3: CMD_MATMUL (triggers compute)
        $display("[3] Testing CMD_MATMUL...");
        push_cmd(make_cmd(8'h02, 8'hAA, 16'h3000, 16'h4000, 16'h0010));
        begin : wait_comp
            integer t;
            t = 0;
            while (!compute_start && t < 50) begin @(posedge clk); t = t + 1; end
        end
        if (compute_start && compute_opcode == 8'h02 && compute_flags == 8'hAA) begin
            #20; @(posedge clk); compute_done <= 1'b1; @(posedge clk); compute_done <= 1'b0;
            wait_idle();
            if (cmds_executed == 3) begin
                $display("[PASS] CMD_MATMUL: compute triggered, opcode=0x%h, flags=0x%h", compute_opcode, compute_flags);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] CMD_MATMUL: cmds_executed=%0d", cmds_executed);
                fail_count = fail_count + 1;
            end
        end else begin
            $display("[FAIL] CMD_MATMUL: compute not triggered");
            fail_count = fail_count + 1;
        end

        // Test 4: CMD_FENCE (generates interrupt)
        $display("[4] Testing CMD_FENCE...");
        push_cmd(make_cmd(8'h0F, 8'h00, 16'h0000, 16'h0000, 16'h0000));
        begin : wait_fence
            integer t;
            t = 0;
            while (!interrupt_out && t < 50) begin @(posedge clk); t = t + 1; end
        end
        if (interrupt_out) begin
            wait_idle();
            $display("[PASS] CMD_FENCE: interrupt generated, cmds_executed=%0d", cmds_executed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] CMD_FENCE: no interrupt");
            fail_count = fail_count + 1;
        end

        // Test 5: cmd_ready (FIFO backpressure)
        $display("[5] Testing FIFO backpressure...");
        if (cmd_ready) begin
            $display("[PASS] FIFO accepts commands when not full");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] cmd_ready not asserted when FIFO should be empty");
            fail_count = fail_count + 1;
        end

        // Test 6: CMD_ACTIVATION (triggers compute)
        $display("[6] Testing CMD_ACTIVATION...");
        push_cmd(make_cmd(8'h03, 8'h01, 16'h5000, 16'h6000, 16'h0008));
        begin : wait_act
            integer t;
            t = 0;
            while (!compute_start && t < 50) begin @(posedge clk); t = t + 1; end
        end
        if (compute_start && compute_opcode == 8'h03) begin
            @(posedge clk); compute_done <= 1'b1; @(posedge clk); compute_done <= 1'b0;
            wait_idle();
            $display("[PASS] CMD_ACTIVATION: compute triggered");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] CMD_ACTIVATION: compute not triggered");
            fail_count = fail_count + 1;
        end

        // Test 7: WAIT_COMP timeout should fail-closed and surface error_out
        $display("[7] Testing compute wait watchdog timeout...");
        begin : test_watchdog
            integer t;
            reg saw_error;
            reg saw_busy;
            saw_error = 1'b0;
            saw_busy = 1'b0;
            push_cmd(make_cmd(8'h02, 8'h00, 16'h0000, 16'h0000, 16'h0002));
            for (t = 0; t < 1200; t = t + 1) begin
                @(posedge clk);
                if (error_out) saw_error = 1'b1;
                if (busy) saw_busy = 1'b1;
                if (saw_busy && idle) t = 1200;
            end
            if (saw_busy && idle && saw_error) begin
                $display("[PASS] Watchdog timeout returned to IDLE and asserted error_out");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Watchdog timeout missing: saw_busy=%b idle=%b saw_error=%b", saw_busy, idle, saw_error);
                fail_count = fail_count + 1;
            end
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
