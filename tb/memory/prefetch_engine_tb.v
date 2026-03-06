`timescale 1ns / 1ps
module prefetch_engine_tb;
    parameter BUFFER_DEPTH=64, DATA_WIDTH=32, ADDR_WIDTH=6;
    reg clk, rst, start, layer_done, dma_done_sig;
    reg [7:0] total_layers;
    wire dma_request;
    wire [31:0] dma_src_addr;
    wire [15:0] dma_length;
    reg buf_read_en, buf_write_en;
    reg [ADDR_WIDTH-1:0] buf_read_addr, buf_write_addr;
    wire [DATA_WIDTH-1:0] buf_read_data;
    reg [DATA_WIDTH-1:0] buf_write_data;
    wire compute_ready, prefetch_active, all_done;
    wire [7:0] current_layer, prefetch_layer;

    prefetch_engine #(.BUFFER_DEPTH(BUFFER_DEPTH), .DATA_WIDTH(DATA_WIDTH)
    ) dut (.clk(clk), .rst(rst), .start(start), .layer_done(layer_done),
        .total_layers(total_layers), .dma_request(dma_request),
        .dma_src_addr(dma_src_addr), .dma_length(dma_length), .dma_done(dma_done_sig),
        .buf_read_en(buf_read_en), .buf_read_addr(buf_read_addr),
        .buf_read_data(buf_read_data), .buf_write_en(buf_write_en),
        .buf_write_addr(buf_write_addr), .buf_write_data(buf_write_data),
        .compute_ready(compute_ready), .prefetch_active(prefetch_active),
        .current_layer(current_layer), .prefetch_layer(prefetch_layer),
        .all_done(all_done));

    always #5 clk = ~clk;
    integer tp=0, tt=0, i;

    initial begin
        clk=0; rst=1; start=0; layer_done=0; dma_done_sig=0;
        total_layers=3; buf_read_en=0; buf_write_en=0;
        buf_read_addr=0; buf_write_addr=0; buf_write_data=0;
        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);

        $display("=================================================");
        $display("   Hardware Prefetch Engine Tests");
        $display("   Paper: Google 'Four Architectural Opportunities' (Jan 2026)");
        $display("   Strategy: Overlap compute + memory transfer");
        $display("=================================================");

        // TEST 1: Start prefetch pipeline → DMA request generated
        tt = tt + 1;
        start = 1; @(negedge clk); start = 0;
        @(negedge clk);
        if (dma_request || prefetch_active) begin
            $display("[PASS] Test 1: Prefetch started — dma_request generated");
            tp = tp + 1;
        end else $display("[FAIL] Test 1");

        // Simulate DMA completing layer 0 write
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            buf_write_en = 1; buf_write_addr = i; buf_write_data = 32'hDEAD0000 + i;
        end
        buf_write_en = 0;
        @(negedge clk); dma_done_sig = 1; @(negedge clk); dma_done_sig = 0;

        // TEST 2: After first DMA done → compute_ready asserted
        tt = tt + 1;
        repeat(3) @(negedge clk);
        if (compute_ready) begin
            $display("[PASS] Test 2: First layer loaded — compute_ready = %b", compute_ready);
            tp = tp + 1;
        end else $display("[FAIL] Test 2: compute_ready=%b", compute_ready);

        // TEST 3: Read from compute buffer
        tt = tt + 1;
        buf_read_en = 1; buf_read_addr = 0;
        @(negedge clk);
        buf_read_en = 0;
        $display("    Read buffer[0] = 0x%08H", buf_read_data);
        if (buf_read_data != 0) begin
            $display("[PASS] Test 3: Buffer read returns data");
            tp = tp + 1;
        end else $display("[FAIL] Test 3");

        // Simulate DMA for layer 1 completing
        @(negedge clk); dma_done_sig = 1; @(negedge clk); dma_done_sig = 0;

        // TEST 4: Signal layer done → should swap buffers and advance
        tt = tt + 1;
        @(negedge clk); layer_done = 1; @(negedge clk); layer_done = 0;
        repeat(4) @(negedge clk); // Allow state machine to process
        if (current_layer >= 1) begin
            $display("[PASS] Test 4: Layer advanced — current_layer=%0d", current_layer);
            tp = tp + 1;
        end else $display("[FAIL] Test 4: current_layer=%0d", current_layer);

        $display("=================================================");
        $display("   Prefetch Tests: %0d / %0d PASSED", tp, tt);
        $display("=================================================");
        #10 $finish;
    end
    initial begin #50000; $display("TIMEOUT"); $finish; end
endmodule
