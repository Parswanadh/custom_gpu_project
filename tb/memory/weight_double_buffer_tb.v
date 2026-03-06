// ============================================================================
// Testbench: weight_double_buffer_tb
// Tests double-buffered weight storage with bank swapping
// ============================================================================
`timescale 1ns / 1ps

module weight_double_buffer_tb;

    parameter NW = 16;
    parameter DW = 16;
    parameter AW = $clog2(NW);

    reg                     clk, rst;
    reg                     load_en, read_en, swap_banks;
    reg  [AW-1:0]          load_addr, read_addr;
    reg  signed [DW-1:0]   load_data;
    wire signed [DW-1:0]   read_data;
    wire                   read_valid;
    wire                   active_bank;
    wire [31:0]            loads_completed, swaps_completed;

    weight_double_buffer #(.NUM_WEIGHTS(NW), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst),
        .load_en(load_en), .load_addr(load_addr), .load_data(load_data),
        .read_en(read_en), .read_addr(read_addr),
        .read_data(read_data), .read_valid(read_valid),
        .swap_banks(swap_banks), .active_bank(active_bank),
        .loads_completed(loads_completed), .swaps_completed(swaps_completed)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer i;

    initial begin
        $dumpfile("sim/waveforms/weight_double_buffer.vcd");
        $dumpvars(0, weight_double_buffer_tb);
    end

    initial begin
        $display("============================================");
        $display("  Weight Double Buffer Testbench");
        $display("============================================");

        rst = 1; load_en = 0; read_en = 0; swap_banks = 0;
        load_addr = 0; read_addr = 0; load_data = 0;
        #25; rst = 0; #15;

        // ---- Test 1: Load into Bank A (initially active), read should get it ----
        $display("[1] Loading 4 weights into Bank A...");
        // Bank A is active, so writes go to Bank B (inactive)
        // First load Bank A directly by ensuring active_bank=0, but writes go to inactive B
        // We need to load Bank A: bank_a is active when active_bank=0
        // To load into Bank A, we need active_bank=1 (A becomes inactive)
        // Let's swap first so Bank B is active (Bank A becomes writable)
        @(posedge clk); swap_banks <= 1;
        @(posedge clk); swap_banks <= 0;

        // Now active_bank=1 (Bank B active), writes go to Bank A
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            load_en <= 1; load_addr <= i; load_data <= (i + 1) * 100;  // 100, 200, 300, 400
        end
        @(posedge clk); load_en <= 0;

        // Swap back: Bank A becomes active
        @(posedge clk); swap_banks <= 1;
        @(posedge clk); swap_banks <= 0;

        // Read from Bank A
        @(posedge clk); read_en <= 1; read_addr <= 0;
        @(posedge clk); read_en <= 0;
        @(posedge clk); #1;
        if (read_data == 16'sd100) begin
            $display("[PASS] Bank A read: got %0d (expected 100)", read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Bank A read: got %0d (expected 100)", read_data);
            fail_count = fail_count + 1;
        end

        // ---- Test 2: Load Bank B while Bank A is active ----
        $display("[2] Loading Bank B while Bank A computes...");
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            load_en <= 1; load_addr <= i; load_data <= (i + 1) * 50;  // 50, 100, 150, 200
        end
        @(posedge clk); load_en <= 0;

        // Verify Bank A still intact
        @(posedge clk); read_en <= 1; read_addr <= 2;
        @(posedge clk); read_en <= 0;
        @(posedge clk); #1;
        if (read_data == 16'sd300) begin
            $display("[PASS] Bank A still intact: got %0d (expected 300)", read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Bank A intact check: got %0d (expected 300)", read_data);
            fail_count = fail_count + 1;
        end

        // ---- Test 3: Swap and read Bank B ----
        $display("[3] Swapping banks and reading Bank B...");
        @(posedge clk); swap_banks <= 1;
        @(posedge clk); swap_banks <= 0;

        @(posedge clk); read_en <= 1; read_addr <= 1;
        @(posedge clk); read_en <= 0;
        @(posedge clk); #1;
        if (read_data == 16'sd100) begin
            $display("[PASS] Bank B read after swap: got %0d (expected 100)", read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Bank B read after swap: got %0d (expected 100)", read_data);
            fail_count = fail_count + 1;
        end

        // ---- Test 4: Counter verification ----
        if (loads_completed == 32'd8) begin
            $display("[PASS] Load counter: %0d (expected 8)", loads_completed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Load counter: %0d (expected 8)", loads_completed);
            fail_count = fail_count + 1;
        end

        if (swaps_completed == 32'd3) begin
            $display("[PASS] Swap counter: %0d (expected 3)", swaps_completed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Swap counter: %0d (expected 3)", swaps_completed);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
