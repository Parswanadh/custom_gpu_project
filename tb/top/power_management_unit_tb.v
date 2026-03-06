// ============================================================================
// Testbench: power_management_unit_tb
// Tests PMU power modes, auto-sleep, and wake functionality
// ============================================================================
`timescale 1ns / 1ps

module power_management_unit_tb;

    parameter NC = 4;
    parameter TIMEOUT = 16;

    reg                     clk, rst;
    reg  [NC-1:0]          core_active;
    reg  [1:0]             power_mode_req;
    reg                    wake_interrupt;
    wire [NC-1:0]          core_clk_en;
    wire [1:0]             current_mode;
    wire [31:0]            idle_cycles, active_cycles, sleep_cycles, gated_core_cycles;

    power_management_unit #(.NUM_CORES(NC), .IDLE_TIMEOUT(TIMEOUT)) uut (
        .clk(clk), .rst(rst),
        .core_active(core_active),
        .power_mode_req(power_mode_req),
        .wake_interrupt(wake_interrupt),
        .core_clk_en(core_clk_en),
        .current_mode(current_mode),
        .idle_cycles(idle_cycles), .active_cycles(active_cycles),
        .sleep_cycles(sleep_cycles), .gated_core_cycles(gated_core_cycles)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $dumpfile("sim/waveforms/power_management_unit.vcd");
        $dumpvars(0, power_management_unit_tb);
    end

    initial begin
        $display("============================================");
        $display("  Power Management Unit Testbench");
        $display("============================================");

        rst = 1; core_active = 0; power_mode_req = 2'b00; wake_interrupt = 0;
        #25; rst = 0; #15;

        // Test 1: All cores active → FULL mode
        core_active = 4'b1111;
        repeat(5) @(posedge clk);
        if (current_mode == 2'd0 && core_clk_en == 4'b1111) begin
            $display("[PASS] Full mode: all cores enabled");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Full mode: mode=%0d, clk_en=%b", current_mode, core_clk_en);
            fail_count = fail_count + 1;
        end

        // Test 2: Force ECO mode
        power_mode_req = 2'b10;
        core_active = 4'b0101;  // Cores 0,2 active
        repeat(3) @(posedge clk);
        if (current_mode == 2'd1 && core_clk_en == 4'b0101) begin
            $display("[PASS] ECO mode: only active cores enabled (0101)");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] ECO mode: mode=%0d, clk_en=%b", current_mode, core_clk_en);
            fail_count = fail_count + 1;
        end

        // Test 3: Force SLEEP mode
        power_mode_req = 2'b11;
        repeat(3) @(posedge clk);
        if (current_mode == 2'd2 && core_clk_en == 4'b0000) begin
            $display("[PASS] Sleep mode: all cores disabled");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Sleep mode: mode=%0d, clk_en=%b", current_mode, core_clk_en);
            fail_count = fail_count + 1;
        end

        // Test 4: AUTO mode → idle → auto-sleep
        power_mode_req = 2'b00;
        core_active = 4'b0000;
        repeat(TIMEOUT + 5) @(posedge clk);
        if (current_mode == 2'd2) begin
            $display("[PASS] Auto-sleep after %0d idle cycles", TIMEOUT);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Auto-sleep: mode=%0d after timeout", current_mode);
            fail_count = fail_count + 1;
        end

        // Test 5: Wake from sleep via interrupt
        wake_interrupt = 1;
        core_active = 4'b1111;  // Also signal active to keep awake
        @(posedge clk);
        wake_interrupt = 0;
        repeat(3) @(posedge clk);
        if (current_mode == 2'd0) begin
            $display("[PASS] Wake from sleep: mode=FULL");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Wake from sleep: mode=%0d", current_mode);
            fail_count = fail_count + 1;
        end

        // Test 6: Verify gated_core_cycles > 0
        if (gated_core_cycles > 0) begin
            $display("[PASS] Power savings tracked: %0d gated core-cycles", gated_core_cycles);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] No gated core-cycles tracked");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
