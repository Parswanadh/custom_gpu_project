// ============================================================================
// Testbench: scratchpad_tb
// Tests dual-port SRAM read/write on both ports
// ============================================================================
`timescale 1ns / 1ps

module scratchpad_tb;

    parameter DEPTH  = 256;  // Small for testing
    parameter DATA_W = 16;
    parameter ADDR_W = $clog2(DEPTH);

    reg                 clk, rst;

    // Port A
    reg                 a_read_en, a_write_en;
    reg  [ADDR_W-1:0]  a_read_addr, a_write_addr;
    reg  [DATA_W-1:0]  a_write_data;
    wire [DATA_W-1:0]  a_read_data;
    wire                a_read_valid;

    // Port B
    reg                 b_read_en, b_write_en;
    reg  [ADDR_W-1:0]  b_read_addr, b_write_addr;
    reg  [DATA_W-1:0]  b_write_data;
    wire [DATA_W-1:0]  b_read_data;
    wire                b_read_valid;

    wire [ADDR_W:0]     usage_count;

    scratchpad #(.DEPTH(DEPTH), .DATA_W(DATA_W)) uut (
        .clk(clk), .rst(rst),
        .a_read_en(a_read_en), .a_read_addr(a_read_addr), .a_read_data(a_read_data),
        .a_read_valid(a_read_valid),
        .a_write_en(a_write_en), .a_write_addr(a_write_addr), .a_write_data(a_write_data),
        .b_read_en(b_read_en), .b_read_addr(b_read_addr), .b_read_data(b_read_data),
        .b_read_valid(b_read_valid),
        .b_write_en(b_write_en), .b_write_addr(b_write_addr), .b_write_data(b_write_data),
        .usage_count(usage_count)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    initial begin
        $display("============================================");
        $display("  Scratchpad (Dual-Port SRAM) Testbench");
        $display("============================================");

        rst = 1;
        a_read_en = 0; a_write_en = 0; a_read_addr = 0; a_write_addr = 0; a_write_data = 0;
        b_read_en = 0; b_write_en = 0; b_read_addr = 0; b_write_addr = 0; b_write_data = 0;
        #25; rst = 0; #15;

        // Test 1: Write via Port A, read via Port A
        $display("[1] Port A write + Port A read...");
        @(posedge clk);
        a_write_en   <= 1'b1;
        a_write_addr <= 8'd10;
        a_write_data <= 16'hCAFE;
        @(posedge clk);
        a_write_en <= 1'b0;

        @(posedge clk);
        a_read_en   <= 1'b1;
        a_read_addr <= 8'd10;
        @(posedge clk);
        a_read_en <= 1'b0;
        @(posedge clk); // Wait for read_valid

        if (a_read_valid && a_read_data == 16'hCAFE) begin
            $display("[PASS] Port A: read 0x%04H from addr 10", a_read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Port A: read 0x%04H, valid=%b (expected 0xCAFE)", a_read_data, a_read_valid);
            fail_count = fail_count + 1;
        end

        // Test 2: Write via Port A, read via Port B (cross-port)
        $display("[2] Port A write, Port B read...");
        @(posedge clk);
        a_write_en   <= 1'b1;
        a_write_addr <= 8'd20;
        a_write_data <= 16'hBEEF;
        @(posedge clk);
        a_write_en <= 1'b0;

        @(posedge clk);
        b_read_en   <= 1'b1;
        b_read_addr <= 8'd20;
        @(posedge clk);
        b_read_en <= 1'b0;
        @(posedge clk);

        if (b_read_valid && b_read_data == 16'hBEEF) begin
            $display("[PASS] Port B: read 0x%04H from addr 20 (cross-port)", b_read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Port B: read 0x%04H, valid=%b (expected 0xBEEF)", b_read_data, b_read_valid);
            fail_count = fail_count + 1;
        end

        // Test 3: Write via Port B, read via Port A
        $display("[3] Port B write, Port A read...");
        @(posedge clk);
        b_write_en   <= 1'b1;
        b_write_addr <= 8'd30;
        b_write_data <= 16'hDEAD;
        @(posedge clk);
        b_write_en <= 1'b0;

        @(posedge clk);
        a_read_en   <= 1'b1;
        a_read_addr <= 8'd30;
        @(posedge clk);
        a_read_en <= 1'b0;
        @(posedge clk);

        if (a_read_valid && a_read_data == 16'hDEAD) begin
            $display("[PASS] Port A: read 0x%04H from addr 30 (cross-port)", a_read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Port A: read 0x%04H, valid=%b (expected 0xDEAD)", a_read_data, a_read_valid);
            fail_count = fail_count + 1;
        end

        // Test 4: Simultaneous writes to different addresses
        $display("[4] Simultaneous write from Port A and Port B...");
        @(posedge clk);
        a_write_en   <= 1'b1;
        a_write_addr <= 8'd40;
        a_write_data <= 16'h1111;
        b_write_en   <= 1'b1;
        b_write_addr <= 8'd50;
        b_write_data <= 16'h2222;
        @(posedge clk);
        a_write_en <= 1'b0;
        b_write_en <= 1'b0;

        // Read back both
        @(posedge clk);
        a_read_en   <= 1'b1;
        a_read_addr <= 8'd40;
        @(posedge clk);
        a_read_en <= 1'b0;
        @(posedge clk);

        if (a_read_valid && a_read_data == 16'h1111) begin
            // Check second write
            b_read_en   <= 1'b1;
            b_read_addr <= 8'd50;
            @(posedge clk);
            b_read_en <= 1'b0;
            @(posedge clk);

            if (b_read_valid && b_read_data == 16'h2222) begin
                $display("[PASS] Simultaneous writes: addr40=0x%04H, addr50=0x%04H", 16'h1111, b_read_data);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Port B data: 0x%04H (expected 0x2222)", b_read_data);
                fail_count = fail_count + 1;
            end
        end else begin
            $display("[FAIL] Port A data: 0x%04H (expected 0x1111)", a_read_data);
            fail_count = fail_count + 1;
        end

        // Test 5: Same-address collision policy (Port B wins)
        $display("[5] Same-address simultaneous write (Port B priority)...");
        @(posedge clk);
        a_write_en   <= 1'b1;
        a_write_addr <= 8'd60;
        a_write_data <= 16'hAAAA;
        b_write_en   <= 1'b1;
        b_write_addr <= 8'd60;
        b_write_data <= 16'hBBBB;
        @(posedge clk);
        a_write_en <= 1'b0;
        b_write_en <= 1'b0;
        @(posedge clk);
        a_read_en   <= 1'b1;
        a_read_addr <= 8'd60;
        @(posedge clk);
        a_read_en <= 1'b0;
        @(posedge clk);

        if (a_read_valid && a_read_data == 16'hBBBB) begin
            $display("[PASS] Collision policy enforced: addr60=0x%04H (Port B wins)", a_read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Collision policy mismatch: got=0x%04H valid=%b (expected 0xBBBB)", a_read_data, a_read_valid);
            fail_count = fail_count + 1;
        end

        // Test 6: Burst write and sequential read
        $display("[6] Burst write/read pattern...");
        begin : burst_test
            integer i;
            reg [DATA_W-1:0] expected;
            reg burst_ok;
            // Write 8 values
            for (i = 0; i < 8; i = i + 1) begin
                @(posedge clk);
                a_write_en   <= 1'b1;
                a_write_addr <= i[ADDR_W-1:0];
                a_write_data <= (i + 1) * 100;
            end
            @(posedge clk); a_write_en <= 1'b0;

            // Read back
            burst_ok = 1;
            for (i = 0; i < 8; i = i + 1) begin
                @(posedge clk);
                b_read_en   <= 1'b1;
                b_read_addr <= i[ADDR_W-1:0];
                @(posedge clk);
                b_read_en <= 1'b0;
                @(posedge clk);
                expected = (i + 1) * 100;
                if (!b_read_valid || b_read_data != expected) begin
                    $display("    Mismatch at addr %0d: got 0x%04H, expected %0d", i, b_read_data, expected);
                    burst_ok = 0;
                end
            end

            if (burst_ok) begin
                $display("[PASS] Burst write/read: 8 values verified");
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] Burst write/read: mismatches detected");
                fail_count = fail_count + 1;
            end
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        if (fail_count != 0)
            $fatal(1, "scratchpad_tb failed with %0d checks failing", fail_count);
        $finish;
    end

endmodule
