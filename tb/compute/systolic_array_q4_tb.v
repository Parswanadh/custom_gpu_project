// ============================================================================
// Testbench: systolic_array_q4_tb
// Tests: 4x4 systolic array in Q4 quantized weight mode
//
// Q4 dequantization in each PE: dequant_w = (w_int4 - zero) * scale
// Weight data stores INT4 value in the lower nibble of weight_data[3:0].
// ============================================================================
`timescale 1ns / 1ps

module systolic_array_q4_tb;

    parameter N  = 4;
    parameter DW = 16;
    parameter AW = 32;

    reg                     clk, rst;
    reg                     load_weight;
    reg  [1:0]              weight_row, weight_col;
    reg  [DW-1:0]           weight_data;
    reg                     valid_in;
    reg  [N*DW-1:0]         act_in;
    reg                     clear_acc;
    reg  [1:0]              precision_mode;
    reg  [7:0]              q4_block_scale;
    reg  [3:0]              q4_block_zero;
    wire [N*AW-1:0]         result_out;
    wire                    valid_out;

    systolic_array #(.ARRAY_SIZE(N), .DATA_WIDTH(DW), .ACC_WIDTH(AW)) uut (
        .clk(clk), .rst(rst),
        .load_weight(load_weight), .weight_row(weight_row), .weight_col(weight_col), .weight_data(weight_data),
        .valid_in(valid_in), .act_in(act_in),
        .clear_acc(clear_acc),
        .precision_mode(precision_mode),
        .q4_block_scale(q4_block_scale),
        .q4_block_zero(q4_block_zero),
        .result_out(result_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task load_w;
        input [1:0] r, c;
        input [DW-1:0] val;
        begin
            @(negedge clk);
            load_weight = 1'b1;
            weight_row = r; weight_col = c; weight_data = val;
            @(negedge clk);
            load_weight = 1'b0;
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/systolic_array_q4.vcd");
        $dumpvars(0, systolic_array_q4_tb);
    end

    initial begin
        $display("============================================");
        $display("  Systolic Array Q4 Mode Testbench");
        $display("============================================");

        // Initialize all signals
        rst = 1; load_weight = 0; valid_in = 0; act_in = 0;
        weight_row = 0; weight_col = 0; weight_data = 0;
        clear_acc = 0;
        precision_mode = 2'd1;
        q4_block_scale = 8'd1;
        q4_block_zero  = 4'd8;
        #25; rst = 0; #15;

        // ==================================================================
        // Test 1: Q4 diagonal matrix
        //   INT4 weights: diagonal = [9,10,11,12], off-diagonal = 8 (zero pt)
        //   scale=1, zero=8
        //   Dequant diagonal: [(9-8)*1, (10-8)*1, (11-8)*1, (12-8)*1]
        //                   = [1, 2, 3, 4]
        //   Off-diagonal dequant: (8-8)*1 = 0
        //   act = [10, 20, 30, 40]
        //   Expected: [10*1, 20*2, 30*3, 40*4] = [10, 40, 90, 160]
        // ==================================================================
        $display("\n--- Test 1: Q4 diagonal matrix (scale=1, zero=8) ---");
        precision_mode = 2'd1;
        q4_block_scale = 8'd1;
        q4_block_zero  = 4'd8;

        // Load all 16 weights: off-diagonal = zero_point (8) → dequant = 0
        load_w(0, 0, 16'd9);  load_w(0, 1, 16'd8);  load_w(0, 2, 16'd8);  load_w(0, 3, 16'd8);
        load_w(1, 0, 16'd8);  load_w(1, 1, 16'd10); load_w(1, 2, 16'd8);  load_w(1, 3, 16'd8);
        load_w(2, 0, 16'd8);  load_w(2, 1, 16'd8);  load_w(2, 2, 16'd11); load_w(2, 3, 16'd8);
        load_w(3, 0, 16'd8);  load_w(3, 1, 16'd8);  load_w(3, 2, 16'd8);  load_w(3, 3, 16'd12);

        #10;

        @(negedge clk);
        act_in = {16'd40, 16'd30, 16'd20, 16'd10};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        begin : wait1
            integer wcnt;
            wcnt = 0;
            while (!valid_out && wcnt < 100) begin
                @(posedge clk); #1;
                wcnt = wcnt + 1;
            end
        end

        if (valid_out &&
            result_out[31:0]   == 32'd10 &&
            result_out[63:32]  == 32'd40 &&
            result_out[95:64]  == 32'd90 &&
            result_out[127:96] == 32'd160) begin
            $display("[PASS] Q4 diagonal: [10, 40, 90, 160]");
            pass_count = pass_count + 1;
        end else if (valid_out) begin
            $display("[FAIL] Q4 diagonal: expected [10,40,90,160], got [%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end else begin
            $display("[FAIL] Q4 diagonal: no valid output");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // Test 2: Q4 full matrix with scale=2
        //   INT4 weights:          Dequant (scale=2, zero=8):
        //   Row 0:  9, 10,  8, 12    →  2,  4,  0,  8
        //   Row 1:  8,  9, 10,  8    →  0,  2,  4,  0
        //   Row 2: 11,  8,  9,  8    →  6,  0,  2,  0
        //   Row 3: 10, 12,  8,  9    →  4,  8,  0,  2
        //   act = [1, 1, 1, 1]
        //   result[col] = column sum of dequant matrix
        //   Expected: [12, 14, 6, 10]
        // ==================================================================
        @(negedge clk); rst = 1; clear_acc = 1; @(negedge clk); rst = 0; clear_acc = 0; #10;

        $display("\n--- Test 2: Q4 full matrix (scale=2, zero=8) ---");
        precision_mode = 2'd1;
        q4_block_scale = 8'd2;
        q4_block_zero  = 4'd8;

        load_w(0, 0, 16'd9);  load_w(0, 1, 16'd10); load_w(0, 2, 16'd8);  load_w(0, 3, 16'd12);
        load_w(1, 0, 16'd8);  load_w(1, 1, 16'd9);  load_w(1, 2, 16'd10); load_w(1, 3, 16'd8);
        load_w(2, 0, 16'd11); load_w(2, 1, 16'd8);  load_w(2, 2, 16'd9);  load_w(2, 3, 16'd8);
        load_w(3, 0, 16'd10); load_w(3, 1, 16'd12); load_w(3, 2, 16'd8);  load_w(3, 3, 16'd9);

        #10;

        @(negedge clk);
        act_in = {16'd1, 16'd1, 16'd1, 16'd1};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        begin : wait2
            integer wcnt;
            wcnt = 0;
            while (!valid_out && wcnt < 100) begin
                @(posedge clk); #1;
                wcnt = wcnt + 1;
            end
        end

        if (valid_out &&
            result_out[31:0]   == 32'd12 &&
            result_out[63:32]  == 32'd14 &&
            result_out[95:64]  == 32'd6 &&
            result_out[127:96] == 32'd10) begin
            $display("[PASS] Q4 full matrix: [12, 14, 6, 10]");
            pass_count = pass_count + 1;
        end else if (valid_out) begin
            $display("[FAIL] Q4 full matrix: expected [12,14,6,10], got [%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end else begin
            $display("[FAIL] Q4 full matrix: no valid output");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // Test 3: Q4 with non-standard zero-point offset
        //   scale=3, zero=4
        //   INT4 weights:          Dequant (scale=3, zero=4):
        //   Row 0: 5, 6, 4, 7       →  3,  6,  0,  9
        //   Row 1: 4, 5, 6, 4       →  0,  3,  6,  0
        //   Row 2: 6, 4, 5, 8       →  6,  0,  3, 12
        //   Row 3: 7, 5, 4, 5       →  9,  3,  0,  3
        //   act = [2, 1, 3, 1]
        //   result[0] = 2*3 + 1*0 + 3*6 + 1*9 = 33
        //   result[1] = 2*6 + 1*3 + 3*0 + 1*3 = 18
        //   result[2] = 2*0 + 1*6 + 3*3 + 1*0 = 15
        //   result[3] = 2*9 + 1*0 + 3*12 + 1*3 = 57
        //   Expected: [33, 18, 15, 57]
        // ==================================================================
        @(negedge clk); rst = 1; clear_acc = 1; @(negedge clk); rst = 0; clear_acc = 0; #10;

        $display("\n--- Test 3: Q4 zero-point offset (scale=3, zero=4) ---");
        precision_mode = 2'd1;
        q4_block_scale = 8'd3;
        q4_block_zero  = 4'd4;

        load_w(0, 0, 16'd5); load_w(0, 1, 16'd6); load_w(0, 2, 16'd4); load_w(0, 3, 16'd7);
        load_w(1, 0, 16'd4); load_w(1, 1, 16'd5); load_w(1, 2, 16'd6); load_w(1, 3, 16'd4);
        load_w(2, 0, 16'd6); load_w(2, 1, 16'd4); load_w(2, 2, 16'd5); load_w(2, 3, 16'd8);
        load_w(3, 0, 16'd7); load_w(3, 1, 16'd5); load_w(3, 2, 16'd4); load_w(3, 3, 16'd5);

        #10;

        @(negedge clk);
        act_in = {16'd1, 16'd3, 16'd1, 16'd2};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        begin : wait3
            integer wcnt;
            wcnt = 0;
            while (!valid_out && wcnt < 100) begin
                @(posedge clk); #1;
                wcnt = wcnt + 1;
            end
        end

        if (valid_out &&
            result_out[31:0]   == 32'd33 &&
            result_out[63:32]  == 32'd18 &&
            result_out[95:64]  == 32'd15 &&
            result_out[127:96] == 32'd57) begin
            $display("[PASS] Q4 zero-point offset: [33, 18, 15, 57]");
            pass_count = pass_count + 1;
        end else if (valid_out) begin
            $display("[FAIL] Q4 zero-point offset: expected [33,18,15,57], got [%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end else begin
            $display("[FAIL] Q4 zero-point offset: no valid output");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // Test 4: Q8.8 mode regression (precision_mode=0)
        //   Standard 16-bit weights, same diagonal test as original TB
        //   W = diag([1, 2, 3, 4]), act = [10, 20, 30, 40]
        //   Expected: [10, 40, 90, 160]
        // ==================================================================
        @(negedge clk); rst = 1; clear_acc = 1; @(negedge clk); rst = 0; clear_acc = 0; #10;

        $display("\n--- Test 4: Q8.8 regression (precision_mode=0) ---");
        precision_mode = 2'd0;
        q4_block_scale = 8'd0;
        q4_block_zero  = 4'd0;

        // Only load diagonal — off-diagonal are 0 after reset (zero-skipped in Q8.8)
        load_w(0, 0, 16'd1);
        load_w(1, 1, 16'd2);
        load_w(2, 2, 16'd3);
        load_w(3, 3, 16'd4);

        #10;

        @(negedge clk);
        act_in = {16'd40, 16'd30, 16'd20, 16'd10};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        begin : wait4
            integer wcnt;
            wcnt = 0;
            while (!valid_out && wcnt < 100) begin
                @(posedge clk); #1;
                wcnt = wcnt + 1;
            end
        end

        if (valid_out &&
            result_out[31:0]   == 32'd10 &&
            result_out[63:32]  == 32'd40 &&
            result_out[95:64]  == 32'd90 &&
            result_out[127:96] == 32'd160) begin
            $display("[PASS] Q8.8 regression: [10, 40, 90, 160]");
            pass_count = pass_count + 1;
        end else if (valid_out) begin
            $display("[FAIL] Q8.8 regression: expected [10,40,90,160], got [%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end else begin
            $display("[FAIL] Q8.8 regression: no valid output");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // Summary
        // ==================================================================
        #20;
        $display("\n============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
