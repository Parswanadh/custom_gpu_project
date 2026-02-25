// ============================================================================
// Testbench: systolic_array_tb
// Tests: 4x4 matrix-vector multiply with known values
// ============================================================================
`timescale 1ns / 1ps

module systolic_array_tb;

    parameter N  = 4;
    parameter DW = 16;
    parameter AW = 32;

    reg                     clk, rst;
    reg                     load_weight;
    reg  [1:0]              weight_row, weight_col;
    reg  [DW-1:0]           weight_data;
    reg                     valid_in;
    reg  [N*DW-1:0]         act_in;
    wire [N*AW-1:0]         result_out;
    wire                    valid_out;

    systolic_array #(.ARRAY_SIZE(N), .DATA_WIDTH(DW), .ACC_WIDTH(AW)) uut (
        .clk(clk), .rst(rst),
        .load_weight(load_weight), .weight_row(weight_row), .weight_col(weight_col), .weight_data(weight_data),
        .valid_in(valid_in), .act_in(act_in),
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
        $dumpfile("sim/waveforms/systolic_array.vcd");
        $dumpvars(0, systolic_array_tb);
    end

    initial begin
        $display("============================================");
        $display("  Systolic Array (4x4) Testbench");
        $display("============================================");

        rst = 1; load_weight = 0; valid_in = 0; act_in = 0;
        weight_row = 0; weight_col = 0; weight_data = 0;
        #25; rst = 0; #15;

        // Load 4x4 identity-like weight matrix:
        // W = [[1,0,0,0],
        //      [0,2,0,0],
        //      [0,0,3,0],
        //      [0,0,0,4]]
        load_w(0, 0, 16'd1);
        load_w(1, 1, 16'd2);
        load_w(2, 2, 16'd3);
        load_w(3, 3, 16'd4);

        #10;

        // Activation vector: [10, 20, 30, 40]
        // Expected: [10*1, 20*2, 30*3, 40*4] = [10, 40, 90, 160]
        @(negedge clk);
        act_in = {16'd40, 16'd30, 16'd20, 16'd10};  // packed little-endian
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;
        #1;

        if (valid_out &&
            result_out[31:0]   == 32'd10 &&
            result_out[63:32]  == 32'd40 &&
            result_out[95:64]  == 32'd90 &&
            result_out[127:96] == 32'd160) begin
            $display("[PASS] Diagonal matrix: [10,40,90,160]");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Diagonal matrix | got=[%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end

        // Reset and load a full matrix:
        // W = [[1,2,3,4],
        //      [5,6,7,8],
        //      [9,10,11,12],
        //      [13,14,15,16]]
        @(negedge clk); rst = 1; @(negedge clk); rst = 0; #10;

        load_w(0, 0, 16'd1);  load_w(0, 1, 16'd2);  load_w(0, 2, 16'd3);  load_w(0, 3, 16'd4);
        load_w(1, 0, 16'd5);  load_w(1, 1, 16'd6);  load_w(1, 2, 16'd7);  load_w(1, 3, 16'd8);
        load_w(2, 0, 16'd9);  load_w(2, 1, 16'd10); load_w(2, 2, 16'd11); load_w(2, 3, 16'd12);
        load_w(3, 0, 16'd13); load_w(3, 1, 16'd14); load_w(3, 2, 16'd15); load_w(3, 3, 16'd16);

        #10;

        // Activation vector: [1, 1, 1, 1]
        // Expected: col sums = [1+5+9+13, 2+6+10+14, 3+7+11+15, 4+8+12+16] = [28, 32, 36, 40]
        @(negedge clk);
        act_in = {16'd1, 16'd1, 16'd1, 16'd1};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;
        #1;

        if (valid_out &&
            result_out[31:0]   == 32'd28 &&
            result_out[63:32]  == 32'd32 &&
            result_out[95:64]  == 32'd36 &&
            result_out[127:96] == 32'd40) begin
            $display("[PASS] Full matrix col sums: [28,32,36,40]");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Full matrix | got=[%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end

        // Test with zeros in activation (sparsity test)
        @(negedge clk);
        act_in = {16'd0, 16'd0, 16'd1, 16'd0};  // only act[1]=1
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;
        #1;

        // Expected: row 1 only: [5, 6, 7, 8]
        if (valid_out &&
            result_out[31:0]   == 32'd5 &&
            result_out[63:32]  == 32'd6 &&
            result_out[95:64]  == 32'd7 &&
            result_out[127:96] == 32'd8) begin
            $display("[PASS] Sparse activation [0,1,0,0]: [5,6,7,8]");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Sparse activation | got=[%0d,%0d,%0d,%0d]",
                     result_out[31:0], result_out[63:32], result_out[95:64], result_out[127:96]);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
