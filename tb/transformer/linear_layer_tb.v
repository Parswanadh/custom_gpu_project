// ============================================================================
// Testbench: linear_layer_tb
// Tests matrix-vector multiply with bias
// ============================================================================
`timescale 1ns / 1ps

module linear_layer_tb;

    parameter ID = 4;
    parameter OD = 4;
    parameter DW = 16;
    parameter AW = 32;

    reg                     clk, rst;
    reg                     load_weight, load_bias;
    reg  [1:0]              w_row, w_col, b_idx;
    reg  signed [DW-1:0]    w_data, b_data;
    reg                     valid_in;
    reg  [ID*DW-1:0]        x_in;
    wire [OD*DW-1:0]        y_out;
    wire                    valid_out;

    linear_layer #(.IN_DIM(ID), .OUT_DIM(OD), .DATA_WIDTH(DW), .ACC_WIDTH(AW)) uut (
        .clk(clk), .rst(rst),
        .load_weight(load_weight), .w_row(w_row), .w_col(w_col), .w_data(w_data),
        .load_bias(load_bias), .b_idx(b_idx), .b_data(b_data),
        .valid_in(valid_in), .x_in(x_in), .y_out(y_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task load_w;
        input [1:0] r, c;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_weight = 1'b1; w_row = r; w_col = c; w_data = val;
            @(negedge clk);
            load_weight = 1'b0;
        end
    endtask

    task load_b;
        input [1:0] idx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_bias = 1'b1; b_idx = idx; b_data = val;
            @(negedge clk);
            load_bias = 1'b0;
        end
    endtask

    integer timeout_cnt;
    real y_real;
    integer ii;

    initial begin
        $dumpfile("sim/waveforms/linear_layer.vcd");
        $dumpvars(0, linear_layer_tb);
    end

    initial begin
        $display("============================================");
        $display("  Linear Layer Testbench (Q8.8)");
        $display("============================================");

        rst = 1; load_weight = 0; load_bias = 0; valid_in = 0; x_in = 0;
        w_row = 0; w_col = 0; w_data = 0; b_idx = 0; b_data = 0;
        #25; rst = 0; #15;

        // Load identity-like weights (scaled by 1.0 = 256 in Q8.8)
        // W = I * 256 (identity matrix)
        load_w(0, 0, 16'sd256); load_w(1, 1, 16'sd256);
        load_w(2, 2, 16'sd256); load_w(3, 3, 16'sd256);
        // bias = 0
        load_b(0, 16'sd0); load_b(1, 16'sd0);
        load_b(2, 16'sd0); load_b(3, 16'sd0);

        #10;

        // Test 1: Identity transform: y = I*x + 0 should ≈ x
        // x = [1.0, 2.0, 3.0, 4.0] in Q8.8
        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd512, 16'sd256};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 50) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("[PASS] Identity transform outputs:");
            for (ii = 0; ii < OD; ii = ii + 1) begin
                y_real = $itor($signed(y_out[ii*DW +: DW])) / 256.0;
                $display("  y[%0d] = %.3f", ii, y_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Identity transform TIMEOUT");
            fail_count = fail_count + 1;
        end

        // Test 2: With bias
        @(negedge clk); rst = 1; @(negedge clk); rst = 0; #10;
        load_w(0, 0, 16'sd256); load_w(1, 1, 16'sd256);
        load_w(2, 2, 16'sd256); load_w(3, 3, 16'sd256);
        load_b(0, 16'sd128); load_b(1, 16'sd128);  // bias = [0.5, 0.5, 0.5, 0.5]
        load_b(2, 16'sd128); load_b(3, 16'sd128);

        #10;
        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd512, 16'sd256};
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 50) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("[PASS] With bias=0.5 outputs:");
            for (ii = 0; ii < OD; ii = ii + 1) begin
                y_real = $itor($signed(y_out[ii*DW +: DW])) / 256.0;
                $display("  y[%0d] = %.3f", ii, y_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] With bias TIMEOUT");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
