// ============================================================================
// Testbench: layer_norm_tb
// Tests layer normalization with known input vectors
// ============================================================================
`timescale 1ns / 1ps

module layer_norm_tb;

    parameter D = 4;
    parameter DW = 16;

    reg                     clk, rst, valid_in;
    reg  [D*DW-1:0]         x_in, gamma_in, beta_in;
    wire [D*DW-1:0]         y_out;
    wire                    valid_out;

    layer_norm #(.DIM(D), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .gamma_in(gamma_in), .beta_in(beta_in),
        .y_out(y_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer i;
    real y_real;

    task run_layernorm;
        input [D*DW-1:0] x, g, b;
        input [80*8-1:0] test_name;
        integer timeout;
        begin
            @(negedge clk);
            x_in = x; gamma_in = g; beta_in = b;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            timeout = 0;
            while (!valid_out && timeout < 50) begin
                @(negedge clk);
                timeout = timeout + 1;
            end

            if (valid_out) begin
                $display("[PASS] %0s | outputs:", test_name);
                for (i = 0; i < D; i = i + 1) begin
                    y_real = $itor($signed(y_out[i*DW +: DW])) / 256.0;
                    $display("  y[%0d] = %.3f (0x%04h)", i, y_real, y_out[i*DW +: DW]);
                end
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s | TIMEOUT", test_name);
                fail_count = fail_count + 1;
            end
            repeat(3) @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/layer_norm.vcd");
        $dumpvars(0, layer_norm_tb);
    end

    initial begin
        $display("============================================");
        $display("  Layer Normalization Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0; gamma_in = 0; beta_in = 0;
        #25; rst = 0; #15;

        // Test 1: Simple vector [1, 2, 3, 4] with gamma=1, beta=0
        // Mean = 2.5, should normalize to roughly [-1.3, -0.4, 0.4, 1.3]
        run_layernorm(
            {16'sd1024, 16'sd768, 16'sd512, 16'sd256}, // [1.0, 2.0, 3.0, 4.0]
            {16'sd256, 16'sd256, 16'sd256, 16'sd256},   // gamma = [1,1,1,1]
            {16'sd0, 16'sd0, 16'sd0, 16'sd0},           // beta = [0,0,0,0]
            "x=[1,2,3,4] g=1 b=0"
        );

        // Test 2: All same values → all outputs should be ~beta
        run_layernorm(
            {16'sd512, 16'sd512, 16'sd512, 16'sd512},   // [2.0, 2.0, 2.0, 2.0]
            {16'sd256, 16'sd256, 16'sd256, 16'sd256},   // gamma = [1,1,1,1]
            {16'sd128, 16'sd128, 16'sd128, 16'sd128},   // beta = [0.5,0.5,0.5,0.5]
            "x=[2,2,2,2] g=1 b=0.5"
        );

        // Test 3: With gamma scaling and beta offset
        run_layernorm(
            {16'sd768, 16'sd256, 16'sd512, 16'sd0},     // [0.0, 2.0, 1.0, 3.0]
            {16'sd512, 16'sd512, 16'sd512, 16'sd512},   // gamma = [2,2,2,2]
            {16'sd256, 16'sd256, 16'sd256, 16'sd256},   // beta = [1,1,1,1]
            "x=[0,2,1,3] g=2 b=1"
        );

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
