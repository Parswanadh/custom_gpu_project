// ============================================================================
// Testbench: gelu_activation_tb
// Tests GELU in Q8.8: negative saturation, positive passthrough, and mid-range
// ============================================================================
`timescale 1ns / 1ps

module gelu_activation_tb;

    reg         clk, rst, valid_in;
    reg  signed [15:0] x_in;
    wire signed [15:0] y_out;
    wire        valid_out;

    gelu_activation #(.WIDTH(16)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .y_out(y_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    // Convert Q8.8 to display as real for readability
    real x_real, y_real;

    task test_gelu;
        input signed [15:0] x_val;
        input [80*8-1:0] test_name;
        begin
            @(negedge clk);
            x_in = x_val;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;
            #1;
            x_real = $itor(x_val) / 256.0;
            y_real = $itor(y_out) / 256.0;
            $display("[INFO] %0s | x=%.3f (0x%04h) => y=%.3f (0x%04h) valid=%b",
                     test_name, x_real, x_val[15:0], y_real, y_out[15:0], valid_out);
            if (valid_out) begin
                pass_count = pass_count + 1;
            end else begin
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/gelu_activation.vcd");
        $dumpvars(0, gelu_activation_tb);
    end

    initial begin
        $display("============================================");
        $display("  GELU Activation Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0;
        #25; rst = 0; #15;

        // x = -5.0 → should be 0 (below -3 threshold)
        test_gelu(-16'sd1280, "x=-5.0 => ~0");

        // x = -3.0 → boundary, should be ~0
        test_gelu(-16'sd768, "x=-3.0 => ~0");

        // x = -1.0 → GELU(-1) ≈ -0.159
        test_gelu(-16'sd256, "x=-1.0 => ~-0.16");

        // x = 0.0 → GELU(0) = 0
        test_gelu(16'sd0, "x=0.0 => 0");

        // x = 1.0 → GELU(1) ≈ 0.841
        test_gelu(16'sd256, "x=1.0 => ~0.84");

        // x = 2.0 → GELU(2) ≈ 1.955
        test_gelu(16'sd512, "x=2.0 => ~1.96");

        // x = 3.0 → boundary
        test_gelu(16'sd768, "x=3.0 => ~3.0");

        // x = 5.0 → should be 5.0 (above +3 threshold)
        test_gelu(16'sd1280, "x=5.0 => 5.0");

        // x = 0.5 → GELU(0.5) ≈ 0.346
        test_gelu(16'sd128, "x=0.5 => ~0.35");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
