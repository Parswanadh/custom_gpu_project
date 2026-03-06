// ============================================================================
// Testbench: sparsity_decoder_2_4_tb
// Tests 2:4 structured sparsity decoder with all bitmap patterns
// ============================================================================
`timescale 1ns / 1ps

module sparsity_decoder_2_4_tb;

    parameter DW = 16;

    reg                         clk, rst, valid_in;
    reg  [4*DW-1:0]            weights_in, activations_in;
    reg  [1:0]                 sparsity_bitmap;
    wire [1:0]                 nz_idx_0, nz_idx_1;
    wire signed [DW-1:0]       nz_weight_0, nz_weight_1;
    wire signed [DW-1:0]       nz_act_0, nz_act_1;
    wire signed [2*DW-1:0]     result;
    wire                       valid_out;
    wire [31:0]                skipped_count, computed_count;

    sparsity_decoder_2_4 #(.DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .weights_in(weights_in), .activations_in(activations_in),
        .sparsity_bitmap(sparsity_bitmap),
        .nz_idx_0(nz_idx_0), .nz_idx_1(nz_idx_1),
        .nz_weight_0(nz_weight_0), .nz_weight_1(nz_weight_1),
        .nz_act_0(nz_act_0), .nz_act_1(nz_act_1),
        .result(result), .valid_out(valid_out),
        .skipped_count(skipped_count), .computed_count(computed_count)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task test_sparsity;
        input signed [DW-1:0] w0, w1, w2, w3;
        input signed [DW-1:0] a0, a1, a2, a3;
        input [1:0] bitmap;
        input signed [2*DW-1:0] expected;
        input [40*8-1:0] test_name;
        integer timeout;
        begin
            @(negedge clk);
            weights_in = {w3, w2, w1, w0};
            activations_in = {a3, a2, a1, a0};
            sparsity_bitmap = bitmap;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            // Wait for valid_out
            timeout = 0;
            while (!valid_out && timeout < 10) begin
                @(posedge clk); #1;
                timeout = timeout + 1;
            end

            if (valid_out && result == expected) begin
                $display("[PASS] %0s | result=%0d (expected %0d), nz=[%0d,%0d]",
                    test_name, result, expected, nz_idx_0, nz_idx_1);
                pass_count = pass_count + 1;
            end else if (valid_out) begin
                $display("[FAIL] %0s | result=%0d (expected %0d)", test_name, result, expected);
                fail_count = fail_count + 1;
            end else begin
                $display("[FAIL] %0s | no valid output", test_name);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/sparsity_decoder_2_4.vcd");
        $dumpvars(0, sparsity_decoder_2_4_tb);
    end

    initial begin
        $display("============================================");
        $display("  2:4 Structured Sparsity Decoder TB");
        $display("============================================");

        rst = 1; valid_in = 0; weights_in = 0; activations_in = 0; sparsity_bitmap = 0;
        #25; rst = 0; #15;

        // Weights: [10, 20, 0, 0], Activations: [3, 4, 5, 6]
        // Bitmap 00: positions [0,1] active → 10*3 + 20*4 = 30 + 80 = 110
        test_sparsity(16'sd10, 16'sd20, 16'sd0, 16'sd0,
                      16'sd3, 16'sd4, 16'sd5, 16'sd6,
                      2'b00, 32'sd110, "Bitmap 00: [0,1] active");

        // Weights: [5, 0, 7, 0], Activations: [2, 3, 4, 5]
        // Bitmap 01: positions [0,2] active → 5*2 + 7*4 = 10 + 28 = 38
        test_sparsity(16'sd5, 16'sd0, 16'sd7, 16'sd0,
                      16'sd2, 16'sd3, 16'sd4, 16'sd5,
                      2'b01, 32'sd38, "Bitmap 01: [0,2] active");

        // Weights: [0, 8, 6, 0], Activations: [1, 2, 3, 4]
        // Bitmap 10: positions [1,2] active → 8*2 + 6*3 = 16 + 18 = 34
        test_sparsity(16'sd0, 16'sd8, 16'sd6, 16'sd0,
                      16'sd1, 16'sd2, 16'sd3, 16'sd4,
                      2'b10, 32'sd34, "Bitmap 10: [1,2] active");

        // Weights: [9, 0, 0, 3], Activations: [4, 5, 6, 7]
        // Bitmap 11: positions [0,3] active → 9*4 + 3*7 = 36 + 21 = 57
        test_sparsity(16'sd9, 16'sd0, 16'sd0, 16'sd3,
                      16'sd4, 16'sd5, 16'sd6, 16'sd7,
                      2'b11, 32'sd57, "Bitmap 11: [0,3] active");

        // Negative weights test
        // Bitmap 00: positions [0,1] → (-5)*3 + 10*(-2) = -15 + (-20) = -35
        test_sparsity(-16'sd5, 16'sd10, 16'sd0, 16'sd0,
                      16'sd3, -16'sd2, 16'sd7, 16'sd8,
                      2'b00, -32'sd35, "Negative weights");

        // Verify counters: 5 operations × 2 computed = 10, 5 × 2 skipped = 10
        if (computed_count == 32'd10 && skipped_count == 32'd10) begin
            $display("[PASS] Counters: computed=%0d, skipped=%0d", computed_count, skipped_count);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Counters: computed=%0d (exp 10), skipped=%0d (exp 10)",
                computed_count, skipped_count);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
