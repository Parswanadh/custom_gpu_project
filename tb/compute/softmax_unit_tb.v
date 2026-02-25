// ============================================================================
// Testbench: softmax_unit_tb
// Tests softmax on small vectors, checks probability distribution
// ============================================================================
`timescale 1ns / 1ps

module softmax_unit_tb;

    parameter VL = 4;   // Use 4-element vectors for testing
    parameter DW = 16;

    reg                     clk, rst, valid_in;
    reg  [VL*DW-1:0]        x_in;
    wire [VL*8-1:0]         prob_out;
    wire                    valid_out;

    softmax_unit #(.VECTOR_LEN(VL), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .prob_out(prob_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer i;
    integer prob_sum;

    task run_softmax;
        input signed [DW-1:0] v0, v1, v2, v3;
        input [80*8-1:0] test_name;
        integer timeout;
        begin
            @(negedge clk);
            x_in = {v3, v2, v1, v0};  // Pack vector
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            // Wait for output
            timeout = 0;
            while (!valid_out && timeout < 50) begin
                @(negedge clk);
                timeout = timeout + 1;
            end

            if (valid_out) begin
                prob_sum = 0;
                for (i = 0; i < VL; i = i + 1)
                    prob_sum = prob_sum + prob_out[i*8 +: 8];

                $display("[INFO] %0s | probs=[%0d, %0d, %0d, %0d] sum=%0d",
                         test_name,
                         prob_out[7:0], prob_out[15:8], prob_out[23:16], prob_out[31:24],
                         prob_sum);

                // Check: probabilities should sum to roughly 256 (±20 for rounding)
                if (prob_sum >= 230 && prob_sum <= 280) begin
                    $display("[PASS] %0s | sum close to 256", test_name);
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] %0s | sum=%0d (expected ~256)", test_name, prob_sum);
                    fail_count = fail_count + 1;
                end
            end else begin
                $display("[FAIL] %0s | TIMEOUT", test_name);
                fail_count = fail_count + 1;
            end

            repeat(3) @(negedge clk);
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/softmax_unit.vcd");
        $dumpvars(0, softmax_unit_tb);
    end

    initial begin
        $display("============================================");
        $display("  Softmax Unit Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0;
        #25; rst = 0; #15;

        // Test 1: Equal values → uniform distribution
        // All 1.0 (256 in Q8.8) → each should be ~64 (=256/4)
        run_softmax(16'sd256, 16'sd256, 16'sd256, 16'sd256, "Uniform [1,1,1,1]");

        // Test 2: One dominant value
        // [0, 0, 0, 3.0] → last should dominate
        run_softmax(16'sd0, 16'sd0, 16'sd0, 16'sd768, "Dominant [0,0,0,3]");

        // Test 3: Graduated values
        // [0, 1.0, 2.0, 3.0]
        run_softmax(16'sd0, 16'sd256, 16'sd512, 16'sd768, "Graduated [0,1,2,3]");

        // Test 4: All zeros
        run_softmax(16'sd0, 16'sd0, 16'sd0, 16'sd0, "All zeros [0,0,0,0]");

        // Test 5: Negative values
        run_softmax(-16'sd256, -16'sd512, 16'sd0, 16'sd256, "Mixed [-1,-2,0,1]");

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
