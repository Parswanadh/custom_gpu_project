`timescale 1ns / 1ps

module q4_weight_pipeline_tb;

    reg clk, rst, start;
    reg signed [7:0] activation_in;
    wire signed [31:0] mac_result;
    wire done;
    wire [31:0] weights_processed;

    q4_weight_pipeline #(
        .NUM_WEIGHTS(32),
        .GROUP_SIZE(8)
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .activation_in(activation_in),
        .mac_result(mac_result),
        .done(done),
        .weights_processed(weights_processed)
    );

    always #5 clk = ~clk;
    integer tests_passed = 0, tests_total = 0;
    reg test_done;

    task wait_for_done;
        input integer max_cycles;
        output reg success;
        integer count;
        begin
            success = 0;
            for (count = 0; count < max_cycles; count = count + 1) begin
                @(negedge clk);
                if (done) begin
                    success = 1;
                    count = max_cycles;
                end
            end
        end
    endtask

    task run_case_exact;
        input signed [7:0] act_value;
        input integer expected_mac;
        input [255:0] test_name;
        begin
            tests_total = tests_total + 1;
            activation_in = act_value;
            start = 1;
            @(negedge clk);
            start = 0;

            wait_for_done(500, test_done);

            if (test_done && (weights_processed == 32) && ($signed(mac_result) == expected_mac)) begin
                $display("[PASS] %0s: MAC=%0d (expected %0d), processed=%0d",
                         test_name, $signed(mac_result), expected_mac, weights_processed);
                tests_passed = tests_passed + 1;
            end else begin
                $display("[FAIL] %0s: done=%b processed=%0d MAC=%0d expected=%0d",
                         test_name, done, weights_processed, $signed(mac_result), expected_mac);
            end
        end
    endtask

    initial begin
        clk = 0; rst = 1; start = 0; activation_in = 0;
        @(negedge clk); @(negedge clk); rst = 0; @(negedge clk);

        $display("=================================================");
        $display("   Q4 Weight Pipeline Tests");
        $display("   (End-to-End INT4 → MAC Inference Pipeline)");
        $display("=================================================");

        // ================================================================
        // TEST 1: Default quant params (zp=0, scale=1), activation=1
        // Expected MAC = 61
        // ================================================================
        run_case_exact(8'd1, 61, "Test 1 default");

        // ================================================================
        // TEST 2: Default quant params, activation=10
        // Expected MAC = 610
        // ================================================================
        run_case_exact(8'd10, 610, "Test 2 default x10");

        // ================================================================
        // TEST 3: Per-group quant params applied before MAC
        // g0: zp=1 sc=1 -> sum 9
        // g1: zp=2 sc=2 -> sum 0
        // g2: zp=0 sc=3 -> sum 12
        // g3: zp=3 sc=1 -> sum 0
        // Total = 21, activation=4 -> expected 84
        // ================================================================
        @(negedge clk);
        dut.group_zp[0] = 8'd1;  dut.group_scale[0] = 8'd1;
        dut.group_zp[1] = 8'd2;  dut.group_scale[1] = 8'd2;
        dut.group_zp[2] = 8'd0;  dut.group_scale[2] = 8'd3;
        dut.group_zp[3] = 8'd3;  dut.group_scale[3] = 8'd1;
        run_case_exact(8'd4, 84, "Test 3 per-group");

        // ================================================================
        // TEST 4: Distinct per-group params to verify group indexing
        // g0: zp=0 sc=1 -> sum 17
        // g1: zp=1 sc=2 -> sum 16
        // g2: zp=0 sc=4 -> sum 16
        // g3: zp=2 sc=1 -> sum 8
        // Total = 57, activation=2 -> expected 114
        // ================================================================
        @(negedge clk);
        dut.group_zp[0] = 8'd0;  dut.group_scale[0] = 8'd1;
        dut.group_zp[1] = 8'd1;  dut.group_scale[1] = 8'd2;
        dut.group_zp[2] = 8'd0;  dut.group_scale[2] = 8'd4;
        dut.group_zp[3] = 8'd2;  dut.group_scale[3] = 8'd1;
        run_case_exact(8'd2, 114, "Test 4 index check");

        // ================================================================
        // TEST 5: Signed activation path (negative activation)
        // default group params -> sum(weights)=61, activation=-1 -> expected -61
        // ================================================================
        @(negedge clk);
        dut.group_zp[0] = 8'd0;  dut.group_scale[0] = 8'd1;
        dut.group_zp[1] = 8'd0;  dut.group_scale[1] = 8'd1;
        dut.group_zp[2] = 8'd0;  dut.group_scale[2] = 8'd1;
        dut.group_zp[3] = 8'd0;  dut.group_scale[3] = 8'd1;
        run_case_exact(-8'sd1, -61, "Test 5 signed activation");

        $display("=================================================");
        $display("   Q4 Pipeline Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");

        if (tests_passed != tests_total)
            $fatal(1, "Q4 pipeline exact-MAC tests failed");
        
        #10 $finish;
    end
    
    initial begin
        #100000;
        $fatal(1, "TIMEOUT");
    end

endmodule
