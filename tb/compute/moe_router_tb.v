`timescale 1ns / 1ps

module moe_router_tb;

    reg clk;
    reg rst_n;
    
    // Inputs
    reg [63:0] scores_in;
    reg valid_in;
    
    // Outputs
    wire [1:0] expert_id_out;
    wire [3:0] expert_mask_out;
    wire valid_out;

    // Instantiate DUT
    moe_router #(
        .NUM_EXPERTS(4),
        .SCORE_WIDTH(16)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .scores_in(scores_in),
        .valid_in(valid_in),
        .expert_id_out(expert_id_out),
        .expert_mask_out(expert_mask_out),
        .valid_out(valid_out)
    );

    // Clock generation (10ns period)
    always #5 clk = ~clk;

    integer tests_passed = 0;
    integer tests_total = 0;

    initial begin
        clk = 0;
        rst_n = 0;
        scores_in = 0;
        valid_in = 0;
        
        #20 rst_n = 1;
        
        $display("=================================================");
        $display("   MoE Top-1 Router Tests Starting");
        $display("   (4 Experts, 16-bit Signed Gating Scores)");
        $display("=================================================");

        // --------------------------------------------------------------------
        // TEST 1: Expert 2 is highest
        // scores: e3=-5, e2=50, e1=10, e0=0
        // Expected ID=2, Mask=0100 (4'b0100)
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        // Cycle 0: Provide inputs
        @(negedge clk);
        valid_in = 1;
        scores_in = {-16'sd5, 16'sd50, 16'sd10, 16'sd0};
        
        // Cycle 1: Posedge latched inputs, drops valid_in
        @(negedge clk); 
        valid_in = 0;
        
        // Verify output which is registered on posedge
        if (expert_id_out == 2 && expert_mask_out == 4'b0100 && valid_out == 1) begin
            $display("[PASS] Test 1: Expert 2 correctly selected");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: Expected 2 / 0100, got %d / %b (valid_out=%b)", expert_id_out, expert_mask_out, valid_out);
        end

        // --------------------------------------------------------------------
        // TEST 2: Expert 0 is highest (all negative)
        // scores: e3=-100, e2=-50, e1=-25, e0=-10
        // Expected ID=0, Mask=0001
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        scores_in = {-16'sd100, -16'sd50, -16'sd25, -16'sd10};
        
        @(negedge clk); 
        valid_in = 0;
        
        if (expert_id_out == 0 && expert_mask_out == 4'b0001 && valid_out == 1) begin
            $display("[PASS] Test 2: Expert 0 (all-negative scenario) correctly selected");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: Expected 0 / 0001, got %d / %b", expert_id_out, expert_mask_out);
        end

        // --------------------------------------------------------------------
        // TEST 3: Expert 3 is highest
        // scores: e3=32000, e2=5, e1=500, e0=100
        // Expected ID=3, Mask=1000
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        scores_in = {16'sd32000, 16'sd5, 16'sd500, 16'sd100};
        
        @(negedge clk); 
        valid_in = 0;
        
        if (expert_id_out == 3 && expert_mask_out == 4'b1000 && valid_out == 1) begin
            $display("[PASS] Test 3: Expert 3 properly selected");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: Expected 3 / 1000, got %d / %b", expert_id_out, expert_mask_out);
        end
        
        // --------------------------------------------------------------------
        // TEST 4: Tied scores (e1 = e2 = 50)
        // Expected: Should deterministically pick one (our logic favors lower ID due to how pairs resolve, but either is fine)
        // Here, 01 compares 0 and 1 -> gets 1. 23 compares 2 and 3 -> gets 2.
        // Final compares 1 and 2 -> if (s1 > s2) gets 1, else 2. So if tied, it picks 2.
        // Let's test the tie-breaker behavior to ensure it doesn't crash.
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        scores_in = {16'sd0, 16'sd50, 16'sd50, 16'sd0};
        
        @(negedge clk); 
        valid_in = 0;
        
        if ((expert_id_out == 1 || expert_id_out == 2) && valid_out == 1) begin
            $display("[PASS] Test 4: Tie-breaker resolved safely (Picked Expert %d)", expert_id_out);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: Tie failed, got ID %d valid %b", expert_id_out, valid_out);
        end

        // Summary
        $display("=================================================");
        $display("   MoE Router Tests Complete: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        
        #10 $finish;
    end
    
    // Safety timeout
    initial begin
        #5000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
