`timescale 1ns / 1ps

module w4a8_decompressor_tb;

    reg clk;
    reg rst_n;
    
    // Inputs
    reg [31:0] packed_w4_in;
    reg signed [7:0] scale_in;
    reg [3:0] zero_point_in;
    reg valid_in;
    
    // Outputs
    wire [63:0] unpacked_w8_out;
    wire valid_out;

    // Instantiate DUT
    w4a8_decompressor #(
        .WEIGHTS_PER_WORD(8),
        .W4_BITS(4),
        .W8_BITS(8)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .packed_w4_in(packed_w4_in),
        .scale_in(scale_in),
        .zero_point_in(zero_point_in),
        .valid_in(valid_in),
        .unpacked_w8_out(unpacked_w8_out),
        .valid_out(valid_out)
    );

    // Clock generation (10ns period)
    always #5 clk = ~clk;

    // Helper to extract 8-bit signed values from 64-bit flat wire
    function integer get_val;
        input integer idx;
        reg [7:0] raw_val;
        begin
            raw_val = unpacked_w8_out[(idx*8) +: 8];
            if (raw_val[7]) get_val = raw_val | ~32'hFF; // Sign extend
            else get_val = raw_val;
        end
    endfunction

    integer tests_passed = 0;
    integer tests_total = 0;
    integer v0, v1, v2, v3, v4, v5, v6, v7;

    initial begin
        clk = 0;
        rst_n = 0;
        packed_w4_in = 0;
        scale_in = 0;
        zero_point_in = 0;
        valid_in = 0;
        
        #20 rst_n = 1;
        
        $display("=================================================");
        $display("   W4A8 Decompressor Tests Starting");
        $display("=================================================");

        // --------------------------------------------------------------------
        // TEST 1: All Zeros, ZP=8, Scale=2
        // Output: (0-8)*2 = -16
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        packed_w4_in = 32'h0000_0000;
        zero_point_in = 4'd8; 
        scale_in = 8'd2;
        
        @(negedge clk); 
        valid_in = 0;
        
        v0 = get_val(0); v7 = get_val(7);
        if (v0 == -16 && v7 == -16 && valid_out == 1) begin
            $display("[PASS] Test 1: Zero input negative mapping correct");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: Expected -16, got %d, %d (valid_out=%b)", v0, v7, valid_out);
        end

        // --------------------------------------------------------------------
        // TEST 2: Max values (15), ZP=8, Scale=2
        // Output: (15-8)*2 = 14
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        packed_w4_in = 32'hFFFF_FFFF;
        zero_point_in = 4'd8;
        scale_in = 8'd2;
        
        @(negedge clk); 
        valid_in = 0;
        
        v0 = get_val(0);
        if (v0 == 14 && valid_out == 1) begin
            $display("[PASS] Test 2: Max positive mapping correct");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: Expected 14, got %d", v0);
        end

        // --------------------------------------------------------------------
        // TEST 3: Mixed weights
        // W = [0, 4, 8, 12, 15, 2, 6, 10]
        // ZP = 8, Scale = 10
        // Expected: [-80, -40, 0, 40, 70, -60, -20, 20]
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        packed_w4_in = {4'd10, 4'd6, 4'd2, 4'd15, 4'd12, 4'd8, 4'd4, 4'd0};
        zero_point_in = 4'd8;
        scale_in = 8'd10;
        
        @(negedge clk); 
        valid_in = 0;
        
        v0 = get_val(0); v1 = get_val(1); v2 = get_val(2); v3 = get_val(3);
        v4 = get_val(4); v5 = get_val(5); v6 = get_val(6); v7 = get_val(7);
        
        if (v0 == -80 && v1 == -40 && v2 == 0 && v3 == 40 &&
            v4 == 70 && v5 == -60 && v6 == -20 && v7 == 20 && valid_out == 1) begin
            $display("[PASS] Test 3: Mixed weight processing correct");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: Mixed weights incorrect.");
            $display("       Got: %d, %d, %d, %d, %d, %d, %d, %d", v0,v1,v2,v3,v4,v5,v6,v7);
        end

        // --------------------------------------------------------------------
        // TEST 4: Saturation (Clamping) testing
        // W0 = 15, ZP = 8, Scale = 20 -> (15-8)*20 = 140 -> clamp to 127
        // W1 = 0, ZP = 8, Scale = 20 -> (0-8)*20 = -160 -> clamp to -128
        // --------------------------------------------------------------------
        tests_total = tests_total + 1;
        
        @(negedge clk);
        valid_in = 1;
        packed_w4_in = {4'd8, 4'd8, 4'd8, 4'd8, 4'd8, 4'd8, 4'd0, 4'd15};
        zero_point_in = 4'd8;
        scale_in = 8'd20;
        
        @(negedge clk); 
        valid_in = 0;
        
        v0 = get_val(0); v1 = get_val(1);
        if (v0 == 127 && v1 == -128 && valid_out == 1) begin
            $display("[PASS] Test 4: Saturation clamping correct (127, -128)");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: Expected 127 and -128, got %d and %d", v0, v1);
        end

        // Summary
        $display("=================================================");
        $display("   W4A8 Decompressor Tests Complete: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        
        #10 $finish;
    end

    // Safety Timeout
    initial begin
        #5000;
        $display("TIMEOUT waiting for test completion");
        $finish;
    end

endmodule
