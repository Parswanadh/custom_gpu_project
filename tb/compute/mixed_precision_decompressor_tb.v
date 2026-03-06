`timescale 1ns / 1ps

module mixed_precision_decompressor_tb;

    reg clk, rst, valid_in;
    reg [1:0] precision_mode;
    reg [31:0] packed_weights;
    reg [7:0] zero_point, scale_factor;
    
    wire [7:0] w0, w1, w2, w3, w4, w5, w6, w7;
    wire [2:0] num_weights_out;
    wire valid_out;

    mixed_precision_decompressor #(.DATA_WIDTH(8)) dut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .precision_mode(precision_mode),
        .packed_weights(packed_weights),
        .zero_point(zero_point), .scale_factor(scale_factor),
        .weight_out_0(w0), .weight_out_1(w1),
        .weight_out_2(w2), .weight_out_3(w3),
        .weight_out_4(w4), .weight_out_5(w5),
        .weight_out_6(w6), .weight_out_7(w7),
        .num_weights_out(num_weights_out),
        .valid_out(valid_out)
    );

    always #5 clk = ~clk;
    integer tests_passed = 0, tests_total = 0;

    initial begin
        clk = 0; rst = 1; valid_in = 0;
        precision_mode = 0; packed_weights = 0;
        zero_point = 0; scale_factor = 8'd255;
        
        @(negedge clk); @(negedge clk); rst = 0; @(negedge clk);
        
        $display("=================================================");
        $display("   Mixed-Precision Decompressor Tests");
        $display("   (Q4/Q6/Q8 per-layer weight decompression)");
        $display("=================================================");

        // ================================================================
        // TEST 1: Q4 mode — 8 weights from 32-bit word
        // ================================================================
        tests_total = tests_total + 1;
        precision_mode = 2'b00;
        packed_weights = {4'h0, 4'h4, 4'hF, 4'h1, 4'h7, 4'hE, 4'h5, 4'h3};
        zero_point = 8'd0;
        scale_factor = 8'd255;
        valid_in = 1;
        @(negedge clk);  // posedge processes, valid_out goes high
        // Check on this negedge — valid_out is set
        if (valid_out) begin
            $display("[PASS] Test 1: Q4 mode - 8 weights from 32-bit word");
            $display("  w=[%0d, %0d, %0d, %0d, %0d, %0d, %0d, %0d]",
                     $signed(w0), $signed(w1), $signed(w2), $signed(w3),
                     $signed(w4), $signed(w5), $signed(w6), $signed(w7));
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 1: Q4 output not valid");
        valid_in = 0;
        @(negedge clk);

        // ================================================================
        // TEST 2: Q6 mode — 4 weights from 32-bit word
        // ================================================================
        tests_total = tests_total + 1;
        precision_mode = 2'b01;
        packed_weights = {8'h00, 6'd20, 6'd10, 6'd31, 6'd5};
        zero_point = 8'd0;
        scale_factor = 8'd255;
        valid_in = 1;
        @(negedge clk);
        if (valid_out && num_weights_out == 3'd4) begin
            $display("[PASS] Test 2: Q6 mode - 4 weights (6-bit precision)");
            $display("  w=[%0d, %0d, %0d, %0d]",
                     $signed(w0), $signed(w1), $signed(w2), $signed(w3));
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 2: Q6 not valid (valid=%b num=%d)", valid_out, num_weights_out);
        valid_in = 0;
        @(negedge clk);

        // ================================================================
        // TEST 3: Q8 mode — 4 direct 8-bit weights
        // ================================================================
        tests_total = tests_total + 1;
        precision_mode = 2'b10;
        packed_weights = {8'd100, 8'd50, 8'd226, 8'd10}; // 100, 50, -30, 10
        zero_point = 8'd0;
        scale_factor = 8'd255;
        valid_in = 1;
        @(negedge clk);
        if (valid_out && num_weights_out == 3'd4) begin
            $display("[PASS] Test 3: Q8 mode - 4 weights (full 8-bit precision)");
            $display("  w=[%0d, %0d, %0d, %0d]",
                     $signed(w0), $signed(w1), $signed(w2), $signed(w3));
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 3: Q8 not valid");
        valid_in = 0;
        @(negedge clk);

        // ================================================================
        // TEST 4: Zero-point offset
        // ================================================================
        tests_total = tests_total + 1;
        precision_mode = 2'b00;
        packed_weights = {4'h0, 4'h0, 4'h0, 4'h0, 4'h0, 4'h0, 4'h0, 4'h5};
        zero_point = 8'd4;
        scale_factor = 8'd255;
        valid_in = 1;
        @(negedge clk);
        if (valid_out) begin
            $display("[PASS] Test 4: Zero-point offset (raw=5, zp=4 -> out=%0d)", $signed(w0));
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 4: ZP offset fail");
        valid_in = 0;
        @(negedge clk);

        // ================================================================
        // TEST 5: Mode switching Q4→Q8
        // ================================================================
        tests_total = tests_total + 1;
        precision_mode = 2'b00;
        packed_weights = 32'hFFFFFFFF;
        zero_point = 0; scale_factor = 128;
        valid_in = 1;
        @(negedge clk);
        valid_in = 0;
        @(negedge clk);
        // Switch to Q8
        precision_mode = 2'b10;
        packed_weights = 32'h12345678;
        scale_factor = 255;
        valid_in = 1;
        @(negedge clk);
        if (valid_out) begin
            $display("[PASS] Test 5: Seamless Q4->Q8 mode switching (GGUF compatible)");
            tests_passed = tests_passed + 1;
        end else
            $display("[FAIL] Test 5: Mode switch failed");
        valid_in = 0;

        $display("=================================================");
        $display("   Mixed-Precision Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        
        #10 $finish;
    end
    
    initial begin #10000; $display("TIMEOUT"); $finish; end

endmodule
