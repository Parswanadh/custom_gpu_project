`timescale 1ns/1ps

module ternary_unpacker_tb;

    reg  [15:0] packed_word;
    wire [15:0] unpacked_weights;
    reg  clk;

    // Instantiate the Unit Under Test (UUT)
    ternary_unpacker uut (
        .packed_word(packed_word),
        .unpacked_weights(unpacked_weights)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    integer i;
    reg [1:0] expected_weight;
    reg failed;

    initial begin
        failed = 0;
        packed_word = 16'h0000;
        
        $display("Starting Ternary Unpacker Testbench...");
        
        // Test case 1: All zeros
        @(negedge clk);
        packed_word = 16'h0000;
        @(negedge clk);
        if (unpacked_weights !== 16'h0000) begin
            $display("FAIL: Test Case 1 - Expected 0000, got %h", unpacked_weights);
            failed = 1;
        end

        // Test case 2: Alternating 01 and 10
        // 10 01 10 01 10 01 10 01 = 2 1 2 1 2 1 2 1 in hex-ish
        // 1001 = 9, 1001 = 9 ... -> 16'h9999
        @(negedge clk);
        packed_word = 16'h9999;
        @(negedge clk);
        if (unpacked_weights !== 16'h9999) begin
            $display("FAIL: Test Case 2 - Expected 9999, got %h", unpacked_weights);
            failed = 1;
        end

        // Test case 3: Incrementing pattern
        // W0=0, W1=1, W2=2, W3=3, W4=0, W5=1, W6=2, W7=3
        // Binary: 11 10 01 00 11 10 01 00
        // Hex: E 4 E 4 -> 16'hE4E4
        @(negedge clk);
        packed_word = 16'hE4E4;
        @(negedge clk);
        if (unpacked_weights !== 16'hE4E4) begin
            $display("FAIL: Test Case 3 - Expected E4E4, got %h", unpacked_weights);
            failed = 1;
        end

        // Detailed check for Test Case 3
        for (i = 0; i < 8; i = i + 1) begin
            expected_weight = (i % 4);
            if (unpacked_weights[i*2 +: 2] !== expected_weight) begin
                $display("FAIL: Weight %0d mismatch. Expected %b, got %b", i, expected_weight, unpacked_weights[i*2 +: 2]);
                failed = 1;
            end
        end

        if (!failed) begin
            $display("PASS: Ternary Unpacker Testbench.");
        end else begin
            $display("FAILURE: Ternary Unpacker Testbench.");
        end
        
        $finish;
    end

    initial begin
        $dumpfile("ternary_unpacker_tb.vcd");
        $dumpvars(0, ternary_unpacker_tb);
    end

endmodule
