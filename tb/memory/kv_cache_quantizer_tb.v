`timescale 1ns / 1ps
module kv_cache_quantizer_tb;
    parameter VEC_LEN=4, DATA_WIDTH=16;
    reg clk, rst, quant_valid, dequant_valid;
    reg [VEC_LEN*DATA_WIDTH-1:0] kv_in;
    wire [VEC_LEN*4-1:0] kv_quantized;
    wire signed [DATA_WIDTH-1:0] quant_min;
    wire [DATA_WIDTH-1:0] quant_scale;
    wire quant_done;
    reg [VEC_LEN*4-1:0] kv_q_in;
    reg signed [DATA_WIDTH-1:0] dequant_min;
    reg [DATA_WIDTH-1:0] dequant_scale;
    wire [VEC_LEN*DATA_WIDTH-1:0] kv_dequantized;
    wire dequant_done;
    wire [31:0] bytes_saved;

    kv_cache_quantizer #(.VEC_LEN(VEC_LEN)) dut (
        .clk(clk), .rst(rst), .quant_valid(quant_valid), .kv_in(kv_in),
        .kv_quantized(kv_quantized), .quant_min(quant_min), .quant_scale(quant_scale),
        .quant_done(quant_done), .dequant_valid(dequant_valid),
        .kv_q_in(kv_q_in), .dequant_min(dequant_min), .dequant_scale(dequant_scale),
        .kv_dequantized(kv_dequantized), .dequant_done(dequant_done),
        .bytes_saved(bytes_saved));

    always #5 clk = ~clk;
    integer tp=0, tt=0;

    initial begin
        clk=0; rst=1; quant_valid=0; dequant_valid=0; kv_in=0;
        kv_q_in=0; dequant_min=0; dequant_scale=0;
        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);

        $display("=================================================");
        $display("   KV Cache INT4 Quantizer Tests");
        $display("   Paper: QuantSpec (Apple, ICML 2025)");
        $display("   Impact: 4x KV cache memory savings");
        $display("=================================================");

        // TEST 1: Quantize [100, 200, 300, 400]
        tt = tt + 1;
        kv_in = {16'sd400, 16'sd300, 16'sd200, 16'sd100};
        quant_valid = 1; @(negedge clk);
        if (quant_done) begin
            $display("[PASS] Test 1: Quantized to Q4 — min=%0d, scale=%0d", 
                $signed(quant_min), quant_scale);
            $display("         Q4 values: [%0d, %0d, %0d, %0d]",
                kv_quantized[3:0], kv_quantized[7:4], kv_quantized[11:8], kv_quantized[15:12]);
            tp = tp + 1;
        end else $display("[FAIL] Test 1");
        quant_valid = 0; @(negedge clk);

        // TEST 2: Dequantize back
        tt = tt + 1;
        kv_q_in = kv_quantized;
        dequant_min = quant_min;
        dequant_scale = quant_scale;
        dequant_valid = 1; @(negedge clk);
        if (dequant_done) begin
            $display("[PASS] Test 2: Dequantized — [%0d, %0d, %0d, %0d]",
                $signed(kv_dequantized[15:0]), $signed(kv_dequantized[31:16]),
                $signed(kv_dequantized[47:32]), $signed(kv_dequantized[63:48]));
            tp = tp + 1;
        end else $display("[FAIL] Test 2");
        dequant_valid = 0; @(negedge clk);

        // TEST 3: Memory savings tracking
        tt = tt + 1;
        if (bytes_saved == VEC_LEN) begin
            $display("[PASS] Test 3: bytes_saved = %0d (16-bit → 4-bit = 4x compression)", bytes_saved);
            tp = tp + 1;
        end else $display("[FAIL] Test 3: bytes_saved=%0d", bytes_saved);

        // TEST 4: All same values → should quantize to same Q4 value
        tt = tt + 1;
        kv_in = {16'sd256, 16'sd256, 16'sd256, 16'sd256};
        quant_valid = 1; @(negedge clk);
        quant_valid = 0;
        if (quant_done && kv_quantized[3:0] == kv_quantized[7:4]) begin
            $display("[PASS] Test 4: Uniform values → uniform Q4 output");
            tp = tp + 1;
        end else $display("[FAIL] Test 4");

        $display("=================================================");
        $display("   KV Quantizer Tests: %0d / %0d PASSED", tp, tt);
        $display("=================================================");
        #10 $finish;
    end
    initial begin #10000; $display("TIMEOUT"); $finish; end
endmodule
