// ============================================================================
// Testbench: activation_compressor_tb
// Tests compression and decompression of Q8.8 activations
// ============================================================================
`timescale 1ns / 1ps

module activation_compressor_tb;

    parameter VL = 4;
    parameter DW = 16;

    reg                     clk, rst;
    reg                     compress_valid, decompress_valid;
    reg  [VL*DW-1:0]       data_in;
    wire [VL*8-1:0]        compressed_out;
    wire [7:0]             scale_out;
    wire                   compress_done;
    reg  [VL*8-1:0]        compressed_in;
    reg  [7:0]             scale_in;
    wire [VL*DW-1:0]       decompressed_out;
    wire                   decompress_done;
    wire [31:0]            total_compressions, total_bytes_saved;

    activation_compressor #(.VECTOR_LEN(VL), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst),
        .compress_valid(compress_valid), .data_in(data_in),
        .compressed_out(compressed_out), .scale_out(scale_out),
        .compress_done(compress_done),
        .decompress_valid(decompress_valid),
        .compressed_in(compressed_in), .scale_in(scale_in),
        .decompressed_out(decompressed_out), .decompress_done(decompress_done),
        .total_compressions(total_compressions),
        .total_bytes_saved(total_bytes_saved)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer i;

    initial begin
        $dumpfile("sim/waveforms/activation_compressor.vcd");
        $dumpvars(0, activation_compressor_tb);
    end

    initial begin
        $display("============================================");
        $display("  Activation Compressor Testbench");
        $display("============================================");

        rst = 1; compress_valid = 0; decompress_valid = 0;
        data_in = 0; compressed_in = 0; scale_in = 0;
        #25; rst = 0; #15;

        // Test 1: Compress small values (within 8-bit range)
        $display("[1] Compressing small values...");
        @(negedge clk);
        data_in = {16'sd50, 16'sd30, 16'sd20, 16'sd10};
        compress_valid = 1'b1;
        @(negedge clk);
        compress_valid = 1'b0;

        // Wait for compress_done
        begin : wait1
            integer t;
            t = 0;
            while (!compress_done && t < 20) begin
                @(posedge clk); #1;
                t = t + 1;
            end
        end

        if (compress_done) begin
            $display("[PASS] Compression completed, scale=%0d", scale_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Compression did not complete");
            fail_count = fail_count + 1;
        end

        repeat(2) @(posedge clk);

        // Test 2: Compress large values (require scaling)
        $display("[2] Compressing large values...");
        @(negedge clk);
        data_in = {16'sd1024, 16'sd512, 16'sd256, 16'sd128};
        compress_valid = 1'b1;
        @(negedge clk);
        compress_valid = 1'b0;

        begin : wait2
            integer t;
            t = 0;
            while (!compress_done && t < 20) begin
                @(posedge clk); #1;
                t = t + 1;
            end
        end

        if (compress_done && scale_out > 0) begin
            $display("[PASS] Large value compression, scale=%0d", scale_out);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Large value compression failed, done=%b scale=%0d", compress_done, scale_out);
            fail_count = fail_count + 1;
        end

        repeat(2) @(posedge clk);

        // Test 3: Decompress
        $display("[3] Decompressing...");
        @(negedge clk);
        compressed_in = compressed_out;
        scale_in = scale_out;
        decompress_valid = 1'b1;
        @(negedge clk);
        decompress_valid = 1'b0;

        begin : wait3
            integer t;
            t = 0;
            while (!decompress_done && t < 20) begin
                @(posedge clk); #1;
                t = t + 1;
            end
        end

        if (decompress_done) begin
            $display("[PASS] Decompression completed");
            for (i = 0; i < VL; i = i + 1)
                $display("    [%0d] original=%0d decompressed=%0d", i,
                    $signed(data_in[i*DW +: DW]),
                    $signed(decompressed_out[i*DW +: DW]));
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Decompression did not complete");
            fail_count = fail_count + 1;
        end

        // Test 4: Counter verification
        if (total_compressions == 32'd2) begin
            $display("[PASS] Compression counter: %0d (expected 2)", total_compressions);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Compression counter: %0d (expected 2)", total_compressions);
            fail_count = fail_count + 1;
        end

        if (total_bytes_saved == 32'd8) begin
            $display("[PASS] Bytes saved: %0d (expected 8)", total_bytes_saved);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Bytes saved: %0d (expected 8)", total_bytes_saved);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
