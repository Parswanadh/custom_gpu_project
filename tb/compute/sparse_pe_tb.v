// ============================================================================
// Testbench: sparse_pe_tb
// Tests both sparsity_encoder (compress) and sparse_pe (compute).
// ============================================================================
`timescale 1ns/1ps

module sparse_pe_tb;

    reg         clk, rst, enc_valid, pe_valid;

    // Encoder
    reg  signed [7:0] w0, w1, w2, w3;
    wire        [7:0] enc_w0, enc_w1;
    wire        [3:0] enc_mask;
    wire              enc_valid_out;

    // PE
    reg  [7:0] a0, a1, a2, a3;
    wire signed [15:0] pe_product;
    wire               pe_valid_out;
    wire [1:0]         pe_zero_skips;

    sparsity_encoder #(.DATA_WIDTH(8)) u_enc (
        .clk(clk), .rst(rst), .valid_in(enc_valid),
        .w0(w0), .w1(w1), .w2(w2), .w3(w3),
        .out_weight0(enc_w0), .out_weight1(enc_w1),
        .out_mask(enc_mask), .valid_out(enc_valid_out)
    );

    sparse_pe #(.DATA_WIDTH(8)) u_pe (
        .clk(clk), .rst(rst), .valid_in(pe_valid),
        .weight0(enc_w0), .weight1(enc_w1), .mask(enc_mask),
        .act0(a0), .act1(a1), .act2(a2), .act3(a3),
        .product_out(pe_product), .valid_out(pe_valid_out),
        .zero_skips(pe_zero_skips)
    );

    always #5 clk = ~clk;

    integer pass_count, fail_count;

    initial begin
        clk = 0; rst = 1; enc_valid = 0; pe_valid = 0;
        pass_count = 0; fail_count = 0;
        w0 = 0; w1 = 0; w2 = 0; w3 = 0;
        a0 = 0; a1 = 0; a2 = 0; a3 = 0;

        #20; rst = 0;

        // ==================================================================
        // TEST 1: Encode [3, -7, 1, 5], activations [10, 20, 30, 40]
        // Expected: keep |-7|=7 and |5|=5, prune 3 and 1
        // Mask = 1010 (positions 1 and 3)
        // ==================================================================
        $display("\n=== TEST 1: Encode [3, -7, 1, 5] ===");

        @(posedge clk);
        w0 <= 8'sd3; w1 <= -8'sd7; w2 <= 8'sd1; w3 <= 8'sd5;
        enc_valid <= 1;
        @(posedge clk);
        enc_valid <= 0;
        @(posedge clk);  // Encoder registers on this edge: output valid now
        // Read encoder output here (it was set on previous posedge)

        $display("  Encoder: w0=%0d, w1=%0d, mask=%b",
                 $signed(enc_w0), $signed(enc_w1), enc_mask);

        // Feed to sparse PE
        a0 <= 8'd10; a1 <= 8'd20; a2 <= 8'd30; a3 <= 8'd40;
        pe_valid <= 1;
        @(posedge clk);
        pe_valid <= 0;
        @(posedge clk);  // PE registers on this edge

        $display("  Sparse product: %0d (zero_skips=%0d)", pe_product, pe_zero_skips);

        if (pe_zero_skips == 2) begin
            $display("  PASS: 2 zero-skips (50%% sparsity)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected 2 zero-skips");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 2: Encode [10, 0, 0, 20]
        // Expected: keep positions 0 and 3, mask = 1001
        // Sparse product with acts [10,20,30,40]: 10*10 + 20*40 = 900
        // ==================================================================
        $display("\n=== TEST 2: Encode [10, 0, 0, 20] ===");

        @(posedge clk);
        w0 <= 8'sd10; w1 <= 8'sd0; w2 <= 8'sd0; w3 <= 8'sd20;
        enc_valid <= 1;
        @(posedge clk);
        enc_valid <= 0;
        @(posedge clk);

        $display("  Encoder: w0=%0d, w1=%0d, mask=%b",
                 $signed(enc_w0), $signed(enc_w1), enc_mask);

        if (enc_mask == 4'b1001) begin
            $display("  PASS: Mask = 1001 (positions 0 and 3)");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected mask 1001, got %b", enc_mask);
            fail_count = fail_count + 1;
        end

        // Feed to PE
        a0 <= 8'd10; a1 <= 8'd20; a2 <= 8'd30; a3 <= 8'd40;
        pe_valid <= 1;
        @(posedge clk);
        pe_valid <= 0;
        @(posedge clk);

        $display("  Sparse product: %0d (expected 900)", pe_product);
        if (pe_product == 900) begin
            $display("  PASS: Correct product");
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL: Expected 900");
            fail_count = fail_count + 1;
        end

        // ==================================================================
        // TEST 3: Equal magnitudes [5, -5, 5, -5] → any 2 positions valid
        // ==================================================================
        $display("\n=== TEST 3: Equal magnitudes [5, -5, 5, -5] ===");

        @(posedge clk);
        w0 <= 8'sd5; w1 <= -8'sd5; w2 <= 8'sd5; w3 <= -8'sd5;
        enc_valid <= 1;
        @(posedge clk);
        enc_valid <= 0;
        @(posedge clk);

        $display("  Encoder: w0=%0d, w1=%0d, mask=%b",
                 $signed(enc_w0), $signed(enc_w1), enc_mask);

        begin : check_mask
            integer ones;
            ones = enc_mask[0] + enc_mask[1] + enc_mask[2] + enc_mask[3];
            if (ones == 2) begin
                $display("  PASS: Valid 2:4 mask (exactly 2 bits set)");
                pass_count = pass_count + 1;
            end else begin
                $display("  FAIL: Expected 2 bits set, got %0d", ones);
                fail_count = fail_count + 1;
            end
        end

        // ==================================================================
        // SUMMARY
        // ==================================================================
        $display("\n=== 2:4 SPARSITY TEST SUMMARY ===");
        $display("  Tests: %0d/%0d passed", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("  ALL PASSED");
        else
            $display("  %0d FAILURES", fail_count);

        #50;
        $finish;
    end

endmodule
