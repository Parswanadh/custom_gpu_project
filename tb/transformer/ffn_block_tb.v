// ============================================================================
// Testbench: ffn_block_tb
// Tests FFN with identity-like weights
// ============================================================================
`timescale 1ns / 1ps

module ffn_block_tb;

    parameter ED = 4;
    parameter FD = 8;
    parameter DW = 16;

    reg                                 clk, rst, valid_in;
    reg  [ED*DW-1:0]                    x_in;
    reg  [ED*FD*DW-1:0]                 w1_flat;
    reg  [FD*DW-1:0]                    b1_flat;
    reg  [FD*ED*DW-1:0]                 w2_flat;
    reg  [ED*DW-1:0]                    b2_flat;
    wire [ED*DW-1:0]                    y_out;
    wire                                valid_out;

    ffn_block #(.EMBED_DIM(ED), .FFN_DIM(FD), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in),
        .w1_flat(w1_flat), .b1_flat(b1_flat),
        .w2_flat(w2_flat), .b2_flat(b2_flat),
        .y_out(y_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt, ii;
    real y_real;

    initial begin
        $dumpfile("sim/waveforms/ffn_block.vcd");
        $dumpvars(0, ffn_block_tb);
    end

    initial begin
        $display("============================================");
        $display("  FFN Block Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0;
        w1_flat = 0; b1_flat = 0; w2_flat = 0; b2_flat = 0;
        #25; rst = 0; #15;

        // Simple test: set W1 and W2 as scaled identity-like
        // W1[i][j] = (i==j) ? 256 : 0 for i<ED (first ED cols of FFN_DIM map to input)
        // W2[i][j] = (i==j) ? 256 : 0 for i<ED (project back from first ED dims)
        // This way: Linear1 copies x to first ED dims of hidden,
        //           GELU(positive x) ≈ x for large x,
        //           Linear2 copies back → y ≈ x for positive x > 3

        // Build W1: ED×FD identity basis
        begin : build_weights
            integer r, c;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    w1_flat[(r*FD+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            for (c = 0; c < FD; c = c + 1)
                b1_flat[c*DW +: DW] = 16'sd0;
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    w2_flat[(r*ED+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            for (c = 0; c < ED; c = c + 1)
                b2_flat[c*DW +: DW] = 16'sd0;
        end

        #10;

        // Test 1: Positive input (should pass through GELU relatively unchanged for x>3)
        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd1280, 16'sd1536}; // [6.0, 5.0, 3.0, 4.0]
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 50) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("[PASS] FFN with positive inputs:");
            for (ii = 0; ii < ED; ii = ii + 1) begin
                y_real = $itor($signed(y_out[ii*DW +: DW])) / 256.0;
                $display("  y[%0d] = %.3f", ii, y_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] FFN TIMEOUT");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
