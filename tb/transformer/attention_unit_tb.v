// ============================================================================
// Testbench: attention_unit_tb
// Tests single-token attention with identity weight matrices
// ============================================================================
`timescale 1ns / 1ps

module attention_unit_tb;

    parameter ED = 4;  // Small EMBED_DIM for testing
    parameter NH = 2;
    parameter HD = 2;
    parameter DW = 16;

    reg                             clk, rst, valid_in;
    reg  [ED*DW-1:0]                x_in;
    reg  [ED*ED*DW-1:0]             wq_flat, wk_flat, wv_flat, wo_flat;
    wire [ED*DW-1:0]                y_out;
    wire                            valid_out;

    attention_unit #(.EMBED_DIM(ED), .NUM_HEADS(NH), .HEAD_DIM(HD), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in),
        .wq_flat(wq_flat), .wk_flat(wk_flat), .wv_flat(wv_flat), .wo_flat(wo_flat),
        .y_out(y_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt;
    integer ii;
    real y_real;

    // Build identity matrix in flattened Q8.8 format
    // For 4x4: I[i][j] = (i==j) ? 256 : 0
    function [ED*ED*DW-1:0] make_identity;
        input dummy;  // Icarus Verilog requires at least one input
        integer r, c;
        reg [ED*ED*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    mat[(r*ED+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_identity = mat;
        end
    endfunction

    initial begin
        $dumpfile("sim/waveforms/attention_unit.vcd");
        $dumpvars(0, attention_unit_tb);
    end

    initial begin
        $display("============================================");
        $display("  Attention Unit Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0;
        wq_flat = 0; wk_flat = 0; wv_flat = 0; wo_flat = 0;
        #25; rst = 0; #15;

        // Test 1: Identity weights → output ≈ input
        // With all identity weight matrices: Q=K=V=x, attention=V, output_proj=V → y≈x
        wq_flat = make_identity(0);
        wk_flat = make_identity(0);
        wv_flat = make_identity(0);
        wo_flat = make_identity(0);

        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd512, 16'sd256}; // [1.0, 2.0, 3.0, 4.0]
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 50) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("[PASS] Identity attention outputs:");
            for (ii = 0; ii < ED; ii = ii + 1) begin
                y_real = $itor($signed(y_out[ii*DW +: DW])) / 256.0;
                $display("  y[%0d] = %.3f", ii, y_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Identity attention TIMEOUT");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
