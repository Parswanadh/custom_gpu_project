// ============================================================================
// Testbench: attention_unit_tb
// Tests attention with SRAM-loaded weights (updated for Issue #7 SRAM interface)
// ============================================================================
`timescale 1ns / 1ps

module attention_unit_tb;

    parameter ED = 4;   // EMBED_DIM
    parameter NH = 2;   // NUM_HEADS
    parameter HD = 2;   // HEAD_DIM
    parameter DW = 16;  // DATA_WIDTH
    parameter MSL = 8;  // MAX_SEQ_LEN

    reg                             clk, rst, valid_in;
    reg  [ED*DW-1:0]                x_in;
    reg  [$clog2(MSL)-1:0]          seq_pos;

    // Weight loading interface
    reg                             weight_load_en;
    reg  [1:0]                      weight_matrix_sel;
    reg  [$clog2(ED)-1:0]           weight_row, weight_col;
    reg  signed [DW-1:0]            weight_data;
    reg                             causal_mask_en;

    wire [ED*DW-1:0]                y_out;
    wire                            valid_out;
    wire [31:0]                     zero_skip_count;

    attention_unit #(
        .EMBED_DIM(ED), .NUM_HEADS(NH), .HEAD_DIM(HD),
        .MAX_SEQ_LEN(MSL), .DATA_WIDTH(DW)
    ) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in), .seq_pos(seq_pos),
        .weight_load_en(weight_load_en),
        .weight_matrix_sel(weight_matrix_sel),
        .weight_row(weight_row), .weight_col(weight_col),
        .weight_data(weight_data),
        .causal_mask_en(causal_mask_en),
        .y_out(y_out), .valid_out(valid_out),
        .zero_skip_count(zero_skip_count)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt;
    integer ii;
    real y_real;

    // Task: Load identity matrix into a weight matrix
    task load_identity;
        input [1:0] matrix_sel;
        integer r, c;
        begin
            for (r = 0; r < ED; r = r + 1) begin
                for (c = 0; c < ED; c = c + 1) begin
                    @(posedge clk);
                    weight_load_en   <= 1'b1;
                    weight_matrix_sel <= matrix_sel;
                    weight_row       <= r[$clog2(ED)-1:0];
                    weight_col       <= c[$clog2(ED)-1:0];
                    weight_data      <= (r == c) ? 16'sd256 : 16'sd0; // Q8.8 identity
                end
            end
            @(posedge clk);
            weight_load_en <= 1'b0;
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/attention_unit.vcd");
        $dumpvars(0, attention_unit_tb);
    end

    initial begin
        $display("============================================");
        $display("  Attention Unit Testbench (Q8.8, SRAM)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0; seq_pos = 0;
        weight_load_en = 0; causal_mask_en = 1;
        weight_matrix_sel = 0; weight_row = 0; weight_col = 0; weight_data = 0;
        #25; rst = 0; #15;

        // Load identity weights into all 4 matrices (Wq, Wk, Wv, Wo)
        $display("[1] Loading identity weights into Wq/Wk/Wv/Wo...");
        load_identity(2'd0);  // Wq
        load_identity(2'd1);  // Wk
        load_identity(2'd2);  // Wv
        load_identity(2'd3);  // Wo
        $display("    Done loading weights.");

        #20;

        // Test 1: Single token with identity weights
        $display("");
        $display("[2] Test: Single token, identity weights");
        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd512, 16'sd256}; // [1.0, 2.0, 3.0, 4.0] Q8.8
        seq_pos = 0;
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 100) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("    Output received after %0d cycles", timeout_cnt);
            for (ii = 0; ii < ED; ii = ii + 1) begin
                y_real = $signed(y_out[ii*DW +: DW]) / 256.0;
                $display("    y[%0d] = %0d (%.3f)", ii, $signed(y_out[ii*DW +: DW]), y_real);
            end
            $display("    Zero-skips: %0d", zero_skip_count);
            $display("[PASS] Single token with identity weights");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Timeout waiting for attention output!");
            fail_count = fail_count + 1;
        end

        #50;

        // Test 2: Second token (tests KV cache)
        $display("");
        $display("[3] Test: Second token (KV cache)");
        @(negedge clk);
        x_in = {16'sd512, 16'sd512, 16'sd512, 16'sd512}; // [2.0, 2.0, 2.0, 2.0]
        seq_pos = 1;
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 100) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("    Output received after %0d cycles", timeout_cnt);
            for (ii = 0; ii < ED; ii = ii + 1) begin
                y_real = $signed(y_out[ii*DW +: DW]) / 256.0;
                $display("    y[%0d] = %0d (%.3f)", ii, $signed(y_out[ii*DW +: DW]), y_real);
            end
            $display("[PASS] Second token with KV cache");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Timeout on second token!");
            fail_count = fail_count + 1;
        end

        #20;
        $display("");
        $display("============================================");
        $display("  Results: %0d passed, %0d failed", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
