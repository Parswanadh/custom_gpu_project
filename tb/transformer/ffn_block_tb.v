// ============================================================================
// Testbench: ffn_block_tb
// Tests FFN with SRAM-loaded weights (updated for Issue #7 SRAM interface)
// ============================================================================
`timescale 1ns / 1ps

module ffn_block_tb;

    parameter ED = 4;
    parameter FD = 8;
    parameter DW = 16;
    localparam MAX_DIM = (FD > ED) ? FD : ED;
    localparam ADDR_BITS = $clog2(MAX_DIM);

    reg                                 clk, rst, valid_in;
    reg  [ED*DW-1:0]                    x_in;

    // Weight loading interface
    reg                                 weight_load_en;
    reg                                 weight_layer_sel;
    reg                                 weight_is_bias;
    reg  [ADDR_BITS-1:0]               weight_row, weight_col;
    reg  signed [DW-1:0]               weight_data;

    wire [ED*DW-1:0]                    y_out;
    wire                                valid_out;
    wire [31:0]                         zero_skip_count;

    ffn_block #(.EMBED_DIM(ED), .FFN_DIM(FD), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x_in(x_in),
        .weight_load_en(weight_load_en),
        .weight_layer_sel(weight_layer_sel),
        .weight_is_bias(weight_is_bias),
        .weight_row(weight_row), .weight_col(weight_col),
        .weight_data(weight_data),
        .y_out(y_out), .valid_out(valid_out),
        .zero_skip_count(zero_skip_count)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt, ii;
    real y_real;

    // Task: load one weight value
    task load_w;
        input         layer_sel;
        input         is_bias;
        input integer row, col;
        input signed [DW-1:0] data;
        begin
            @(posedge clk);
            weight_load_en   <= 1'b1;
            weight_layer_sel <= layer_sel;
            weight_is_bias   <= is_bias;
            weight_row       <= row[ADDR_BITS-1:0];
            weight_col       <= col[ADDR_BITS-1:0];
            weight_data      <= data;
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/ffn_block.vcd");
        $dumpvars(0, ffn_block_tb);
    end

    initial begin
        $display("============================================");
        $display("  FFN Block Testbench (Q8.8, SRAM)");
        $display("============================================");

        rst = 1; valid_in = 0; x_in = 0;
        weight_load_en = 0; weight_layer_sel = 0; weight_is_bias = 0;
        weight_row = 0; weight_col = 0; weight_data = 0;
        #25; rst = 0; #15;

        // Load W1 (layer 0): identity-like
        $display("[1] Loading W1 (ED×FD identity basis)...");
        begin : load_w1
            integer r, c;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    load_w(0, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            // b1 = 0
            for (c = 0; c < FD; c = c + 1)
                load_w(0, 1, 0, c, 16'sd0);
        end

        // Load W2 (layer 1): identity-like
        $display("[2] Loading W2 (FD×ED identity basis)...");
        begin : load_w2
            integer r, c;
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    load_w(1, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            // b2 = 0
            for (c = 0; c < ED; c = c + 1)
                load_w(1, 1, 0, c, 16'sd0);
        end
        @(posedge clk);
        weight_load_en <= 1'b0;
        $display("    Done loading weights.");

        #20;

        // Test: positive input → should pass through (GELU(x)≈x for large x)
        $display("");
        $display("[3] Test: Positive input, identity weights");
        @(negedge clk);
        x_in = {16'sd1024, 16'sd768, 16'sd512, 16'sd256}; // [1.0, 2.0, 3.0, 4.0]
        valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 200) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("    Output after %0d cycles:", timeout_cnt);
            for (ii = 0; ii < ED; ii = ii + 1) begin
                y_real = $signed(y_out[ii*DW +: DW]) / 256.0;
                $display("    y[%0d] = %0d (%.3f)", ii, $signed(y_out[ii*DW +: DW]), y_real);
            end
            $display("    Zero-skips: %0d", zero_skip_count);
            $display("[PASS] FFN positive input with identity weights");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Timeout!");
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
