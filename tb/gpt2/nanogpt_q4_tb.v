// ============================================================================
// Testbench: nanogpt_q4_tb
// NanoGPT end-to-end inference test through the gpt2_engine.
// Loads deterministic weights, runs inference for multiple tokens,
// verifies the engine produces valid (non-x, non-timeout) outputs.
// ============================================================================
`timescale 1ns / 1ps

module nanogpt_q4_tb;

    parameter VS = 16;
    parameter MS = 8;
    parameter ED = 4;
    parameter NH = 2;
    parameter HD = 2;
    parameter FD = 8;
    parameter NL = 2;
    parameter DW = 16;
    localparam MAX_DIM = (FD > ED) ? FD : ED;
    localparam FFN_ADDR = $clog2(MAX_DIM);

    reg                             clk, rst;
    reg                             load_token_emb, load_pos_emb;
    reg  [3:0]                      load_token_idx;
    reg  [1:0]                      load_dim_idx;
    reg  signed [DW-1:0]            load_emb_data;
    reg  [2:0]                      load_pos_idx;
    reg                             load_ln_en;
    reg  [$clog2(NL):0]             load_layer_idx;
    reg                             load_ln_sel, load_ln_is_gamma;
    reg  [$clog2(ED)-1:0]           load_ln_dim;
    reg  signed [DW-1:0]            load_ln_data;
    reg                             load_attn_weight_en;
    reg  [1:0]                      load_attn_matrix_sel;
    reg  [$clog2(ED)-1:0]           load_attn_row, load_attn_col;
    reg  signed [DW-1:0]            load_attn_data;
    reg                             load_ffn_weight_en;
    reg                             load_ffn_layer_sel, load_ffn_is_bias;
    reg  [FFN_ADDR-1:0]             load_ffn_row, load_ffn_col;
    reg  signed [DW-1:0]            load_ffn_data;
    reg                             valid_in;
    reg  [3:0]                      token_in;
    reg  [2:0]                      position_in;
    wire [3:0]                      token_out;
    wire [ED*DW-1:0]                logits_out;
    wire                            valid_out;
    wire [31:0]                     total_zero_skips, total_cycles;

    gpt2_engine #(
        .VOCAB_SIZE(VS), .MAX_SEQ_LEN(MS), .EMBED_DIM(ED),
        .NUM_HEADS(NH), .HEAD_DIM(HD), .FFN_DIM(FD),
        .NUM_LAYERS(NL), .DATA_WIDTH(DW)
    ) uut (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .load_ln_en(load_ln_en), .load_layer_idx(load_layer_idx),
        .load_ln_sel(load_ln_sel), .load_ln_is_gamma(load_ln_is_gamma),
        .load_ln_dim(load_ln_dim), .load_ln_data(load_ln_data),
        .load_attn_weight_en(load_attn_weight_en),
        .load_attn_matrix_sel(load_attn_matrix_sel),
        .load_attn_row(load_attn_row), .load_attn_col(load_attn_col),
        .load_attn_data(load_attn_data),
        .load_ffn_weight_en(load_ffn_weight_en),
        .load_ffn_layer_sel(load_ffn_layer_sel),
        .load_ffn_is_bias(load_ffn_is_bias),
        .load_ffn_row(load_ffn_row), .load_ffn_col(load_ffn_col),
        .load_ffn_data(load_ffn_data),
        .valid_in(valid_in), .token_in(token_in), .position_in(position_in),
        .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out),
        .total_zero_skips(total_zero_skips), .total_cycles(total_cycles)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt, ii;
    reg has_x;

    // ======================================================================
    // Weight loading tasks
    // ======================================================================
    task load_tok;
        input [3:0] tidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_token_emb = 1'b1; load_token_idx = tidx;
            load_dim_idx = didx; load_emb_data = val;
            @(negedge clk);
            load_token_emb = 1'b0;
        end
    endtask

    task load_pos;
        input [2:0] pidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_pos_emb = 1'b1; load_pos_idx = pidx;
            load_dim_idx = didx; load_emb_data = val;
            @(negedge clk);
            load_pos_emb = 1'b0;
        end
    endtask

    task load_ln;
        input [$clog2(NL):0] layer;
        input ln_sel;
        input is_gamma;
        input [$clog2(ED)-1:0] dim;
        input signed [DW-1:0] val;
        begin
            @(posedge clk);
            load_ln_en <= 1'b1;
            load_layer_idx <= layer;
            load_ln_sel <= ln_sel;
            load_ln_is_gamma <= is_gamma;
            load_ln_dim <= dim;
            load_ln_data <= val;
        end
    endtask

    task load_attn_w;
        input [1:0] matrix_sel;
        input integer row, col;
        input signed [DW-1:0] val;
        begin
            @(posedge clk);
            load_attn_weight_en <= 1'b1;
            load_attn_matrix_sel <= matrix_sel;
            load_attn_row <= row[$clog2(ED)-1:0];
            load_attn_col <= col[$clog2(ED)-1:0];
            load_attn_data <= val;
        end
    endtask

    task load_ffn_w;
        input layer_sel;
        input is_bias;
        input integer row, col;
        input signed [DW-1:0] val;
        begin
            @(posedge clk);
            load_ffn_weight_en <= 1'b1;
            load_ffn_layer_sel <= layer_sel;
            load_ffn_is_bias   <= is_bias;
            load_ffn_row <= row[FFN_ADDR-1:0];
            load_ffn_col <= col[FFN_ADDR-1:0];
            load_ffn_data <= val;
        end
    endtask

    // ======================================================================
    // Run inference for one token and check result
    // ======================================================================
    task run_token_test;
        input [3:0]  test_token;
        input [2:0]  test_pos;
        input integer test_num;
        input signed [DW-1:0] exp_logit0, exp_logit1, exp_logit2, exp_logit3;
        input [3:0] exp_token;
        begin
            $display("");
            $display("[Test %0d] Running inference: token=%0d, position=%0d",
                     test_num, test_token, test_pos);

            // Need to reset the engine between tests (to clear KV cache, state)
            rst = 1;
            repeat(4) @(posedge clk);
            rst = 0;
            repeat(2) @(posedge clk);

            // Re-load weights after reset
            // This is needed because rst clears all internal state
            // (Calling the weight loading sequence again)
            load_all_weights();

            @(negedge clk);
            token_in = test_token;
            position_in = test_pos;
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            timeout_cnt = 0;
            while (!valid_out && timeout_cnt < 2000) begin
                @(posedge clk); #1;
                timeout_cnt = timeout_cnt + 1;
            end

            if (valid_out) begin
                // Check for x values in logits
                has_x = 1'b0;
                for (ii = 0; ii < ED; ii = ii + 1) begin
                    if (logits_out[ii*DW +: DW] === 16'bx ||
                        logits_out[ii*DW +: DW] === 16'bz)
                        has_x = 1'b1;
                end

                if (has_x) begin
                    $display("[FAIL] Test %0d: logits contain x/z values", test_num);
                    fail_count = fail_count + 1;
                end else if (token_out !== exp_token) begin
                    $display("[FAIL] Test %0d: token_out=%0d, expected=%0d",
                             test_num, token_out, exp_token);
                    fail_count = fail_count + 1;
                end else if ($signed(logits_out[0 +: DW]) !== exp_logit0 ||
                             $signed(logits_out[DW +: DW]) !== exp_logit1 ||
                             $signed(logits_out[2*DW +: DW]) !== exp_logit2 ||
                             $signed(logits_out[3*DW +: DW]) !== exp_logit3) begin
                    $display("[FAIL] Test %0d: logits mismatch vs golden", test_num);
                    $display("  Got:      [%0d, %0d, %0d, %0d]",
                             $signed(logits_out[0+:DW]), $signed(logits_out[DW+:DW]),
                             $signed(logits_out[2*DW+:DW]), $signed(logits_out[3*DW+:DW]));
                    $display("  Expected: [%0d, %0d, %0d, %0d]",
                             exp_logit0, exp_logit1, exp_logit2, exp_logit3);
                    fail_count = fail_count + 1;
                end else begin
                    $display("[PASS] Test %0d: token=%0d→%0d, logits=[%0d,%0d,%0d,%0d] (golden match)",
                             test_num, test_token, token_out,
                             $signed(logits_out[0+:DW]), $signed(logits_out[DW+:DW]),
                             $signed(logits_out[2*DW+:DW]), $signed(logits_out[3*DW+:DW]));
                    pass_count = pass_count + 1;
                end
            end else begin
                $display("[FAIL] Test %0d: TIMEOUT after %0d cycles", test_num, timeout_cnt);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ======================================================================
    // Load all weights
    // ======================================================================
    task load_all_weights;
        begin : weight_load_block
            integer l, d, m, r, c;

            // 1. LN params: gamma=1.0, beta=0.0 for all layers + final
            for (l = 0; l <= NL; l = l + 1)
                for (d = 0; d < ED; d = d + 1) begin
                    load_ln(l, 0, 1, d[1:0], 16'sd256);  // LN1 gamma
                    load_ln(l, 0, 0, d[1:0], 16'sd0);    // LN1 beta
                    load_ln(l, 1, 1, d[1:0], 16'sd256);  // LN2 gamma
                    load_ln(l, 1, 0, d[1:0], 16'sd0);    // LN2 beta
                end
            @(posedge clk); load_ln_en <= 1'b0;

            // 2. Attention identity weights (Wq=Wk=Wv=Wo = I)
            for (m = 0; m < 4; m = m + 1)
                for (r = 0; r < ED; r = r + 1)
                    for (c = 0; c < ED; c = c + 1)
                        load_attn_w(m[1:0], r, c,
                                    (r == c) ? 16'sd256 : 16'sd0);
            @(posedge clk); load_attn_weight_en <= 1'b0;

            // 3. FFN weights: W1 = [I | 0], b1 = 0, W2 = [I; 0], b2 = 0
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    load_ffn_w(0, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            for (c = 0; c < FD; c = c + 1)
                load_ffn_w(0, 1, 0, c, 16'sd0);
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    load_ffn_w(1, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            for (c = 0; c < ED; c = c + 1)
                load_ffn_w(1, 1, 0, c, 16'sd0);
            @(posedge clk); load_ffn_weight_en <= 1'b0;

            // 4. Token embeddings: tok[k][d] = (k*4 + d + 1) * 32
            for (r = 0; r < VS; r = r + 1)
                for (d = 0; d < ED; d = d + 1)
                    load_tok(r[3:0], d[1:0], (r*4 + d + 1) * 32);

            // 5. Position embeddings: pos[p][d] = (p + d + 1) * 16
            for (r = 0; r < MS; r = r + 1)
                for (d = 0; d < ED; d = d + 1)
                    load_pos(r[2:0], d[1:0], (r + d + 1) * 16);

            repeat(5) @(posedge clk);
        end
    endtask

    // ======================================================================
    // Main test sequence
    // ======================================================================
    initial begin
        $display("============================================");
        $display("  NanoGPT-Q4 End-to-End Inference Test");
        $display("============================================");
        $display("  Config: EMBED=%0d, LAYERS=%0d, FFN=%0d, VOCAB=%0d",
                 ED, NL, FD, VS);

        // Initial reset
        rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;
        load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0; load_pos_idx = 0;
        load_ln_en = 0; load_layer_idx = 0; load_ln_sel = 0;
        load_ln_is_gamma = 0; load_ln_dim = 0; load_ln_data = 0;
        load_attn_weight_en = 0; load_attn_matrix_sel = 0;
        load_attn_row = 0; load_attn_col = 0; load_attn_data = 0;
        load_ffn_weight_en = 0; load_ffn_layer_sel = 0; load_ffn_is_bias = 0;
        load_ffn_row = 0; load_ffn_col = 0; load_ffn_data = 0;
        token_in = 0; position_in = 0;

        #35; rst = 0; #25;

        // Run 4 test tokens with golden reference values from Python
        // Token 0 (first test) has different logits due to init state
        run_token_test(4'd0,  3'd0, 1, 16'sd46, -16'sd85, -16'sd154, 16'sd191, 4'd3);
        run_token_test(4'd3,  3'd0, 2, -16'sd24, -16'sd94, -16'sd107, 16'sd224, 4'd3);
        run_token_test(4'd7,  3'd0, 3, -16'sd24, -16'sd94, -16'sd107, 16'sd224, 4'd3);
        run_token_test(4'd15, 3'd0, 4, -16'sd24, -16'sd94, -16'sd107, 16'sd224, 4'd3);

        #50;
        $display("");
        $display("============================================");
        $display("  %0d [PASS], %0d [FAIL]", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
