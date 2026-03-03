// ============================================================================
// Testbench: gpt2_engine_tb
// Full GPT-2 inference with SRAM-loaded per-layer weights (Issues #7, #8)
// ============================================================================
`timescale 1ns / 1ps

module gpt2_engine_tb;

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
    // Embedding loading
    reg                             load_token_emb, load_pos_emb;
    reg  [3:0]                      load_token_idx;
    reg  [1:0]                      load_dim_idx;
    reg  signed [DW-1:0]            load_emb_data;
    reg  [2:0]                      load_pos_idx;
    // Per-layer LN loading
    reg                             load_ln_en;
    reg  [$clog2(NL):0]             load_layer_idx;
    reg                             load_ln_sel, load_ln_is_gamma;
    reg  [$clog2(ED)-1:0]           load_ln_dim;
    reg  signed [DW-1:0]            load_ln_data;
    // Attention weight loading
    reg                             load_attn_weight_en;
    reg  [1:0]                      load_attn_matrix_sel;
    reg  [$clog2(ED)-1:0]           load_attn_row, load_attn_col;
    reg  signed [DW-1:0]            load_attn_data;
    // FFN weight loading
    reg                             load_ffn_weight_en;
    reg                             load_ffn_layer_sel, load_ffn_is_bias;
    reg  [FFN_ADDR-1:0]             load_ffn_row, load_ffn_col;
    reg  signed [DW-1:0]            load_ffn_data;
    // Inference
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
    real val_real;

    // Task: load a token embedding dimension
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

    // Task: load a position embedding dimension
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

    // Task: load LN gamma/beta for one layer+dim
    task load_ln;
        input [$clog2(NL):0] layer;
        input         ln_sel;
        input         is_gamma;
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

    // Task: load attention weight
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

    // Task: load FFN weight
    task load_ffn_w;
        input         layer_sel;
        input         is_bias;
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

    initial begin
        $dumpfile("sim/waveforms/gpt2_engine.vcd");
        $dumpvars(0, gpt2_engine_tb);
    end

    initial begin
        $display("============================================");
        $display("  GPT-2 Engine Testbench (SRAM Weights)");
        $display("============================================");

        // Reset
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

        // Load LN params: gamma=1.0, beta=0.0 for all layers
        $display("[1] Loading LayerNorm params (identity)...");
        begin : load_ln_params
            integer l, d;
            for (l = 0; l <= NL; l = l + 1)  // NL layers + final LN
                for (d = 0; d < ED; d = d + 1) begin
                    load_ln(l, 0, 1, d, 16'sd256);   // LN1 gamma = 1.0
                    load_ln(l, 0, 0, d, 16'sd0);     // LN1 beta = 0.0
                    load_ln(l, 1, 1, d, 16'sd256);   // LN2 gamma = 1.0
                    load_ln(l, 1, 0, d, 16'sd0);     // LN2 beta = 0.0
                end
        end
        @(posedge clk); load_ln_en <= 1'b0;

        // Load attention identity weights for all 4 matrices
        $display("[2] Loading attention identity weights...");
        begin : load_attn_weights
            integer m, r, c;
            for (m = 0; m < 4; m = m + 1)
                for (r = 0; r < ED; r = r + 1)
                    for (c = 0; c < ED; c = c + 1)
                        load_attn_w(m[1:0], r, c, (r == c) ? 16'sd256 : 16'sd0);
        end
        @(posedge clk); load_attn_weight_en <= 1'b0;

        // Load FFN identity-like weights
        $display("[3] Loading FFN identity weights...");
        begin : load_ffn_weights
            integer r, c;
            // W1: ED×FD identity basis
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    load_ffn_w(0, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            // b1 = 0
            for (c = 0; c < FD; c = c + 1)
                load_ffn_w(0, 1, 0, c, 16'sd0);
            // W2: FD×ED identity basis
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    load_ffn_w(1, 0, r, c, (r == c) ? 16'sd256 : 16'sd0);
            // b2 = 0
            for (c = 0; c < ED; c = c + 1)
                load_ffn_w(1, 1, 0, c, 16'sd0);
        end
        @(posedge clk); load_ffn_weight_en <= 1'b0;

        // Load token embeddings
        $display("[4] Loading token embeddings...");
        load_tok(4'd3, 2'd0, 16'sd1024);  // 4.0
        load_tok(4'd3, 2'd1, 16'sd1280);  // 5.0
        load_tok(4'd3, 2'd2, 16'sd1536);  // 6.0
        load_tok(4'd3, 2'd3, 16'sd1792);  // 7.0

        // Load position embeddings
        load_pos(3'd0, 2'd0, 16'sd26);
        load_pos(3'd0, 2'd1, 16'sd26);
        load_pos(3'd0, 2'd2, 16'sd26);
        load_pos(3'd0, 2'd3, 16'sd26);

        #20;

        // Run inference
        $display("");
        $display("[5] Running GPT-2 inference: token=3, position=0");
        @(negedge clk);
        token_in = 4'd3; position_in = 3'd0; valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 1000) begin
            @(negedge clk);
            timeout_cnt = timeout_cnt + 1;
        end

        if (valid_out) begin
            $display("[PASS] GPT-2 inference completed in %0d cycles!", timeout_cnt);
            $display("  Predicted next token: %0d", token_out);
            $display("  Logits:");
            for (ii = 0; ii < ED; ii = ii + 1) begin
                val_real = $itor($signed(logits_out[ii*DW +: DW])) / 256.0;
                $display("    logit[%0d] = %.3f", ii, val_real);
            end
            $display("  Total zero-skips: %0d", total_zero_skips);
            $display("  Total cycles: %0d", total_cycles);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] GPT-2 inference TIMEOUT after %0d cycles", timeout_cnt);
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
