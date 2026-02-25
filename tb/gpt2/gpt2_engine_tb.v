// ============================================================================
// Testbench: gpt2_engine_tb
// Full GPT-2 inference: loads embeddings + weights, runs inference
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

    reg                             clk, rst;
    // Embedding loading
    reg                             load_token_emb, load_pos_emb;
    reg  [3:0]                      load_token_idx;
    reg  [1:0]                      load_dim_idx;
    reg  signed [DW-1:0]            load_emb_data;
    reg  [2:0]                      load_pos_idx;
    // LayerNorm params
    reg  [ED*DW-1:0]                ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    reg  [ED*DW-1:0]                ln_final_gamma, ln_final_beta;
    // Attention weights
    reg  [ED*ED*DW-1:0]             wq_flat, wk_flat, wv_flat, wo_flat;
    // FFN weights
    reg  [ED*FD*DW-1:0]             ffn_w1_flat;
    reg  [FD*DW-1:0]                ffn_b1_flat;
    reg  [FD*ED*DW-1:0]             ffn_w2_flat;
    reg  [ED*DW-1:0]                ffn_b2_flat;
    // Inference
    reg                             valid_in;
    reg  [3:0]                      token_in;
    reg  [2:0]                      position_in;
    wire [3:0]                      token_out;
    wire [ED*DW-1:0]                logits_out;
    wire                            valid_out;

    gpt2_engine #(
        .VOCAB_SIZE(VS), .MAX_SEQ_LEN(MS), .EMBED_DIM(ED),
        .NUM_HEADS(NH), .HEAD_DIM(HD), .FFN_DIM(FD),
        .NUM_LAYERS(NL), .DATA_WIDTH(DW)
    ) uut (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
        .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
        .wq_flat(wq_flat), .wk_flat(wk_flat), .wv_flat(wv_flat), .wo_flat(wo_flat),
        .ffn_w1_flat(ffn_w1_flat), .ffn_b1_flat(ffn_b1_flat),
        .ffn_w2_flat(ffn_w2_flat), .ffn_b2_flat(ffn_b2_flat),
        .ln_final_gamma(ln_final_gamma), .ln_final_beta(ln_final_beta),
        .valid_in(valid_in), .token_in(token_in), .position_in(position_in),
        .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer timeout_cnt, ii;
    real val_real;

    // Helper: build identity matrix
    function [ED*ED*DW-1:0] make_identity;
        input dummy;
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

    // Helper: build FFN identity W1 (ED→FD), first ED cols are identity
    function [ED*FD*DW-1:0] make_ffn_w1_identity;
        input dummy;
        integer r, c;
        reg [ED*FD*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    mat[(r*FD+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_ffn_w1_identity = mat;
        end
    endfunction

    // Helper: build FFN identity W2 (FD→ED), first ED rows are identity
    function [FD*ED*DW-1:0] make_ffn_w2_identity;
        input dummy;
        integer r, c;
        reg [FD*ED*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    mat[(r*ED+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_ffn_w2_identity = mat;
        end
    endfunction

    task load_tok;
        input [3:0] tidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_token_emb = 1'b1; load_token_idx = tidx; load_dim_idx = didx; load_emb_data = val;
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
            load_pos_emb = 1'b1; load_pos_idx = pidx; load_dim_idx = didx; load_emb_data = val;
            @(negedge clk);
            load_pos_emb = 1'b0;
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/gpt2_engine.vcd");
        $dumpvars(0, gpt2_engine_tb);
    end

    initial begin
        $display("============================================");
        $display("  GPT-2 Engine Testbench (Full Pipeline)");
        $display("============================================");

        // Reset
        rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;
        load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0; load_pos_idx = 0;
        token_in = 0; position_in = 0;

        // Set all weights to identity-like
        ln1_gamma = {ED{16'sd256}}; ln1_beta = {ED{16'sd0}};
        ln2_gamma = {ED{16'sd256}}; ln2_beta = {ED{16'sd0}};
        ln_final_gamma = {ED{16'sd256}}; ln_final_beta = {ED{16'sd0}};

        wq_flat = make_identity(0);
        wk_flat = make_identity(0);
        wv_flat = make_identity(0);
        wo_flat = make_identity(0);

        ffn_w1_flat = make_ffn_w1_identity(0);
        ffn_b1_flat = {FD{16'sd0}};
        ffn_w2_flat = make_ffn_w2_identity(0);
        ffn_b2_flat = {ED{16'sd0}};

        #35; rst = 0; #25;

        // Load some token embeddings with distinct values
        // Token 3: [4.0, 5.0, 6.0, 7.0] — positive values that survive GELU
        load_tok(4'd3, 2'd0, 16'sd1024);   // 4.0
        load_tok(4'd3, 2'd1, 16'sd1280);   // 5.0
        load_tok(4'd3, 2'd2, 16'sd1536);   // 6.0
        load_tok(4'd3, 2'd3, 16'sd1792);   // 7.0

        // Position 0: [0.1, 0.1, 0.1, 0.1]
        load_pos(3'd0, 2'd0, 16'sd26);
        load_pos(3'd0, 2'd1, 16'sd26);
        load_pos(3'd0, 2'd2, 16'sd26);
        load_pos(3'd0, 2'd3, 16'sd26);

        #20;

        // Run inference: token 3 at position 0
        $display("");
        $display("[INFO] Running GPT-2 inference: token=3, position=0");
        @(negedge clk);
        token_in = 4'd3; position_in = 3'd0; valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;

        // Wait for output (may take many cycles through 2 transformer layers)
        timeout_cnt = 0;
        while (!valid_out && timeout_cnt < 500) begin
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
