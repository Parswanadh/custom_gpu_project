// ============================================================================
// Testbench: accelerated_gpt2_engine_tb
// Tests the accelerated GPT-2 pipeline with per-layer LN loading (Issue #8)
// ============================================================================
`timescale 1ns/1ps

module accelerated_gpt2_engine_tb;

    parameter VOCAB_SIZE  = 8;
    parameter MAX_SEQ_LEN = 8;
    parameter EMBED_DIM   = 4;
    parameter NUM_HEADS   = 2;
    parameter HEAD_DIM    = 2;
    parameter FFN_DIM     = 8;
    parameter NUM_LAYERS  = 1;
    parameter DATA_WIDTH  = 16;

    reg         clk, rst;

    // Embedding loading
    reg         load_token_emb;
    reg [$clog2(VOCAB_SIZE)-1:0] load_token_idx;
    reg [$clog2(EMBED_DIM)-1:0]  load_dim_idx;
    reg signed [DATA_WIDTH-1:0]  load_emb_data;
    reg         load_pos_emb;
    reg [$clog2(MAX_SEQ_LEN)-1:0] load_pos_idx;

    // Per-layer LN loading (Issue #8)
    reg                               load_ln_en;
    reg [$clog2(NUM_LAYERS):0]        load_layer_idx;
    reg                               load_ln_sel;
    reg                               load_ln_is_gamma;
    reg [$clog2(EMBED_DIM)-1:0]       load_ln_dim;
    reg signed [DATA_WIDTH-1:0]       load_ln_data;

    // Weights (still flat for attention/FFN — matching the module interface)
    reg [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq, wk, wv, wo;
    reg [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0] ffn_w1;
    reg [FFN_DIM*DATA_WIDTH-1:0] ffn_b1;
    reg [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0] ffn_w2;
    reg [EMBED_DIM*DATA_WIDTH-1:0] ffn_b2;

    // Inference
    reg         valid_in;
    reg [$clog2(VOCAB_SIZE)-1:0] token_in;
    reg [$clog2(MAX_SEQ_LEN)-1:0] position_in;
    wire [$clog2(VOCAB_SIZE)-1:0] token_out;
    wire [EMBED_DIM*DATA_WIDTH-1:0] logits_out;
    wire        valid_out;
    wire [31:0] total_zero_skips, total_cycles;

    accelerated_gpt2_engine #(
        .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
        .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),
        .NUM_LAYERS(NUM_LAYERS), .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .load_ln_en(load_ln_en), .load_layer_idx(load_layer_idx),
        .load_ln_sel(load_ln_sel), .load_ln_is_gamma(load_ln_is_gamma),
        .load_ln_dim(load_ln_dim), .load_ln_data(load_ln_data),
        .wq_flat(wq), .wk_flat(wk), .wv_flat(wv), .wo_flat(wo),
        .ffn_w1_flat(ffn_w1), .ffn_b1_flat(ffn_b1),
        .ffn_w2_flat(ffn_w2), .ffn_b2_flat(ffn_b2),
        .valid_in(valid_in), .token_in(token_in), .position_in(position_in),
        .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out),
        .total_zero_skips(total_zero_skips), .total_cycles(total_cycles)
    );

    always #5 clk = ~clk;

    integer i, j, cycle, tok;
    reg [31:0] start_cycle;
    reg [$clog2(VOCAB_SIZE)-1:0] tokens_generated [0:2];

    // Task: load LN param
    task load_ln_param;
        input [$clog2(NUM_LAYERS):0] layer;
        input ln_s;
        input is_g;
        input [$clog2(EMBED_DIM)-1:0] dim;
        input signed [DATA_WIDTH-1:0] val;
        begin
            @(posedge clk);
            load_ln_en <= 1'b1;
            load_layer_idx <= layer;
            load_ln_sel <= ln_s;
            load_ln_is_gamma <= is_g;
            load_ln_dim <= dim;
            load_ln_data <= val;
        end
    endtask

    initial begin
        $display("");
        $display("================================================================");
        $display("  Accelerated GPT-2 Engine -- Full Pipeline Test");
        $display("  VOCAB=%0d, EMBED=%0d, HEADS=%0d, FFN=%0d, LAYERS=%0d",
            VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS);
        $display("================================================================");

        clk = 0; rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;
        load_ln_en = 0; load_layer_idx = 0; load_ln_sel = 0;
        load_ln_is_gamma = 0; load_ln_dim = 0; load_ln_data = 0;

        // --- Initialize attention/FFN weights ---
        // Attention weights: identity × 0.5 to avoid overflow
        wq = 0; wk = 0; wv = 0; wo = 0;
        for (i = 0; i < EMBED_DIM; i = i + 1)
            for (j = 0; j < EMBED_DIM; j = j + 1)
                if (i == j) begin
                    wq[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128; // 0.5
                    wk[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
                    wv[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = 16'sd256; // 1.0
                    wo[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
                end

        // FFN: identity-like
        ffn_w1 = 0; ffn_b1 = 0; ffn_w2 = 0; ffn_b2 = 0;
        for (i = 0; i < EMBED_DIM; i = i + 1)
            for (j = 0; j < FFN_DIM; j = j + 1)
                ffn_w1[(i*FFN_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = (i == j) ? 16'sd256 : 16'sd0;
        for (i = 0; i < FFN_DIM; i = i + 1)
            for (j = 0; j < EMBED_DIM; j = j + 1)
                ffn_w2[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] = (i == j) ? 16'sd256 : 16'sd0;

        #35; rst = 0; #25;

        // --- Load per-layer LN params via SRAM interface (Issue #8) ---
        $display("[1] Loading per-layer LN params via SRAM interface...");
        begin : load_ln_all
            integer l, d;
            for (l = 0; l <= NUM_LAYERS; l = l + 1)  // layers + final LN
                for (d = 0; d < EMBED_DIM; d = d + 1) begin
                    load_ln_param(l, 0, 1, d, 16'sd256);  // LN1 gamma = 1.0
                    load_ln_param(l, 0, 0, d, 16'sd0);    // LN1 beta = 0.0
                    load_ln_param(l, 1, 1, d, 16'sd256);  // LN2 gamma = 1.0
                    load_ln_param(l, 1, 0, d, 16'sd0);    // LN2 beta = 0.0
                end
        end
        @(posedge clk); load_ln_en <= 1'b0;

        // --- Load token embeddings ---
        $display("[2] Loading %0d token embeddings...", VOCAB_SIZE);
        for (i = 0; i < VOCAB_SIZE; i = i + 1) begin
            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                @(posedge clk);
                load_token_emb <= 1'b1;
                load_token_idx <= i;
                load_dim_idx   <= j;
                load_emb_data  <= (i * EMBED_DIM + j + 1) * 64;  // Spread values
            end
        end
        @(posedge clk); load_token_emb <= 1'b0;

        // Position embeddings: small values
        $display("[3] Loading %0d position embeddings...", MAX_SEQ_LEN);
        for (i = 0; i < MAX_SEQ_LEN; i = i + 1) begin
            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                @(posedge clk);
                load_pos_emb <= 1'b1;
                load_pos_idx <= i;
                load_dim_idx <= j;
                load_emb_data <= (i + 1) * 16;
            end
        end
        @(posedge clk); load_pos_emb <= 1'b0;

        #20;

        // --- Generate 3 tokens ---
        $display("");
        $display("[4] Generating 3 tokens autoregressively...");
        $display("    Seed token: 2");
        $display("");

        for (tok = 0; tok < 3; tok = tok + 1) begin
            @(posedge clk);
            valid_in   <= 1'b1;
            token_in   <= (tok == 0) ? 2 : tokens_generated[tok-1];
            position_in <= tok;
            start_cycle = total_cycles;
            @(posedge clk);
            valid_in <= 1'b0;

            cycle = 0;
            while (!valid_out && cycle < 2000) begin
                @(posedge clk);
                cycle = cycle + 1;
            end

            if (valid_out) begin
                tokens_generated[tok] = token_out;
                $display("    Token %0d: predicted=%0d, cycles=%0d, zero_skips=%0d",
                    tok, token_out, total_cycles - start_cycle, total_zero_skips);
                for (j = 0; j < EMBED_DIM; j = j + 1)
                    $display("      logit[%0d] = %0d (%.3f)", j,
                        $signed(logits_out[j*DATA_WIDTH +: DATA_WIDTH]),
                        $itor($signed(logits_out[j*DATA_WIDTH +: DATA_WIDTH])) / 256.0);
                $display("[PASS] Token %0d generated successfully", tok);
            end else begin
                $display("    [FAIL] Token %0d TIMEOUT after %0d cycles!", tok, cycle);
            end
            #20;
        end

        $display("");
        $display("================================================================");
        $display("  Performance Summary");
        $display("================================================================");
        $display("  Total inference cycles: %0d", total_cycles);
        $display("  Total zero-skips:       %0d", total_zero_skips);
        $display("  Sequence generated:     %0d → %0d → %0d → %0d",
            2, tokens_generated[0], tokens_generated[1], tokens_generated[2]);
        $display("================================================================");
        $finish;
    end

endmodule
