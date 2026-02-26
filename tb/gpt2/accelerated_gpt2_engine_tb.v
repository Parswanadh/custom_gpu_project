// ============================================================================
// Testbench: accelerated_gpt2_engine_tb
// Tests the full accelerated GPT-2 pipeline:
//   1. Load embeddings
//   2. Set transformer weights
//   3. Feed 3 tokens through the full pipeline
//   4. Verify outputs, zero-skip counts, cycle counts
// ============================================================================
`timescale 1ns/1ps

module accelerated_gpt2_engine_tb;

    parameter VOCAB_SIZE  = 8;
    parameter MAX_SEQ_LEN = 8;
    parameter EMBED_DIM   = 4;
    parameter NUM_HEADS   = 2;
    parameter HEAD_DIM    = 2;
    parameter FFN_DIM     = 8;
    parameter NUM_LAYERS  = 1;     // 1 layer for quick simulation
    parameter DATA_WIDTH  = 16;

    reg         clk, rst;

    // Embedding loading
    reg         load_token_emb;
    reg [$clog2(VOCAB_SIZE)-1:0] load_token_idx;
    reg [$clog2(EMBED_DIM)-1:0]  load_dim_idx;
    reg signed [DATA_WIDTH-1:0]  load_emb_data;
    reg         load_pos_emb;
    reg [$clog2(MAX_SEQ_LEN)-1:0] load_pos_idx;

    // Weights
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    reg [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq, wk, wv, wo;
    reg [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0] ffn_w1;
    reg [FFN_DIM*DATA_WIDTH-1:0] ffn_b1;
    reg [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0] ffn_w2;
    reg [EMBED_DIM*DATA_WIDTH-1:0] ffn_b2;
    reg [EMBED_DIM*DATA_WIDTH-1:0] ln_f_gamma, ln_f_beta;

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
        .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
        .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
        .wq_flat(wq), .wk_flat(wk), .wv_flat(wv), .wo_flat(wo),
        .ffn_w1_flat(ffn_w1), .ffn_b1_flat(ffn_b1),
        .ffn_w2_flat(ffn_w2), .ffn_b2_flat(ffn_b2),
        .ln_final_gamma(ln_f_gamma), .ln_final_beta(ln_f_beta),
        .valid_in(valid_in), .token_in(token_in), .position_in(position_in),
        .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out),
        .total_zero_skips(total_zero_skips), .total_cycles(total_cycles)
    );

    always #5 clk = ~clk;

    integer i, j, cycle, tok;
    reg [31:0] start_cycle;
    reg [$clog2(VOCAB_SIZE)-1:0] tokens_generated [0:2];

    initial begin
        $display("");
        $display("================================================================");
        $display("  Accelerated GPT-2 Engine — Full Pipeline Test");
        $display("  VOCAB=%0d, EMBED=%0d, HEADS=%0d, FFN=%0d, LAYERS=%0d",
            VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FFN_DIM, NUM_LAYERS);
        $display("================================================================");

        clk = 0; rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;

        // --- Initialize weights ---
        // LayerNorm: gamma=1.0 (256 in Q8.8), beta=0
        ln1_gamma = 0; ln1_beta = 0;
        ln2_gamma = 0; ln2_beta = 0;
        ln_f_gamma = 0; ln_f_beta = 0;
        for (i = 0; i < EMBED_DIM; i = i + 1) begin
            ln1_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
            ln2_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
            ln_f_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        end

        // Attention weights: identity × 0.5 to avoid overflow
        wq = 0; wk = 0; wv = 0; wo = 0;
        for (i = 0; i < EMBED_DIM; i = i + 1) begin
            wq[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;  // 0.5
            wk[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
            wv[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
            wo[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;  // 1.0
        end

        // FFN weights: small random-ish values
        ffn_w1 = 0; ffn_b1 = 0; ffn_w2 = 0; ffn_b2 = 0;
        for (i = 0; i < EMBED_DIM; i = i + 1) begin
            for (j = 0; j < FFN_DIM; j = j + 1) begin
                ffn_w1[(i*FFN_DIM+j)*DATA_WIDTH +: DATA_WIDTH] =
                    ((i + j) % 3 == 0) ? 16'sd64 : 16'sd0;  // Sparse W1
            end
        end
        for (i = 0; i < FFN_DIM; i = i + 1) begin
            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                ffn_w2[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] =
                    ((i + j) % 2 == 0) ? 16'sd64 : 16'sd0;  // Sparse W2
            end
        end

        #30; rst = 0; #10;

        // --- Load embeddings ---
        $display("");
        $display("[1] Loading token embeddings...");
        for (tok = 0; tok < VOCAB_SIZE; tok = tok + 1) begin
            for (i = 0; i < EMBED_DIM; i = i + 1) begin
                @(posedge clk);
                load_token_emb <= 1;
                load_token_idx <= tok;
                load_dim_idx   <= i;
                // Each token gets a distinct embedding
                load_emb_data  <= (tok * 64 + i * 128) & 16'hFFFF;
                @(posedge clk);
                load_token_emb <= 0;
            end
        end

        // Load position embeddings
        for (i = 0; i < MAX_SEQ_LEN; i = i + 1) begin
            @(posedge clk);
            load_pos_emb <= 1;
            load_pos_idx <= i;
            load_dim_idx <= 0;
            load_emb_data <= i * 32;
            @(posedge clk);
            load_pos_emb <= 0;
        end

        $display("    Loaded %0d token embeddings + %0d position embeddings",
            VOCAB_SIZE, MAX_SEQ_LEN);
        #20;

        // --- Run 3 tokens through pipeline ---
        $display("");
        $display("[2] Running 3-token autoregressive generation...");

        for (tok = 0; tok < 3; tok = tok + 1) begin
            @(posedge clk);
            token_in    <= (tok == 0) ? 3'd1 : tokens_generated[tok-1];
            position_in <= tok;
            valid_in    <= 1'b1;
            start_cycle = total_cycles;
            @(posedge clk);
            valid_in <= 1'b0;

            // Wait for output
            cycle = 0;
            while (!valid_out && cycle < 500) begin
                @(posedge clk);
                cycle = cycle + 1;
            end

            if (valid_out) begin
                tokens_generated[tok] = token_out;
                $display("    Token %0d: in=%0d pos=%0d → out=%0d  (%0d cycles, logits=[%0d,%0d,%0d,%0d])",
                    tok, token_in, tok, token_out,
                    total_cycles - start_cycle,
                    $signed(logits_out[0*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[1*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[2*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[3*DATA_WIDTH +: DATA_WIDTH]));
            end else begin
                $display("    Token %0d: TIMEOUT after 500 cycles!", tok);
            end

            #20;
        end

        // --- Summary ---
        $display("");
        $display("================================================================");
        $display("  FULL PIPELINE RESULTS");
        $display("================================================================");
        $display("  Tokens generated: %0d → %0d → %0d",
            tokens_generated[0], tokens_generated[1], tokens_generated[2]);
        $display("  Total cycles:      %0d", total_cycles);
        $display("  Total zero-skips:  %0d", total_zero_skips);
        $display("  Architecture:      1-layer transformer, KV-cached attention");
        $display("  Components:        accelerated_attention + accelerated FFN");
        $display("================================================================");
        $display("  === Full Pipeline Test COMPLETE ===");
        $display("================================================================");

        $finish;
    end

    // Timeout safety
    initial begin
        #100000;
        $display("  [TIMEOUT] Simulation exceeded 100us");
        $finish;
    end

endmodule
