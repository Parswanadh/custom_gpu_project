// ============================================================================
// Testbench: multi_layer_test
// Tests the accelerated GPT-2 engine with NUM_LAYERS=2 and NUM_LAYERS=4
// Verifies that multiple transformer layers process correctly
// ============================================================================
`timescale 1ns/1ps

module multi_layer_test;

    parameter VOCAB_SIZE  = 8;
    parameter MAX_SEQ_LEN = 8;
    parameter EMBED_DIM   = 4;
    parameter NUM_HEADS   = 2;
    parameter HEAD_DIM    = 2;
    parameter FFN_DIM     = 8;
    parameter NUM_LAYERS  = 2;     // 2 layers!
    parameter DATA_WIDTH  = 16;

    reg         clk, rst, valid_in;
    reg         load_token_emb, load_pos_emb;
    reg [$clog2(VOCAB_SIZE)-1:0]  load_token_idx;
    reg [$clog2(EMBED_DIM)-1:0]   load_dim_idx;
    reg signed [DATA_WIDTH-1:0]   load_emb_data;
    reg [$clog2(MAX_SEQ_LEN)-1:0] load_pos_idx;

    reg [EMBED_DIM*DATA_WIDTH-1:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    reg [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq, wk, wv, wo;
    reg [EMBED_DIM*FFN_DIM*DATA_WIDTH-1:0] ffn_w1;
    reg [FFN_DIM*DATA_WIDTH-1:0] ffn_b1;
    reg [FFN_DIM*EMBED_DIM*DATA_WIDTH-1:0] ffn_w2;
    reg [EMBED_DIM*DATA_WIDTH-1:0] ffn_b2, ln_f_gamma, ln_f_beta;

    reg [$clog2(VOCAB_SIZE)-1:0]  token_in;
    reg [$clog2(MAX_SEQ_LEN)-1:0] position_in;
    wire [$clog2(VOCAB_SIZE)-1:0] token_out;
    wire [EMBED_DIM*DATA_WIDTH-1:0] logits_out;
    wire valid_out;
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

    integer i, j, tok, cycle;
    reg [31:0] start_cyc;
    reg [$clog2(VOCAB_SIZE)-1:0] gen [0:3];

    initial begin
        $display("");
        $display("================================================================");
        $display("  Multi-Layer Transformer Test");
        $display("  Config: LAYERS=%0d, EMBED=%0d, FFN=%0d, HEADS=%0d",
            NUM_LAYERS, EMBED_DIM, FFN_DIM, NUM_HEADS);
        $display("================================================================");

        clk = 0; rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;

        // Weights (gamma=1.0, beta=0, identity Q/K/V/O, sparse FFN)
        ln1_gamma=0; ln1_beta=0; ln2_gamma=0; ln2_beta=0;
        ln_f_gamma=0; ln_f_beta=0;
        wq=0; wk=0; wv=0; wo=0;
        ffn_w1=0; ffn_b1=0; ffn_w2=0; ffn_b2=0;

        for (i=0; i<EMBED_DIM; i=i+1) begin
            ln1_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
            ln2_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
            ln_f_gamma[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
            wq[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
            wk[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
            wv[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
            wo[(i*EMBED_DIM+i)*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        end
        for (i=0; i<EMBED_DIM; i=i+1)
            for (j=0; j<FFN_DIM; j=j+1)
                ffn_w1[(i*FFN_DIM+j)*DATA_WIDTH +: DATA_WIDTH] =
                    ((i+j)%3==0) ? 16'sd64 : 16'sd0;
        for (i=0; i<FFN_DIM; i=i+1)
            for (j=0; j<EMBED_DIM; j=j+1)
                ffn_w2[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH] =
                    ((i+j)%2==0) ? 16'sd64 : 16'sd0;

        #30; rst=0; #10;

        // Load embeddings
        $display("[1] Loading embeddings...");
        for (tok=0; tok<VOCAB_SIZE; tok=tok+1)
            for (i=0; i<EMBED_DIM; i=i+1) begin
                @(posedge clk);
                load_token_emb<=1; load_token_idx<=tok; load_dim_idx<=i;
                load_emb_data <= (tok*64+i*128) & 16'hFFFF;
                @(posedge clk); load_token_emb<=0;
            end
        for (i=0; i<MAX_SEQ_LEN; i=i+1) begin
            @(posedge clk);
            load_pos_emb<=1; load_pos_idx<=i; load_dim_idx<=0;
            load_emb_data<=i*32;
            @(posedge clk); load_pos_emb<=0;
        end
        $display("    Done.");
        #20;

        // Generate 4 tokens
        $display("[2] Running 4-token generation with %0d layers...", NUM_LAYERS);
        for (tok=0; tok<4; tok=tok+1) begin
            @(posedge clk);
            token_in <= (tok==0) ? 3'd2 : gen[tok-1];
            position_in <= tok;
            valid_in <= 1;
            start_cyc = total_cycles;
            @(posedge clk); valid_in<=0;

            cycle=0;
            while (!valid_out && cycle<5000) begin
                @(posedge clk); cycle=cycle+1;
            end

            if (valid_out) begin
                gen[tok] = token_out;
                $display("    Token %0d: in=%0d → out=%0d (%0d cycles, logits=[%0d,%0d,%0d,%0d])",
                    tok, token_in, token_out, total_cycles-start_cyc,
                    $signed(logits_out[0*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[1*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[2*DATA_WIDTH +: DATA_WIDTH]),
                    $signed(logits_out[3*DATA_WIDTH +: DATA_WIDTH]));
            end else
                $display("    Token %0d: TIMEOUT!", tok);
            #20;
        end

        $display("");
        $display("================================================================");
        $display("  MULTI-LAYER RESULTS (%0d layers)", NUM_LAYERS);
        $display("================================================================");
        $display("  Generated: %0d → %0d → %0d → %0d", gen[0], gen[1], gen[2], gen[3]);
        $display("  Total cycles:     %0d", total_cycles);
        $display("  Total zero-skips: %0d", total_zero_skips);
        $display("  Cycles/layer/token: ~%0d", (total_cycles) / (4 * NUM_LAYERS));
        $display("================================================================");
        $display("  === Test COMPLETE ===");
        $display("================================================================");
        $finish;
    end

    initial begin
        #2000000;
        $display("  [TIMEOUT] Simulation exceeded 2ms");
        $finish;
    end

endmodule
