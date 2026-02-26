// ============================================================================
// Testbench: real_weight_test
// Loads REAL GPT-2 weights from hex files via $readmemh
// Runs 4 tokens through the accelerated pipeline with actual model weights
// ============================================================================
`timescale 1ns/1ps

module real_weight_test;

    parameter VOCAB_SIZE  = 16;
    parameter MAX_SEQ_LEN = 8;
    parameter EMBED_DIM   = 4;
    parameter NUM_HEADS   = 2;
    parameter HEAD_DIM    = 2;
    parameter FFN_DIM     = 8;
    parameter NUM_LAYERS  = 1;
    parameter DATA_WIDTH  = 16;

    reg clk, rst, valid_in;
    reg load_token_emb, load_pos_emb;
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

    // Weight memory for loading from hex files
    reg [DATA_WIDTH-1:0] tok_emb_mem [0:VOCAB_SIZE*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] pos_emb_mem [0:MAX_SEQ_LEN*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] wq_mem  [0:EMBED_DIM*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] wk_mem  [0:EMBED_DIM*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] wv_mem  [0:EMBED_DIM*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] wo_mem  [0:EMBED_DIM*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] w1_mem  [0:EMBED_DIM*FFN_DIM-1];
    reg [DATA_WIDTH-1:0] b1_mem  [0:FFN_DIM-1];
    reg [DATA_WIDTH-1:0] w2_mem  [0:FFN_DIM*EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] b2_mem  [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] ln1g_mem [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] ln1b_mem [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] ln2g_mem [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] ln2b_mem [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] lnfg_mem [0:EMBED_DIM-1];
    reg [DATA_WIDTH-1:0] lnfb_mem [0:EMBED_DIM-1];

    integer i, j, tok, cycle;
    reg [31:0] start_cyc;
    reg [$clog2(VOCAB_SIZE)-1:0] gen [0:3];

    initial begin
        $display("");
        $display("================================================================");
        $display("  Real GPT-2 Weight Test");
        $display("  Loading actual model weights from hex files");
        $display("================================================================");

        clk=0; rst=1; valid_in=0;
        load_token_emb=0; load_pos_emb=0;

        // Load weights from hex files
        $display("[1] Loading real GPT-2 weights from hex files...");
        $readmemh("weights/gpt2_real/hex/token_emb.hex", tok_emb_mem);
        $readmemh("weights/gpt2_real/hex/pos_emb.hex", pos_emb_mem);
        $readmemh("weights/gpt2_real/hex/wq.hex", wq_mem);
        $readmemh("weights/gpt2_real/hex/wk.hex", wk_mem);
        $readmemh("weights/gpt2_real/hex/wv.hex", wv_mem);
        $readmemh("weights/gpt2_real/hex/wo.hex", wo_mem);
        $readmemh("weights/gpt2_real/hex/ffn_w1.hex", w1_mem);
        $readmemh("weights/gpt2_real/hex/ffn_b1.hex", b1_mem);
        $readmemh("weights/gpt2_real/hex/ffn_w2.hex", w2_mem);
        $readmemh("weights/gpt2_real/hex/ffn_b2.hex", b2_mem);
        $readmemh("weights/gpt2_real/hex/ln1_gamma.hex", ln1g_mem);
        $readmemh("weights/gpt2_real/hex/ln1_beta.hex", ln1b_mem);
        $readmemh("weights/gpt2_real/hex/ln2_gamma.hex", ln2g_mem);
        $readmemh("weights/gpt2_real/hex/ln2_beta.hex", ln2b_mem);
        $readmemh("weights/gpt2_real/hex/ln_final_gamma.hex", lnfg_mem);
        $readmemh("weights/gpt2_real/hex/ln_final_beta.hex", lnfb_mem);

        // Pack weights into flat vectors
        wq=0; wk=0; wv=0; wo=0;
        ffn_w1=0; ffn_b1=0; ffn_w2=0; ffn_b2=0;
        ln1_gamma=0; ln1_beta=0; ln2_gamma=0; ln2_beta=0;
        ln_f_gamma=0; ln_f_beta=0;

        for (i=0; i<EMBED_DIM*EMBED_DIM; i=i+1) begin
            wq[i*DATA_WIDTH +: DATA_WIDTH] = wq_mem[i];
            wk[i*DATA_WIDTH +: DATA_WIDTH] = wk_mem[i];
            wv[i*DATA_WIDTH +: DATA_WIDTH] = wv_mem[i];
            wo[i*DATA_WIDTH +: DATA_WIDTH] = wo_mem[i];
        end
        for (i=0; i<EMBED_DIM*FFN_DIM; i=i+1)
            ffn_w1[i*DATA_WIDTH +: DATA_WIDTH] = w1_mem[i];
        for (i=0; i<FFN_DIM; i=i+1)
            ffn_b1[i*DATA_WIDTH +: DATA_WIDTH] = b1_mem[i];
        for (i=0; i<FFN_DIM*EMBED_DIM; i=i+1)
            ffn_w2[i*DATA_WIDTH +: DATA_WIDTH] = w2_mem[i];
        for (i=0; i<EMBED_DIM; i=i+1) begin
            ffn_b2[i*DATA_WIDTH +: DATA_WIDTH] = b2_mem[i];
            ln1_gamma[i*DATA_WIDTH +: DATA_WIDTH] = ln1g_mem[i];
            ln1_beta[i*DATA_WIDTH +: DATA_WIDTH] = ln1b_mem[i];
            ln2_gamma[i*DATA_WIDTH +: DATA_WIDTH] = ln2g_mem[i];
            ln2_beta[i*DATA_WIDTH +: DATA_WIDTH] = ln2b_mem[i];
            ln_f_gamma[i*DATA_WIDTH +: DATA_WIDTH] = lnfg_mem[i];
            ln_f_beta[i*DATA_WIDTH +: DATA_WIDTH] = lnfb_mem[i];
        end

        $display("    Weights loaded from hex files");
        #30; rst=0; #10;

        // Load token embeddings from hex
        $display("[2] Loading embeddings into engine...");
        for (tok=0; tok<VOCAB_SIZE; tok=tok+1)
            for (i=0; i<EMBED_DIM; i=i+1) begin
                @(posedge clk);
                load_token_emb<=1; load_token_idx<=tok; load_dim_idx<=i;
                load_emb_data <= tok_emb_mem[tok*EMBED_DIM + i];
                @(posedge clk); load_token_emb<=0;
            end
        for (i=0; i<MAX_SEQ_LEN; i=i+1) begin
            @(posedge clk);
            load_pos_emb<=1; load_pos_idx<=i; load_dim_idx<=0;
            load_emb_data <= pos_emb_mem[i*EMBED_DIM];
            @(posedge clk); load_pos_emb<=0;
        end
        $display("    Done.");
        #20;

        // Run 4 tokens
        $display("[3] Running 4-token autoregressive generation with REAL weights...");
        for (tok=0; tok<4; tok=tok+1) begin
            @(posedge clk);
            token_in <= (tok==0) ? 4'd5 : gen[tok-1];  // Start with token 5
            position_in <= tok;
            valid_in <= 1;
            start_cyc = total_cycles;
            @(posedge clk); valid_in<=0;

            cycle=0;
            while (!valid_out && cycle<2000) begin
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
        $display("  REAL WEIGHT RESULTS");
        $display("================================================================");
        $display("  Generated: %0d → %0d → %0d → %0d", gen[0],gen[1],gen[2],gen[3]);
        $display("  Total cycles:     %0d", total_cycles);
        $display("  Total zero-skips: %0d", total_zero_skips);
        $display("  Weight source:    Real GPT-2 (117M) via Q8.8 quantization");
        $display("================================================================");
        $display("  === Test COMPLETE ===");
        $display("================================================================");
        $finish;
    end

    initial begin #500000; $display("TIMEOUT"); $finish; end
endmodule
