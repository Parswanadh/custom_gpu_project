`timescale 1ns/1ps
module scaled_cosim_tb;
  parameter VOCAB_SIZE  = 16;
  parameter MAX_SEQ_LEN = 8;
  parameter EMBED_DIM   = 64;
  parameter NUM_HEADS   = 8;
  parameter HEAD_DIM    = 8;
  parameter FFN_DIM     = 256;
  parameter NUM_LAYERS  = 2;
  parameter DATA_WIDTH  = 16;

  reg clk, rst;
  reg valid_in;
  reg [3:0] token_in;
  reg [2:0] position_in;
  reg load_token_emb, load_pos_emb;
  reg [3:0] load_token_idx;
  reg [5:0] load_dim_idx;
  reg signed [15:0] load_emb_data;
  reg [2:0] load_pos_idx;

  reg [1023:0] ln1_gamma, ln1_beta;
  reg [1023:0] ln2_gamma, ln2_beta;
  reg [1023:0] ln_final_gamma, ln_final_beta;
  reg [65535:0] wq_flat, wk_flat, wv_flat, wo_flat;
  reg [262143:0] ffn_w1_flat;
  reg [4095:0] ffn_b1_flat;
  reg [262143:0] ffn_w2_flat;
  reg [1023:0] ffn_b2_flat;
  wire [3:0] token_out;
  wire [1023:0] logits_out;
  wire valid_out;
  integer cycle_count;
  integer total_cycles;
  integer token_count;
  integer idx;

  // Memory arrays for weight loading
  reg [15:0] tok_emb_mem  [0:1023];
  reg [15:0] pos_emb_mem  [0:511];
  reg [15:0] ln1g_mem     [0:63];
  reg [15:0] ln1b_mem     [0:63];
  reg [15:0] ln2g_mem     [0:63];
  reg [15:0] ln2b_mem     [0:63];
  reg [15:0] wq_mem       [0:4095];
  reg [15:0] wk_mem       [0:4095];
  reg [15:0] wv_mem       [0:4095];
  reg [15:0] wo_mem       [0:4095];
  reg [15:0] fw1_mem      [0:16383];
  reg [15:0] fb1_mem      [0:255];
  reg [15:0] fw2_mem      [0:16383];
  reg [15:0] fb2_mem      [0:63];
  reg [15:0] lnfg_mem     [0:63];
  reg [15:0] lnfb_mem     [0:63];

  gpt2_engine #(
    .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),
    .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),
    .NUM_LAYERS(NUM_LAYERS), .DATA_WIDTH(DATA_WIDTH)
  ) dut (
    .clk(clk), .rst(rst),
    .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
    .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),
    .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
    .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
    .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
    .wq_flat(wq_flat), .wk_flat(wk_flat),
    .wv_flat(wv_flat), .wo_flat(wo_flat),
    .ffn_w1_flat(ffn_w1_flat), .ffn_b1_flat(ffn_b1_flat),
    .ffn_w2_flat(ffn_w2_flat), .ffn_b2_flat(ffn_b2_flat),
    .ln_final_gamma(ln_final_gamma), .ln_final_beta(ln_final_beta),
    .valid_in(valid_in), .token_in(token_in),
    .position_in(position_in),
    .token_out(token_out), .logits_out(logits_out),
    .valid_out(valid_out)
  );

  always #5 clk = ~clk;

  initial begin
    // Load weights via $readmemh
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/token_emb.hex", tok_emb_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/pos_emb.hex",   pos_emb_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln1_gamma.hex", ln1g_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln1_beta.hex",  ln1b_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln2_gamma.hex", ln2g_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln2_beta.hex",  ln2b_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/wq.hex",        wq_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/wk.hex",        wk_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/wv.hex",        wv_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/wo.hex",        wo_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ffn_w1.hex",    fw1_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ffn_b1.hex",    fb1_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ffn_w2.hex",    fw2_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ffn_b2.hex",    fb2_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln_final_gamma.hex", lnfg_mem);
    $readmemh("D:/Projects/BitbyBit/custom_gpu_project/weights/gpt2_dim64/hex_sim/ln_final_beta.hex",  lnfb_mem);
  end

  initial begin
    clk = 0; rst = 1;
    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;
    token_in = 0; position_in = 0;
    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;
    load_pos_idx = 0;
    total_cycles = 0; token_count = 0;

    // Pack flat weight buses from memory arrays
    for (idx = 0; idx < 64; idx = idx + 1) begin
      ln1_gamma[idx*DATA_WIDTH +: DATA_WIDTH] = ln1g_mem[idx];
      ln1_beta[idx*DATA_WIDTH +: DATA_WIDTH]  = ln1b_mem[idx];
      ln2_gamma[idx*DATA_WIDTH +: DATA_WIDTH] = ln2g_mem[idx];
      ln2_beta[idx*DATA_WIDTH +: DATA_WIDTH]  = ln2b_mem[idx];
      ln_final_gamma[idx*DATA_WIDTH +: DATA_WIDTH] = lnfg_mem[idx];
      ln_final_beta[idx*DATA_WIDTH +: DATA_WIDTH]  = lnfb_mem[idx];
      ffn_b2_flat[idx*DATA_WIDTH +: DATA_WIDTH] = fb2_mem[idx];
    end
    for (idx = 0; idx < 4096; idx = idx + 1) begin
      wq_flat[idx*DATA_WIDTH +: DATA_WIDTH] = wq_mem[idx];
      wk_flat[idx*DATA_WIDTH +: DATA_WIDTH] = wk_mem[idx];
      wv_flat[idx*DATA_WIDTH +: DATA_WIDTH] = wv_mem[idx];
      wo_flat[idx*DATA_WIDTH +: DATA_WIDTH] = wo_mem[idx];
    end
    for (idx = 0; idx < 16384; idx = idx + 1)
      ffn_w1_flat[idx*DATA_WIDTH +: DATA_WIDTH] = fw1_mem[idx];
    for (idx = 0; idx < 256; idx = idx + 1)
      ffn_b1_flat[idx*DATA_WIDTH +: DATA_WIDTH] = fb1_mem[idx];
    for (idx = 0; idx < 16384; idx = idx + 1)
      ffn_w2_flat[idx*DATA_WIDTH +: DATA_WIDTH] = fw2_mem[idx];

    #35 rst = 0; #25;

    // Load embeddings into DUT
    for (idx = 0; idx < 1024; idx = idx + 1) begin
      @(negedge clk);
      load_token_emb = 1;
      load_token_idx = idx / EMBED_DIM;
      load_dim_idx   = idx % EMBED_DIM;
      load_emb_data  = tok_emb_mem[idx];
      @(negedge clk); load_token_emb = 0;
    end
    for (idx = 0; idx < 512; idx = idx + 1) begin
      @(negedge clk);
      load_pos_emb = 1;
      load_pos_idx   = idx / EMBED_DIM;
      load_dim_idx   = idx % EMBED_DIM;
      load_emb_data  = pos_emb_mem[idx];
      @(negedge clk); load_pos_emb = 0;
    end
    #20;

    $display("");
    $display("CONFIG dim=64 ffn=256 vocab=16 layers=2 heads=8");
    $display("");
    // Token 0: id=8
    @(negedge clk);
    token_in = 8; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d",
               0, 8, token_out, cycle_count);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 3, logits_out[48 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 4, logits_out[64 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 5, logits_out[80 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 6, logits_out[96 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 7, logits_out[112 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 8, logits_out[128 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 9, logits_out[144 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 10, logits_out[160 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 11, logits_out[176 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 12, logits_out[192 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 13, logits_out[208 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 14, logits_out[224 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 0, 15, logits_out[240 +: 16]);
      $display("  ... (48 more dimensions)");
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 0, 8);
    end
    repeat(3) @(negedge clk);

    // Token 1: id=5
    @(negedge clk);
    token_in = 5; position_in = 1;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d",
               1, 5, token_out, cycle_count);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 3, logits_out[48 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 4, logits_out[64 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 5, logits_out[80 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 6, logits_out[96 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 7, logits_out[112 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 8, logits_out[128 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 9, logits_out[144 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 10, logits_out[160 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 11, logits_out[176 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 12, logits_out[192 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 13, logits_out[208 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 14, logits_out[224 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 1, 15, logits_out[240 +: 16]);
      $display("  ... (48 more dimensions)");
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 1, 5);
    end
    repeat(3) @(negedge clk);

    // Token 2: id=12
    @(negedge clk);
    token_in = 12; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d",
               2, 12, token_out, cycle_count);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 3, logits_out[48 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 4, logits_out[64 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 5, logits_out[80 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 6, logits_out[96 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 7, logits_out[112 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 8, logits_out[128 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 9, logits_out[144 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 10, logits_out[160 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 11, logits_out[176 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 12, logits_out[192 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 13, logits_out[208 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 14, logits_out[224 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 2, 15, logits_out[240 +: 16]);
      $display("  ... (48 more dimensions)");
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 2, 12);
    end
    repeat(3) @(negedge clk);

    // Token 3: id=12
    @(negedge clk);
    token_in = 12; position_in = 3;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d",
               3, 12, token_out, cycle_count);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 3, logits_out[48 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 4, logits_out[64 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 5, logits_out[80 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 6, logits_out[96 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 7, logits_out[112 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 8, logits_out[128 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 9, logits_out[144 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 10, logits_out[160 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 11, logits_out[176 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 12, logits_out[192 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 13, logits_out[208 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 14, logits_out[224 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 3, 15, logits_out[240 +: 16]);
      $display("  ... (48 more dimensions)");
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 3, 12);
    end
    repeat(3) @(negedge clk);

    // Token 4: id=15
    @(negedge clk);
    token_in = 15; position_in = 4;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d",
               4, 15, token_out, cycle_count);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 3, logits_out[48 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 4, logits_out[64 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 5, logits_out[80 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 6, logits_out[96 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 7, logits_out[112 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 8, logits_out[128 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 9, logits_out[144 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 10, logits_out[160 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 11, logits_out[176 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 12, logits_out[192 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 13, logits_out[208 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 14, logits_out[224 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h", 4, 15, logits_out[240 +: 16]);
      $display("  ... (48 more dimensions)");
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 4, 15);
    end
    repeat(3) @(negedge clk);

    $display("");
    $display("SUMMARY total_tokens=%0d total_cycles=%0d avg_cycles=%0d",
             token_count, total_cycles, total_cycles / token_count);
    $display("DONE");
    $finish;
  end
endmodule
