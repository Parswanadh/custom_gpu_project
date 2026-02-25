`timescale 1ns/1ps

module cosim_tb;

  parameter VOCAB_SIZE  = 16;
  parameter MAX_SEQ_LEN = 8;
  parameter EMBED_DIM   = 4;
  parameter NUM_HEADS   = 2;
  parameter HEAD_DIM    = 2;
  parameter FFN_DIM     = 8;
  parameter NUM_LAYERS  = 2;
  parameter DATA_WIDTH  = 16;

  reg clk, rst;
  reg valid_in;
  reg [3:0] token_in;
  reg [2:0] position_in;

  // Embedding loading
  reg load_token_emb, load_pos_emb;
  reg [3:0] load_token_idx;
  reg [1:0] load_dim_idx;
  reg signed [15:0] load_emb_data;
  reg [2:0] load_pos_idx;

  // Weight buses
  reg [63:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
  reg [63:0] ln_final_gamma, ln_final_beta;
  reg [255:0] wq_flat, wk_flat, wv_flat, wo_flat;
  reg [511:0] ffn_w1_flat;
  reg [127:0] ffn_b1_flat;
  reg [511:0] ffn_w2_flat;
  reg [63:0] ffn_b2_flat;

  // Output
  wire [3:0] token_out;
  wire [63:0] logits_out;
  wire valid_out;

  integer cycle_count;
  integer i;

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
    $dumpfile("cosim.vcd");
    $dumpvars(0, cosim_tb);
  end

  initial begin
    clk = 0; rst = 1;
    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;
    token_in = 0; position_in = 0;
    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;
    load_pos_idx = 0;

    // ===== REAL GPT-2 WEIGHTS (Q8.8) =====
    ln1_gamma = 64'h0100010001000100;
    ln1_beta  = 64'h0000000000000000;
    ln2_gamma = 64'h0100010001000100;
    ln2_beta  = 64'h0000000000000000;
    ln_final_gamma = 64'h0100010001000100;
    ln_final_beta  = 64'h0000000000000000;
    wq_flat  = 256'hfff2000200030002fff1fffffff7fffcfff6000a000500060001000700040003;
    wk_flat  = 256'hfff5fffb0006fff8fffe0001fff60001000a00050000fff2fffd00020003fffc;
    wv_flat  = 256'hfff9000100080003fffa0000fffcfffbfffffff6fffefff9fff900040002fffd;
    wo_flat  = 256'h0003fff80000fffafff9fff9fff6fff5fffa0009ffff0002000500040005fffc;
    ffn_w1_flat = 512'h00020003fff300090007fff700040002fffb0006000bfff3fffe0002fffa0006000300000007fffd0001ffff000100040004fffcfffafffb000400010003fffd;
    ffn_b1_flat = 128'h00000000000000000000000000000000;
    ffn_w2_flat = 512'h000000000000fffcfff600060001fffbfff70001fff5fff8fffb0006fffffffe0000ffef00000003fff2ffff00030008fffb000800000003fffbfffa00000006;
    ffn_b2_flat = 64'h0000000000000000;

    #35 rst = 0;
    #25;

    // ===== LOAD TOKEN EMBEDDINGS =====
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 0; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 0; load_dim_idx = 1; load_emb_data = 16'hffff;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 0; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 0; load_dim_idx = 3; load_emb_data = 16'h0008;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 1; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 1; load_dim_idx = 1; load_emb_data = 16'h0009;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 1; load_dim_idx = 2; load_emb_data = 16'h0004;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 1; load_dim_idx = 3; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 2; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 2; load_dim_idx = 1; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 2; load_dim_idx = 2; load_emb_data = 16'hfffa;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 2; load_dim_idx = 3; load_emb_data = 16'hfff1;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 3; load_dim_idx = 0; load_emb_data = 16'h000a;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 3; load_dim_idx = 1; load_emb_data = 16'h0010;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 3; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 3; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 4; load_dim_idx = 0; load_emb_data = 16'h0001;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 4; load_dim_idx = 1; load_emb_data = 16'h0008;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 4; load_dim_idx = 2; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 4; load_dim_idx = 3; load_emb_data = 16'h0009;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 5; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 5; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 5; load_dim_idx = 2; load_emb_data = 16'h0005;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 5; load_dim_idx = 3; load_emb_data = 16'hfff9;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 6; load_dim_idx = 0; load_emb_data = 16'h0004;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 6; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 6; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 6; load_dim_idx = 3; load_emb_data = 16'hfff6;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 7; load_dim_idx = 0; load_emb_data = 16'h000d;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 7; load_dim_idx = 1; load_emb_data = 16'hfff4;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 7; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 7; load_dim_idx = 3; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 8; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 8; load_dim_idx = 1; load_emb_data = 16'hffff;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 8; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 8; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 9; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 9; load_dim_idx = 1; load_emb_data = 16'h0006;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 9; load_dim_idx = 2; load_emb_data = 16'h0005;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 9; load_dim_idx = 3; load_emb_data = 16'hfff9;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 10; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 10; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 10; load_dim_idx = 2; load_emb_data = 16'hffff;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 10; load_dim_idx = 3; load_emb_data = 16'h0008;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 11; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 11; load_dim_idx = 1; load_emb_data = 16'h0002;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 11; load_dim_idx = 2; load_emb_data = 16'hfff7;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 11; load_dim_idx = 3; load_emb_data = 16'h0002;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 12; load_dim_idx = 0; load_emb_data = 16'h0006;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 12; load_dim_idx = 1; load_emb_data = 16'h0001;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 12; load_dim_idx = 2; load_emb_data = 16'hfffb;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 12; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 13; load_dim_idx = 0; load_emb_data = 16'hfffa;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 13; load_dim_idx = 1; load_emb_data = 16'hfffe;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 13; load_dim_idx = 2; load_emb_data = 16'hfffe;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 13; load_dim_idx = 3; load_emb_data = 16'hfff7;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 14; load_dim_idx = 0; load_emb_data = 16'hfff4;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 14; load_dim_idx = 1; load_emb_data = 16'hfffa;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 14; load_dim_idx = 2; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 14; load_dim_idx = 3; load_emb_data = 16'h0000;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 15; load_dim_idx = 0; load_emb_data = 16'h0007;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 15; load_dim_idx = 1; load_emb_data = 16'hfff1;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 15; load_dim_idx = 2; load_emb_data = 16'hfffa;
    @(negedge clk);
    load_token_emb = 0;
    @(negedge clk);
    load_token_emb = 1; load_token_idx = 15; load_dim_idx = 3; load_emb_data = 16'hfffb;
    @(negedge clk);
    load_token_emb = 0;

    // ===== LOAD POSITION EMBEDDINGS =====
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 0; load_emb_data = 16'h0004;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 1; load_emb_data = 16'hfff6;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 2; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 1; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 2; load_emb_data = 16'h0004;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 0; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 1; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 3; load_emb_data = 16'h0001;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 1; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 2; load_emb_data = 16'h0007;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 0; load_emb_data = 16'h0002;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 1; load_emb_data = 16'h000a;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 3; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 0; load_emb_data = 16'h0000;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 1; load_emb_data = 16'h0004;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 1; load_emb_data = 16'h0001;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 2; load_emb_data = 16'h0002;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 3; load_emb_data = 16'h0003;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 1; load_emb_data = 16'hfff8;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 2; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;
    @(negedge clk);
    load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk);
    load_pos_emb = 0;

    #20;

    $display("");
    $display("+=========================================================+");
    $display("|  BitbyBit GPU -- Real GPT-2 Weight Cosimulation          |");
    $display("|  Source: HuggingFace openai-community/gpt2               |");
    $display("|  Weights: Q8.8 quantized (first 4 dims extracted)       |");
    $display("+=========================================================+");
    $display("");

    // ===== INFERENCE: Token 0 =====
    @(negedge clk);
    token_in = 0; position_in = 0;
    valid_in = 1;
    @(negedge clk);
    valid_in = 0;

    cycle_count = 0;
    while (!valid_out && cycle_count < 500) begin
      @(negedge clk);
      cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      $display("COSIM_RESULT token=0 predicted=%0d cycles=%0d logits_hex=%h",
               token_out, cycle_count, logits_out);
      $display("COSIM_LOGIT token=0 dim=0 value=%h",
               logits_out[0 +: 16]);
      $display("COSIM_LOGIT token=0 dim=1 value=%h",
               logits_out[16 +: 16]);
      $display("COSIM_LOGIT token=0 dim=2 value=%h",
               logits_out[32 +: 16]);
      $display("COSIM_LOGIT token=0 dim=3 value=%h",
               logits_out[48 +: 16]);
    end else begin
      $display("COSIM_ERROR token=0 TIMEOUT");
    end

    // Wait between inferences
    repeat(10) @(negedge clk);

    // ===== INFERENCE: Token 3 =====
    @(negedge clk);
    token_in = 3; position_in = 0;
    valid_in = 1;
    @(negedge clk);
    valid_in = 0;

    cycle_count = 0;
    while (!valid_out && cycle_count < 500) begin
      @(negedge clk);
      cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      $display("COSIM_RESULT token=3 predicted=%0d cycles=%0d logits_hex=%h",
               token_out, cycle_count, logits_out);
      $display("COSIM_LOGIT token=3 dim=0 value=%h",
               logits_out[0 +: 16]);
      $display("COSIM_LOGIT token=3 dim=1 value=%h",
               logits_out[16 +: 16]);
      $display("COSIM_LOGIT token=3 dim=2 value=%h",
               logits_out[32 +: 16]);
      $display("COSIM_LOGIT token=3 dim=3 value=%h",
               logits_out[48 +: 16]);
    end else begin
      $display("COSIM_ERROR token=3 TIMEOUT");
    end

    // Wait between inferences
    repeat(10) @(negedge clk);

    // ===== INFERENCE: Token 5 =====
    @(negedge clk);
    token_in = 5; position_in = 0;
    valid_in = 1;
    @(negedge clk);
    valid_in = 0;

    cycle_count = 0;
    while (!valid_out && cycle_count < 500) begin
      @(negedge clk);
      cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      $display("COSIM_RESULT token=5 predicted=%0d cycles=%0d logits_hex=%h",
               token_out, cycle_count, logits_out);
      $display("COSIM_LOGIT token=5 dim=0 value=%h",
               logits_out[0 +: 16]);
      $display("COSIM_LOGIT token=5 dim=1 value=%h",
               logits_out[16 +: 16]);
      $display("COSIM_LOGIT token=5 dim=2 value=%h",
               logits_out[32 +: 16]);
      $display("COSIM_LOGIT token=5 dim=3 value=%h",
               logits_out[48 +: 16]);
    end else begin
      $display("COSIM_ERROR token=5 TIMEOUT");
    end

    // Wait between inferences
    repeat(10) @(negedge clk);

    $display("");
    $display("COSIM_DONE");
    $finish;
  end

endmodule
