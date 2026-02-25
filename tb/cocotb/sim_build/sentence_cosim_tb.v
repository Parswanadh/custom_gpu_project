`timescale 1ns/1ps

module sentence_cosim_tb;
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
  reg load_token_emb, load_pos_emb;
  reg [3:0] load_token_idx;
  reg [1:0] load_dim_idx;
  reg signed [15:0] load_emb_data;
  reg [2:0] load_pos_idx;
  reg [63:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
  reg [63:0] ln_final_gamma, ln_final_beta;
  reg [255:0] wq_flat, wk_flat, wv_flat, wo_flat;
  reg [511:0] ffn_w1_flat;
  reg [127:0] ffn_b1_flat;
  reg [511:0] ffn_w2_flat;
  reg [63:0] ffn_b2_flat;
  wire [3:0] token_out;
  wire [63:0] logits_out;
  wire valid_out;
  integer cycle_count;
  integer total_cycles;
  integer token_count;

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
    $dumpfile("sentence_cosim.vcd");
    $dumpvars(0, sentence_cosim_tb);
  end

  initial begin
    clk = 0; rst = 1;
    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;
    token_in = 0; position_in = 0;
    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;
    load_pos_idx = 0;
    total_cycles = 0; token_count = 0;

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

    #35 rst = 0; #25;

    // ===== LOAD EMBEDDINGS =====
    @(negedge clk); load_token_emb = 1; load_token_idx = 0; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 0; load_dim_idx = 1; load_emb_data = 16'hffff;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 0; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 0; load_dim_idx = 3; load_emb_data = 16'h0008;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 1; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 1; load_dim_idx = 1; load_emb_data = 16'h0009;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 1; load_dim_idx = 2; load_emb_data = 16'h0004;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 1; load_dim_idx = 3; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 2; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 2; load_dim_idx = 1; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 2; load_dim_idx = 2; load_emb_data = 16'hfffa;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 2; load_dim_idx = 3; load_emb_data = 16'hfff1;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 3; load_dim_idx = 0; load_emb_data = 16'h000a;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 3; load_dim_idx = 1; load_emb_data = 16'h0010;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 3; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 3; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 4; load_dim_idx = 0; load_emb_data = 16'h0001;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 4; load_dim_idx = 1; load_emb_data = 16'h0008;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 4; load_dim_idx = 2; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 4; load_dim_idx = 3; load_emb_data = 16'h0009;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 5; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 5; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 5; load_dim_idx = 2; load_emb_data = 16'h0005;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 5; load_dim_idx = 3; load_emb_data = 16'hfff9;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 6; load_dim_idx = 0; load_emb_data = 16'h0004;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 6; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 6; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 6; load_dim_idx = 3; load_emb_data = 16'hfff6;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 7; load_dim_idx = 0; load_emb_data = 16'h000d;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 7; load_dim_idx = 1; load_emb_data = 16'hfff4;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 7; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 7; load_dim_idx = 3; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 8; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 8; load_dim_idx = 1; load_emb_data = 16'hffff;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 8; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 8; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 9; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 9; load_dim_idx = 1; load_emb_data = 16'h0006;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 9; load_dim_idx = 2; load_emb_data = 16'h0005;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 9; load_dim_idx = 3; load_emb_data = 16'hfff9;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 10; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 10; load_dim_idx = 1; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 10; load_dim_idx = 2; load_emb_data = 16'hffff;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 10; load_dim_idx = 3; load_emb_data = 16'h0008;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 11; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 11; load_dim_idx = 1; load_emb_data = 16'h0002;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 11; load_dim_idx = 2; load_emb_data = 16'hfff7;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 11; load_dim_idx = 3; load_emb_data = 16'h0002;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 12; load_dim_idx = 0; load_emb_data = 16'h0006;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 12; load_dim_idx = 1; load_emb_data = 16'h0001;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 12; load_dim_idx = 2; load_emb_data = 16'hfffb;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 12; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 13; load_dim_idx = 0; load_emb_data = 16'hfffa;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 13; load_dim_idx = 1; load_emb_data = 16'hfffe;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 13; load_dim_idx = 2; load_emb_data = 16'hfffe;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 13; load_dim_idx = 3; load_emb_data = 16'hfff7;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 14; load_dim_idx = 0; load_emb_data = 16'hfff4;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 14; load_dim_idx = 1; load_emb_data = 16'hfffa;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 14; load_dim_idx = 2; load_emb_data = 16'hfffd;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 14; load_dim_idx = 3; load_emb_data = 16'h0000;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 15; load_dim_idx = 0; load_emb_data = 16'h0007;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 15; load_dim_idx = 1; load_emb_data = 16'hfff1;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 15; load_dim_idx = 2; load_emb_data = 16'hfffa;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_token_emb = 1; load_token_idx = 15; load_dim_idx = 3; load_emb_data = 16'hfffb;
    @(negedge clk); load_token_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 0; load_emb_data = 16'h0004;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 1; load_emb_data = 16'hfff6;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 2; load_emb_data = 16'hfffd;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 0; load_dim_idx = 3; load_emb_data = 16'hfffe;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 0; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 1; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 2; load_emb_data = 16'h0004;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 1; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 0; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 1; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 2; load_dim_idx = 3; load_emb_data = 16'h0001;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 0; load_emb_data = 16'hfffc;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 1; load_emb_data = 16'hfffd;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 2; load_emb_data = 16'h0007;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 3; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 0; load_emb_data = 16'h0002;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 1; load_emb_data = 16'h000a;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 2; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 4; load_dim_idx = 3; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 0; load_emb_data = 16'h0000;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 1; load_emb_data = 16'h0004;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 2; load_emb_data = 16'h0000;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 5; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 1; load_emb_data = 16'h0001;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 2; load_emb_data = 16'h0002;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 6; load_dim_idx = 3; load_emb_data = 16'h0003;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 0; load_emb_data = 16'hfffd;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 1; load_emb_data = 16'hfff8;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 2; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    @(negedge clk); load_pos_emb = 1; load_pos_idx = 7; load_dim_idx = 3; load_emb_data = 16'hffff;
    @(negedge clk); load_pos_emb = 0;
    #20;

    $display("");
    $display("+=========================================================+");
    $display("|  BitbyBit GPU -- Sentence Processing Cosimulation        |");
    $display("|  Model: Real GPT-2 (Q8.8 quantized)                     |");
    $display("|  Tokens: [8,5,12,12,15]");
    $display("|  Sequence length: 5 tokens                                 |");
    $display("+=========================================================+");
    $display("");

    // ===== TOKEN 0: id=8 =====
    @(negedge clk);
    token_in = 8; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 1000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",
               0, 8, token_out, cycle_count, logits_out);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               0, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               0, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               0, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               0, 3, logits_out[48 +: 16]);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 0, 8);
    end
    repeat(5) @(negedge clk);

    // ===== TOKEN 1: id=5 =====
    @(negedge clk);
    token_in = 5; position_in = 1;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 1000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",
               1, 5, token_out, cycle_count, logits_out);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               1, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               1, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               1, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               1, 3, logits_out[48 +: 16]);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 1, 5);
    end
    repeat(5) @(negedge clk);

    // ===== TOKEN 2: id=12 =====
    @(negedge clk);
    token_in = 12; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 1000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",
               2, 12, token_out, cycle_count, logits_out);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               2, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               2, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               2, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               2, 3, logits_out[48 +: 16]);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 2, 12);
    end
    repeat(5) @(negedge clk);

    // ===== TOKEN 3: id=12 =====
    @(negedge clk);
    token_in = 12; position_in = 3;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 1000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",
               3, 12, token_out, cycle_count, logits_out);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               3, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               3, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               3, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               3, 3, logits_out[48 +: 16]);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 3, 12);
    end
    repeat(5) @(negedge clk);

    // ===== TOKEN 4: id=15 =====
    @(negedge clk);
    token_in = 15; position_in = 4;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 1000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end

    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",
               4, 15, token_out, cycle_count, logits_out);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               4, 0, logits_out[0 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               4, 1, logits_out[16 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               4, 2, logits_out[32 +: 16]);
      $display("  LOGIT pos=%0d dim=%0d hex=%h",
               4, 3, logits_out[48 +: 16]);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 4, 15);
    end
    repeat(5) @(negedge clk);

    $display("");
    $display("SUMMARY total_tokens=%0d total_cycles=%0d", token_count, total_cycles);
    $display("DONE");
    $finish;
  end
endmodule
