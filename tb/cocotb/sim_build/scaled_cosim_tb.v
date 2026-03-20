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

  // Load-based transformer weight interface
  reg load_ln_en;
  reg [1:0] load_layer_idx;
  reg load_ln_sel, load_ln_is_gamma;
  reg [5:0] load_ln_dim;
  reg signed [15:0] load_ln_data;
  reg load_attn_weight_en;
  reg [1:0] load_attn_matrix_sel;
  reg [5:0] load_attn_row, load_attn_col;
  reg signed [15:0] load_attn_data;
  reg load_ffn_weight_en;
  reg load_ffn_layer_sel, load_ffn_is_bias;
  reg [7:0] load_ffn_row, load_ffn_col;
  reg signed [15:0] load_ffn_data;

  wire [3:0] token_out;
  wire [1023:0] logits_out;
  wire valid_out;
  wire [31:0] total_zero_skips;
  wire [31:0] total_cycles_hw;
  integer cycle_count;
  integer total_cycles;
  integer token_count;
  integer idx, row_idx, col_idx, layer_idx_i;

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
    .total_zero_skips(total_zero_skips), .total_cycles(total_cycles_hw)
  );

  always #5 clk = ~clk;

  initial begin
    $dumpfile("scaled_cosim.vcd");
    $dumpvars(0, scaled_cosim_tb);
  end

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
    load_ln_en = 0; load_layer_idx = 0; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = 0; load_ln_data = 0;
    load_attn_weight_en = 0; load_attn_matrix_sel = 0; load_attn_row = 0; load_attn_col = 0; load_attn_data = 0;
    load_ffn_weight_en = 0; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = 0; load_ffn_col = 0; load_ffn_data = 0;
    total_cycles = 0; token_count = 0;

    #35 rst = 0; #25;

    // Load LayerNorm parameters for each transformer layer
    for (layer_idx_i = 0; layer_idx_i < NUM_LAYERS; layer_idx_i = layer_idx_i + 1) begin
      for (idx = 0; idx < EMBED_DIM; idx = idx + 1) begin
        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = ln1g_mem[idx];
        @(negedge clk); load_ln_en = 0;
        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = ln1b_mem[idx];
        @(negedge clk); load_ln_en = 0;
        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 1; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = ln2g_mem[idx];
        @(negedge clk); load_ln_en = 0;
        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 1; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = ln2b_mem[idx];
        @(negedge clk); load_ln_en = 0;
      end
    end

    // Load final LayerNorm (load_layer_idx == NUM_LAYERS)
    for (idx = 0; idx < EMBED_DIM; idx = idx + 1) begin
      @(negedge clk); load_ln_en = 1; load_layer_idx = NUM_LAYERS; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = lnfg_mem[idx];
      @(negedge clk); load_ln_en = 0;
      @(negedge clk); load_ln_en = 1; load_layer_idx = NUM_LAYERS; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = lnfb_mem[idx];
      @(negedge clk); load_ln_en = 0;
    end

    // Load attention matrices Wq/Wk/Wv/Wo
    for (row_idx = 0; row_idx < EMBED_DIM; row_idx = row_idx + 1) begin
      for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin
        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2'd0; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wq_mem[row_idx*EMBED_DIM + col_idx];
        @(negedge clk); load_attn_weight_en = 0;
        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2'd1; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wk_mem[row_idx*EMBED_DIM + col_idx];
        @(negedge clk); load_attn_weight_en = 0;
        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2'd2; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wv_mem[row_idx*EMBED_DIM + col_idx];
        @(negedge clk); load_attn_weight_en = 0;
        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2'd3; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wo_mem[row_idx*EMBED_DIM + col_idx];
        @(negedge clk); load_attn_weight_en = 0;
      end
    end

    // Load FFN W1 and b1
    for (row_idx = 0; row_idx < EMBED_DIM; row_idx = row_idx + 1) begin
      for (col_idx = 0; col_idx < FFN_DIM; col_idx = col_idx + 1) begin
        @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = row_idx; load_ffn_col = col_idx; load_ffn_data = fw1_mem[row_idx*FFN_DIM + col_idx];
        @(negedge clk); load_ffn_weight_en = 0;
      end
    end
    for (col_idx = 0; col_idx < FFN_DIM; col_idx = col_idx + 1) begin
      @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = col_idx; load_ffn_data = fb1_mem[col_idx];
      @(negedge clk); load_ffn_weight_en = 0;
    end

    // Load FFN W2 and b2
    for (row_idx = 0; row_idx < FFN_DIM; row_idx = row_idx + 1) begin
      for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin
        @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 0; load_ffn_row = row_idx; load_ffn_col = col_idx; load_ffn_data = fw2_mem[row_idx*EMBED_DIM + col_idx];
        @(negedge clk); load_ffn_weight_en = 0;
      end
    end
    for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin
      @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = col_idx; load_ffn_data = fb2_mem[col_idx];
      @(negedge clk); load_ffn_weight_en = 0;
    end

    // Load embeddings into DUT
    for (idx = 0; idx < VOCAB_SIZE*EMBED_DIM; idx = idx + 1) begin
      @(negedge clk);
      load_token_emb = 1;
      load_token_idx = idx / EMBED_DIM;
      load_dim_idx   = idx % EMBED_DIM;
      load_emb_data  = tok_emb_mem[idx];
      @(negedge clk); load_token_emb = 0;
    end
    for (idx = 0; idx < MAX_SEQ_LEN*EMBED_DIM; idx = idx + 1) begin
      @(negedge clk);
      load_pos_emb = 1;
      load_pos_idx = idx / EMBED_DIM;
      load_dim_idx = idx % EMBED_DIM;
      load_emb_data = pos_emb_mem[idx];
      @(negedge clk); load_pos_emb = 0;
    end
    #20;

    $display("");
    $display("CONFIG dim=64 ffn=256 vocab=16 layers=2 heads=8");
    $display("");
    // Token 0: id=1
    @(negedge clk);
    token_in = 1; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d", 0, 1, token_out, cycle_count);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d dim=%0d hex=%h", 0, 15, logits_out[240 +: 16]);
      $display("LOGIT_TRUNC pos=%0d dims_remaining=%0d", 0, 48);
    end else begin
      $display("TOKEN pos=%0d id=%0d TIMEOUT", 0, 1);
    end
    repeat(3) @(negedge clk);

    $display("");
    if (token_count > 0)
      $display("SUMMARY total_tokens=%0d total_cycles=%0d avg_cycles=%0d",
               token_count, total_cycles, total_cycles / token_count);
    else
      $display("SUMMARY total_tokens=0 total_cycles=%0d avg_cycles=0", total_cycles);
    $display("DONE");
    $finish;
  end
endmodule
