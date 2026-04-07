`timescale 1ns/1ps
module scaled_cosim_tb;
  parameter VOCAB_SIZE  = 16;
  parameter MAX_SEQ_LEN = 32;
  parameter EMBED_DIM   = 64;
  parameter NUM_HEADS   = 8;
  parameter HEAD_DIM    = 8;
  parameter FFN_DIM     = 256;
  parameter NUM_LAYERS  = 2;
  parameter DATA_WIDTH  = 16;

  reg clk, rst;
  reg valid_in;
  reg [3:0] token_in;
  reg [4:0] position_in;
  reg load_token_emb, load_pos_emb;
  reg [3:0] load_token_idx;
  reg [5:0] load_dim_idx;
  reg signed [15:0] load_emb_data;
  reg [4:0] load_pos_idx;

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
  reg ckpt_capture_en;
  integer ck_dim;

  // Memory arrays for weight loading
  reg [15:0] tok_emb_mem  [0:1023];
  reg [15:0] pos_emb_mem  [0:2047];
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

  // Optional checkpoint emission for WS1 parity harness
  always @(posedge clk) begin
    if (!rst && ckpt_capture_en) begin
      // Capture layer outputs directly on block completion pulse.
      if (dut.block_done_pulse && dut.block_active) begin
        for (ck_dim = 0; ck_dim < EMBED_DIM; ck_dim = ck_dim + 1) begin
          $display("CKPT pos=%0d input_pos=%0d type=layer layer=%0d dim=%0d hex=%h", token_count, position_in, dut.layer_idx, ck_dim, dut.block_out[ck_dim*DATA_WIDTH +: DATA_WIDTH]);
        end
      end

      // Capture final-LN output during OUTPUT state, after final_hidden latches.
      if (dut.state == 4'd5) begin
        for (ck_dim = 0; ck_dim < EMBED_DIM; ck_dim = ck_dim + 1) begin
          $display("CKPT pos=%0d input_pos=%0d type=final_ln layer=%0d dim=%0d hex=%h", token_count, position_in, NUM_LAYERS, ck_dim, dut.final_hidden[ck_dim*DATA_WIDTH +: DATA_WIDTH]);
        end
      end
    end
  end

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
    ckpt_capture_en = 0;

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

    // Startup warmup tokens (7) to stabilize internal state
    // Warmup 0: id=0, input_pos=0
    @(negedge clk);
    token_in = 0; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 1: id=0, input_pos=1
    @(negedge clk);
    token_in = 0; position_in = 1;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 2: id=0, input_pos=2
    @(negedge clk);
    token_in = 0; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 3: id=0, input_pos=3
    @(negedge clk);
    token_in = 0; position_in = 3;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 4: id=0, input_pos=4
    @(negedge clk);
    token_in = 0; position_in = 4;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 5: id=0, input_pos=5
    @(negedge clk);
    token_in = 0; position_in = 5;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    // Warmup 6: id=0, input_pos=6
    @(negedge clk);
    token_in = 0; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    repeat(3) @(negedge clk);

    ckpt_capture_en = 1;

    $display("");
    $display("CONFIG dim=64 ffn=256 vocab=16 layers=2 heads=8");
    $display("");
    // Token 0: id=5, input_pos=2
    @(negedge clk);
    token_in = 5; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 0, 2, 5, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 0, 2, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 0, 2, 5);
    end
    repeat(3) @(negedge clk);

    // Token 1: id=15, input_pos=0
    @(negedge clk);
    token_in = 15; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 1, 0, 15, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 1, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 1, 0, 15);
    end
    repeat(3) @(negedge clk);

    // Token 2: id=7, input_pos=4
    @(negedge clk);
    token_in = 7; position_in = 4;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 2, 4, 7, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 2, 4, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 2, 4, 7);
    end
    repeat(3) @(negedge clk);

    // Token 3: id=4, input_pos=5
    @(negedge clk);
    token_in = 4; position_in = 5;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 3, 5, 4, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 3, 5, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 3, 5, 4);
    end
    repeat(3) @(negedge clk);

    // Token 4: id=13, input_pos=6
    @(negedge clk);
    token_in = 13; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 4, 6, 13, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 4, 6, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 4, 6, 13);
    end
    repeat(3) @(negedge clk);

    // Token 5: id=0, input_pos=5
    @(negedge clk);
    token_in = 0; position_in = 5;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 5, 5, 0, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 5, 5, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 5, 5, 0);
    end
    repeat(3) @(negedge clk);

    // Token 6: id=14, input_pos=6
    @(negedge clk);
    token_in = 14; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 6, 6, 14, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 6, 6, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 6, 6, 14);
    end
    repeat(3) @(negedge clk);

    // Token 7: id=2, input_pos=2
    @(negedge clk);
    token_in = 2; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 7, 2, 2, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 7, 2, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 7, 2, 2);
    end
    repeat(3) @(negedge clk);

    // Token 8: id=2, input_pos=3
    @(negedge clk);
    token_in = 2; position_in = 3;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 8, 3, 2, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 8, 3, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 8, 3, 2);
    end
    repeat(3) @(negedge clk);

    // Token 9: id=13, input_pos=2
    @(negedge clk);
    token_in = 13; position_in = 2;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 9, 2, 13, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 9, 2, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 9, 2, 13);
    end
    repeat(3) @(negedge clk);

    // Token 10: id=10, input_pos=7
    @(negedge clk);
    token_in = 10; position_in = 7;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 10, 7, 10, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 10, 7, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 10, 7, 10);
    end
    repeat(3) @(negedge clk);

    // Token 11: id=9, input_pos=6
    @(negedge clk);
    token_in = 9; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 11, 6, 9, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 11, 6, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 11, 6, 9);
    end
    repeat(3) @(negedge clk);

    // Token 12: id=1, input_pos=0
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
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 12, 0, 1, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 12, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 12, 0, 1);
    end
    repeat(3) @(negedge clk);

    // Token 13: id=7, input_pos=7
    @(negedge clk);
    token_in = 7; position_in = 7;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 13, 7, 7, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 13, 7, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 13, 7, 7);
    end
    repeat(3) @(negedge clk);

    // Token 14: id=5, input_pos=0
    @(negedge clk);
    token_in = 5; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 14, 0, 5, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 14, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 14, 0, 5);
    end
    repeat(3) @(negedge clk);

    // Token 15: id=4, input_pos=0
    @(negedge clk);
    token_in = 4; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 15, 0, 4, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 15, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 15, 0, 4);
    end
    repeat(3) @(negedge clk);

    // Token 16: id=8, input_pos=0
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
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 16, 0, 8, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 16, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 16, 0, 8);
    end
    repeat(3) @(negedge clk);

    // Token 17: id=3, input_pos=0
    @(negedge clk);
    token_in = 3; position_in = 0;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 17, 0, 3, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 17, 0, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 17, 0, 3);
    end
    repeat(3) @(negedge clk);

    // Token 18: id=8, input_pos=6
    @(negedge clk);
    token_in = 8; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 18, 6, 8, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 18, 6, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 18, 6, 8);
    end
    repeat(3) @(negedge clk);

    // Token 19: id=10, input_pos=6
    @(negedge clk);
    token_in = 10; position_in = 6;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 19, 6, 10, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 19, 6, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 19, 6, 10);
    end
    repeat(3) @(negedge clk);

    // Token 20: id=4, input_pos=7
    @(negedge clk);
    token_in = 4; position_in = 7;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 20, 7, 4, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 20, 7, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 20, 7, 4);
    end
    repeat(3) @(negedge clk);

    // Token 21: id=13, input_pos=7
    @(negedge clk);
    token_in = 13; position_in = 7;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 21, 7, 13, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 21, 7, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 21, 7, 13);
    end
    repeat(3) @(negedge clk);

    // Token 22: id=4, input_pos=3
    @(negedge clk);
    token_in = 4; position_in = 3;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 22, 3, 4, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 22, 3, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 22, 3, 4);
    end
    repeat(3) @(negedge clk);

    // Token 23: id=4, input_pos=1
    @(negedge clk);
    token_in = 4; position_in = 1;
    valid_in = 1;
    @(negedge clk); valid_in = 0;
    cycle_count = 0;
    while (!valid_out && cycle_count < 100000) begin
      @(negedge clk); cycle_count = cycle_count + 1;
    end
    if (valid_out) begin
      total_cycles = total_cycles + cycle_count;
      token_count = token_count + 1;
      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", 23, 1, 4, token_out, cycle_count);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 0, logits_out[0 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 1, logits_out[16 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 2, logits_out[32 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 3, logits_out[48 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 4, logits_out[64 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 5, logits_out[80 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 6, logits_out[96 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 7, logits_out[112 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 8, logits_out[128 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 9, logits_out[144 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 10, logits_out[160 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 11, logits_out[176 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 12, logits_out[192 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 13, logits_out[208 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 14, logits_out[224 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 15, logits_out[240 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 16, logits_out[256 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 17, logits_out[272 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 18, logits_out[288 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 19, logits_out[304 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 20, logits_out[320 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 21, logits_out[336 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 22, logits_out[352 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 23, logits_out[368 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 24, logits_out[384 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 25, logits_out[400 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 26, logits_out[416 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 27, logits_out[432 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 28, logits_out[448 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 29, logits_out[464 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 30, logits_out[480 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 31, logits_out[496 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 32, logits_out[512 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 33, logits_out[528 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 34, logits_out[544 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 35, logits_out[560 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 36, logits_out[576 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 37, logits_out[592 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 38, logits_out[608 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 39, logits_out[624 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 40, logits_out[640 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 41, logits_out[656 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 42, logits_out[672 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 43, logits_out[688 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 44, logits_out[704 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 45, logits_out[720 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 46, logits_out[736 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 47, logits_out[752 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 48, logits_out[768 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 49, logits_out[784 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 50, logits_out[800 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 51, logits_out[816 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 52, logits_out[832 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 53, logits_out[848 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 54, logits_out[864 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 55, logits_out[880 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 56, logits_out[896 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 57, logits_out[912 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 58, logits_out[928 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 59, logits_out[944 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 60, logits_out[960 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 61, logits_out[976 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 62, logits_out[992 +: 16]);
      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", 23, 1, 63, logits_out[1008 +: 16]);
    end else begin
      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", 23, 1, 4);
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
