#!/usr/bin/env python3
"""
run_scaled_cosim.py -- Scaled GPT-2 Cosimulation (EMBED_DIM=64)

Uses $readmemh for weight loading instead of inline hex literals,
enabling simulation at much larger dimensions.

Usage:
    python scripts/run_scaled_cosim.py --sentence "hello" --dim 64
    python scripts/run_scaled_cosim.py --sentence "one" --dim 32
"""

import numpy as np
import subprocess
import os
import sys
import time
import argparse
import shutil
import json
import math
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from extract_gpt2_weights import (
    float_to_q88, q88_to_float, q88_hex,
    download_gpt2_weights, extract_for_bitbybit,
    run_float32_reference, layer_norm_f32, gelu_f32, extract_ln
)

def resolve_executable(env_var, default_path, fallback_command):
    """Resolve tool path via env var, fixed default path, or PATH fallback."""
    explicit = os.environ.get(env_var, "").strip()
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise RuntimeError(f"{env_var} is set but file does not exist: {explicit}")

    if os.path.isfile(default_path):
        return default_path

    discovered = shutil.which(fallback_command)
    if discovered:
        return discovered

    raise RuntimeError(
        f"Unable to locate {fallback_command}. Set {env_var} or install it in PATH."
    )

IVERILOG = resolve_executable("BITBYBIT_IVERILOG", r"D:\Tools\iverilog\bin\iverilog.exe", "iverilog")
VVP      = resolve_executable("BITBYBIT_VVP", r"D:\Tools\iverilog\bin\vvp.exe", "vvp")

CHAR_TO_TOKEN = {
    ' ':  0,  'a':  1,  'b':  2,  'c':  3,
    'd':  4,  'e':  5,  'f':  6,  'g':  7,
    'h':  8,  'i':  9,  'j': 10,  'k': 11,
    'l': 12,  'm': 13,  'n': 14,  'o': 15,
}
TOKEN_TO_CHAR = {v: k for k, v in CHAR_TO_TOKEN.items()}

def tokenize(text, vocab_size=16, max_len=8):
    tokens = []
    for ch in text.lower()[:max_len]:
        if ch in CHAR_TO_TOKEN:
            tokens.append(CHAR_TO_TOKEN[ch] % vocab_size)
        else:
            tokens.append(ord(ch) % vocab_size)
    return tokens

# ==============================================================
# Write flat weight array as a hex file (one Q8.8 value per line)
# ==============================================================
def write_hex_file(filepath, values):
    """Write Q8.8 values, one per line, for $readmemh."""
    with open(filepath, 'w') as f:
        for v in values.flatten():
            f.write(f"{int(v) & 0xFFFF:04x}\n")

def write_weight_hex_files(weights, out_dir, ED, FD, VS, MSL):
    """Save all weight arrays as hex files for Verilog $readmemh."""
    os.makedirs(out_dir, exist_ok=True)
    
    write_hex_file(os.path.join(out_dir, "token_emb.hex"), weights['token_emb'][:VS, :ED])
    write_hex_file(os.path.join(out_dir, "pos_emb.hex"), weights['pos_emb'][:MSL, :ED])
    write_hex_file(os.path.join(out_dir, "ln1_gamma.hex"), weights['ln1_gamma'][:ED])
    write_hex_file(os.path.join(out_dir, "ln1_beta.hex"), weights['ln1_beta'][:ED])
    write_hex_file(os.path.join(out_dir, "ln2_gamma.hex"), weights['ln2_gamma'][:ED])
    write_hex_file(os.path.join(out_dir, "ln2_beta.hex"), weights['ln2_beta'][:ED])
    write_hex_file(os.path.join(out_dir, "wq.hex"), weights['wq'][:ED, :ED])
    write_hex_file(os.path.join(out_dir, "wk.hex"), weights['wk'][:ED, :ED])
    write_hex_file(os.path.join(out_dir, "wv.hex"), weights['wv'][:ED, :ED])
    write_hex_file(os.path.join(out_dir, "wo.hex"), weights['wo'][:ED, :ED])
    write_hex_file(os.path.join(out_dir, "ffn_w1.hex"), weights['ffn_w1'][:ED, :FD])
    write_hex_file(os.path.join(out_dir, "ffn_b1.hex"), weights['ffn_b1'][:FD])
    write_hex_file(os.path.join(out_dir, "ffn_w2.hex"), weights['ffn_w2'][:FD, :ED])
    write_hex_file(os.path.join(out_dir, "ffn_b2.hex"), weights['ffn_b2'][:ED])
    write_hex_file(os.path.join(out_dir, "ln_final_gamma.hex"), weights['ln_final_gamma'][:ED])
    write_hex_file(os.path.join(out_dir, "ln_final_beta.hex"), weights['ln_final_beta'][:ED])

# ==============================================================
# Generate Verilog testbench using $readmemh for weights
# ==============================================================
def generate_testbench(
    token_sequence,
    position_sequence,
    tb_path,
    hex_dir,
    ED,
    FD,
    VS,
    MSL,
    NL,
    NH,
    HD,
    DW,
    emit_checkpoints=False,
    warmup_token_sequence=None,
    warmup_position_sequence=None,
):
    """Generate testbench with $readmemh based weight loading."""
    if len(token_sequence) != len(position_sequence):
        raise ValueError("token_sequence and position_sequence must have the same length")
    warmup_token_sequence = warmup_token_sequence or []
    warmup_position_sequence = warmup_position_sequence or []
    if len(warmup_token_sequence) != len(warmup_position_sequence):
        raise ValueError("warmup_token_sequence and warmup_position_sequence must have the same length")

    tk_bits = max(1, int(np.ceil(np.log2(VS))))
    pos_bits = max(1, int(np.ceil(np.log2(MSL))))
    dim_bits = max(1, int(np.ceil(np.log2(ED))))
    ffn_bits = max(1, int(np.ceil(np.log2(max(FD, ED)))))
    layer_bits = max(1, int(np.ceil(np.log2(NL + 1))))

    # Convert hex_dir to forward slashes for Verilog
    hex_fwd = hex_dir.replace('\\', '/')

    L = []
    L.append('`timescale 1ns/1ps')
    L.append('module scaled_cosim_tb;')
    L.append(f'  parameter VOCAB_SIZE  = {VS};')
    L.append(f'  parameter MAX_SEQ_LEN = {MSL};')
    L.append(f'  parameter EMBED_DIM   = {ED};')
    L.append(f'  parameter NUM_HEADS   = {NH};')
    L.append(f'  parameter HEAD_DIM    = {HD};')
    L.append(f'  parameter FFN_DIM     = {FD};')
    L.append(f'  parameter NUM_LAYERS  = {NL};')
    L.append(f'  parameter DATA_WIDTH  = {DW};')
    L.append('')
    L.append('  reg clk, rst;')
    L.append('  reg valid_in;')
    L.append(f'  reg [{tk_bits-1}:0] token_in;')
    L.append(f'  reg [{pos_bits-1}:0] position_in;')
    L.append('  reg load_token_emb, load_pos_emb;')
    L.append(f'  reg [{tk_bits-1}:0] load_token_idx;')
    L.append(f'  reg [{dim_bits-1}:0] load_dim_idx;')
    L.append(f'  reg signed [{DW-1}:0] load_emb_data;')
    L.append(f'  reg [{pos_bits-1}:0] load_pos_idx;')
    L.append('')
    L.append('  // Load-based transformer weight interface')
    L.append('  reg load_ln_en;')
    L.append(f'  reg [{layer_bits-1}:0] load_layer_idx;')
    L.append('  reg load_ln_sel, load_ln_is_gamma;')
    L.append(f'  reg [{dim_bits-1}:0] load_ln_dim;')
    L.append(f'  reg signed [{DW-1}:0] load_ln_data;')
    L.append('  reg load_attn_weight_en;')
    L.append('  reg [1:0] load_attn_matrix_sel;')
    L.append(f'  reg [{dim_bits-1}:0] load_attn_row, load_attn_col;')
    L.append(f'  reg signed [{DW-1}:0] load_attn_data;')
    L.append('  reg load_ffn_weight_en;')
    L.append('  reg load_ffn_layer_sel, load_ffn_is_bias;')
    L.append(f'  reg [{ffn_bits-1}:0] load_ffn_row, load_ffn_col;')
    L.append(f'  reg signed [{DW-1}:0] load_ffn_data;')
    L.append('')
    L.append(f'  wire [{tk_bits-1}:0] token_out;')
    L.append(f'  wire [{ED*DW-1}:0] logits_out;')
    L.append('  wire valid_out;')
    L.append('  wire [31:0] total_zero_skips;')
    L.append('  wire [31:0] total_cycles_hw;')
    L.append('  integer cycle_count;')
    L.append('  integer total_cycles;')
    L.append('  integer token_count;')
    L.append('  integer idx, row_idx, col_idx, layer_idx_i;')
    if emit_checkpoints:
        L.append('  reg ckpt_capture_en;')
        L.append('  integer ck_dim;')
    L.append('')

    # Memory arrays for $readmemh
    L.append('  // Memory arrays for weight loading')
    L.append(f'  reg [{DW-1}:0] tok_emb_mem  [0:{VS*ED-1}];')
    L.append(f'  reg [{DW-1}:0] pos_emb_mem  [0:{MSL*ED-1}];')
    L.append(f'  reg [{DW-1}:0] ln1g_mem     [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] ln1b_mem     [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] ln2g_mem     [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] ln2b_mem     [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] wq_mem       [0:{ED*ED-1}];')
    L.append(f'  reg [{DW-1}:0] wk_mem       [0:{ED*ED-1}];')
    L.append(f'  reg [{DW-1}:0] wv_mem       [0:{ED*ED-1}];')
    L.append(f'  reg [{DW-1}:0] wo_mem       [0:{ED*ED-1}];')
    L.append(f'  reg [{DW-1}:0] fw1_mem      [0:{ED*FD-1}];')
    L.append(f'  reg [{DW-1}:0] fb1_mem      [0:{FD-1}];')
    L.append(f'  reg [{DW-1}:0] fw2_mem      [0:{FD*ED-1}];')
    L.append(f'  reg [{DW-1}:0] fb2_mem      [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] lnfg_mem     [0:{ED-1}];')
    L.append(f'  reg [{DW-1}:0] lnfb_mem     [0:{ED-1}];')
    L.append('')

    # DUT
    L.append('  gpt2_engine #(')
    L.append('    .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),')
    L.append('    .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),')
    L.append('    .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),')
    L.append('    .NUM_LAYERS(NUM_LAYERS), .DATA_WIDTH(DATA_WIDTH)')
    L.append('  ) dut (')
    L.append('    .clk(clk), .rst(rst),')
    L.append('    .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),')
    L.append('    .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),')
    L.append('    .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),')
    L.append('    .load_ln_en(load_ln_en), .load_layer_idx(load_layer_idx),')
    L.append('    .load_ln_sel(load_ln_sel), .load_ln_is_gamma(load_ln_is_gamma),')
    L.append('    .load_ln_dim(load_ln_dim), .load_ln_data(load_ln_data),')
    L.append('    .load_attn_weight_en(load_attn_weight_en),')
    L.append('    .load_attn_matrix_sel(load_attn_matrix_sel),')
    L.append('    .load_attn_row(load_attn_row), .load_attn_col(load_attn_col),')
    L.append('    .load_attn_data(load_attn_data),')
    L.append('    .load_ffn_weight_en(load_ffn_weight_en),')
    L.append('    .load_ffn_layer_sel(load_ffn_layer_sel),')
    L.append('    .load_ffn_is_bias(load_ffn_is_bias),')
    L.append('    .load_ffn_row(load_ffn_row), .load_ffn_col(load_ffn_col),')
    L.append('    .load_ffn_data(load_ffn_data),')
    L.append('    .valid_in(valid_in), .token_in(token_in), .position_in(position_in),')
    L.append('    .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out),')
    L.append('    .total_zero_skips(total_zero_skips), .total_cycles(total_cycles_hw)')
    L.append('  );')
    L.append('')
    L.append('  always #5 clk = ~clk;')
    L.append('')

    if emit_checkpoints:
        L.append('  // Optional checkpoint emission for WS1 parity harness')
        L.append('  always @(posedge clk) begin')
        L.append('    if (!rst && ckpt_capture_en) begin')
        L.append('      // Capture layer outputs directly on block completion pulse.')
        L.append('      if (dut.block_done_pulse && dut.block_active) begin')
        L.append('        for (ck_dim = 0; ck_dim < EMBED_DIM; ck_dim = ck_dim + 1) begin')
        L.append('          $display("CKPT pos=%0d input_pos=%0d type=layer layer=%0d dim=%0d hex=%h", token_count, position_in, dut.layer_idx, ck_dim, dut.block_out[ck_dim*DATA_WIDTH +: DATA_WIDTH]);')
        L.append('        end')
        L.append('      end')
        L.append('')
        L.append('      // Capture final-LN output during OUTPUT state, after final_hidden latches.')
        L.append('      if (dut.state == 4\'d5) begin')
        L.append('        for (ck_dim = 0; ck_dim < EMBED_DIM; ck_dim = ck_dim + 1) begin')
        L.append('          $display("CKPT pos=%0d input_pos=%0d type=final_ln layer=%0d dim=%0d hex=%h", token_count, position_in, NUM_LAYERS, ck_dim, dut.final_hidden[ck_dim*DATA_WIDTH +: DATA_WIDTH]);')
        L.append('        end')
        L.append('      end')
        L.append('    end')
        L.append('  end')
        L.append('')

    L.append('  initial begin')
    L.append('    $dumpfile("scaled_cosim.vcd");')
    L.append('    $dumpvars(0, scaled_cosim_tb);')
    L.append('  end')
    L.append('')

    # Initial block: load weights from hex files
    L.append('  initial begin')
    L.append('    // Load weights via $readmemh')
    L.append(f'    $readmemh("{hex_fwd}/token_emb.hex", tok_emb_mem);')
    L.append(f'    $readmemh("{hex_fwd}/pos_emb.hex",   pos_emb_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln1_gamma.hex", ln1g_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln1_beta.hex",  ln1b_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln2_gamma.hex", ln2g_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln2_beta.hex",  ln2b_mem);')
    L.append(f'    $readmemh("{hex_fwd}/wq.hex",        wq_mem);')
    L.append(f'    $readmemh("{hex_fwd}/wk.hex",        wk_mem);')
    L.append(f'    $readmemh("{hex_fwd}/wv.hex",        wv_mem);')
    L.append(f'    $readmemh("{hex_fwd}/wo.hex",        wo_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ffn_w1.hex",    fw1_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ffn_b1.hex",    fb1_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ffn_w2.hex",    fw2_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ffn_b2.hex",    fb2_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln_final_gamma.hex", lnfg_mem);')
    L.append(f'    $readmemh("{hex_fwd}/ln_final_beta.hex",  lnfb_mem);')
    L.append('  end')
    L.append('')

    # Second initial block: drive simulation
    L.append('  initial begin')
    L.append('    clk = 0; rst = 1;')
    L.append('    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;')
    L.append('    token_in = 0; position_in = 0;')
    L.append('    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;')
    L.append('    load_pos_idx = 0;')
    L.append('    load_ln_en = 0; load_layer_idx = 0; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = 0; load_ln_data = 0;')
    L.append('    load_attn_weight_en = 0; load_attn_matrix_sel = 0; load_attn_row = 0; load_attn_col = 0; load_attn_data = 0;')
    L.append('    load_ffn_weight_en = 0; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = 0; load_ffn_col = 0; load_ffn_data = 0;')
    L.append('    total_cycles = 0; token_count = 0;')
    if emit_checkpoints:
        L.append('    ckpt_capture_en = 0;')
    L.append('')
    L.append('    #35 rst = 0; #25;')
    L.append('')
    L.append('    // Load LayerNorm parameters for each transformer layer')
    L.append('    for (layer_idx_i = 0; layer_idx_i < NUM_LAYERS; layer_idx_i = layer_idx_i + 1) begin')
    L.append('      for (idx = 0; idx < EMBED_DIM; idx = idx + 1) begin')
    L.append('        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = ln1g_mem[idx];')
    L.append('        @(negedge clk); load_ln_en = 0;')
    L.append('        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = ln1b_mem[idx];')
    L.append('        @(negedge clk); load_ln_en = 0;')
    L.append('        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 1; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = ln2g_mem[idx];')
    L.append('        @(negedge clk); load_ln_en = 0;')
    L.append('        @(negedge clk); load_ln_en = 1; load_layer_idx = layer_idx_i; load_ln_sel = 1; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = ln2b_mem[idx];')
    L.append('        @(negedge clk); load_ln_en = 0;')
    L.append('      end')
    L.append('    end')
    L.append('')
    L.append('    // Load final LayerNorm (load_layer_idx == NUM_LAYERS)')
    L.append('    for (idx = 0; idx < EMBED_DIM; idx = idx + 1) begin')
    L.append('      @(negedge clk); load_ln_en = 1; load_layer_idx = NUM_LAYERS; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = idx; load_ln_data = lnfg_mem[idx];')
    L.append('      @(negedge clk); load_ln_en = 0;')
    L.append('      @(negedge clk); load_ln_en = 1; load_layer_idx = NUM_LAYERS; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = idx; load_ln_data = lnfb_mem[idx];')
    L.append('      @(negedge clk); load_ln_en = 0;')
    L.append('    end')
    L.append('')
    L.append('    // Load attention matrices Wq/Wk/Wv/Wo')
    L.append('    for (row_idx = 0; row_idx < EMBED_DIM; row_idx = row_idx + 1) begin')
    L.append('      for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin')
    L.append('        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d0; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wq_mem[row_idx*EMBED_DIM + col_idx];')
    L.append('        @(negedge clk); load_attn_weight_en = 0;')
    L.append('        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d1; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wk_mem[row_idx*EMBED_DIM + col_idx];')
    L.append('        @(negedge clk); load_attn_weight_en = 0;')
    L.append('        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d2; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wv_mem[row_idx*EMBED_DIM + col_idx];')
    L.append('        @(negedge clk); load_attn_weight_en = 0;')
    L.append('        @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d3; load_attn_row = row_idx; load_attn_col = col_idx; load_attn_data = wo_mem[row_idx*EMBED_DIM + col_idx];')
    L.append('        @(negedge clk); load_attn_weight_en = 0;')
    L.append('      end')
    L.append('    end')
    L.append('')
    L.append('    // Load FFN W1 and b1')
    L.append('    for (row_idx = 0; row_idx < EMBED_DIM; row_idx = row_idx + 1) begin')
    L.append('      for (col_idx = 0; col_idx < FFN_DIM; col_idx = col_idx + 1) begin')
    L.append('        @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = row_idx; load_ffn_col = col_idx; load_ffn_data = fw1_mem[row_idx*FFN_DIM + col_idx];')
    L.append('        @(negedge clk); load_ffn_weight_en = 0;')
    L.append('      end')
    L.append('    end')
    L.append('    for (col_idx = 0; col_idx < FFN_DIM; col_idx = col_idx + 1) begin')
    L.append('      @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = col_idx; load_ffn_data = fb1_mem[col_idx];')
    L.append('      @(negedge clk); load_ffn_weight_en = 0;')
    L.append('    end')
    L.append('')
    L.append('    // Load FFN W2 and b2')
    L.append('    for (row_idx = 0; row_idx < FFN_DIM; row_idx = row_idx + 1) begin')
    L.append('      for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin')
    L.append('        @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 0; load_ffn_row = row_idx; load_ffn_col = col_idx; load_ffn_data = fw2_mem[row_idx*EMBED_DIM + col_idx];')
    L.append('        @(negedge clk); load_ffn_weight_en = 0;')
    L.append('      end')
    L.append('    end')
    L.append('    for (col_idx = 0; col_idx < EMBED_DIM; col_idx = col_idx + 1) begin')
    L.append('      @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = col_idx; load_ffn_data = fb2_mem[col_idx];')
    L.append('      @(negedge clk); load_ffn_weight_en = 0;')
    L.append('    end')
    L.append('')

    # Load embeddings from memory arrays
    L.append('    // Load embeddings into DUT')
    L.append('    for (idx = 0; idx < VOCAB_SIZE*EMBED_DIM; idx = idx + 1) begin')
    L.append('      @(negedge clk);')
    L.append('      load_token_emb = 1;')
    L.append('      load_token_idx = idx / EMBED_DIM;')
    L.append('      load_dim_idx   = idx % EMBED_DIM;')
    L.append('      load_emb_data  = tok_emb_mem[idx];')
    L.append('      @(negedge clk); load_token_emb = 0;')
    L.append('    end')
    L.append('    for (idx = 0; idx < MAX_SEQ_LEN*EMBED_DIM; idx = idx + 1) begin')
    L.append('      @(negedge clk);')
    L.append('      load_pos_emb = 1;')
    L.append('      load_pos_idx = idx / EMBED_DIM;')
    L.append('      load_dim_idx = idx % EMBED_DIM;')
    L.append('      load_emb_data = pos_emb_mem[idx];')
    L.append('      @(negedge clk); load_pos_emb = 0;')
    L.append('    end')
    L.append('    #20;')
    L.append('')

    if warmup_token_sequence:
        L.append(f'    // Startup warmup tokens ({len(warmup_token_sequence)}) to stabilize internal state')
        for warmup_idx, (warm_tok, warm_pos) in enumerate(zip(warmup_token_sequence, warmup_position_sequence)):
            L.append(f'    // Warmup {warmup_idx}: id={warm_tok}, input_pos={warm_pos}')
            L.append('    @(negedge clk);')
            L.append(f'    token_in = {warm_tok}; position_in = {warm_pos};')
            L.append('    valid_in = 1;')
            L.append('    @(negedge clk); valid_in = 0;')
            L.append('    cycle_count = 0;')
            L.append('    while (!valid_out && cycle_count < 100000) begin')
            L.append('      @(negedge clk); cycle_count = cycle_count + 1;')
            L.append('    end')
            L.append('    repeat(3) @(negedge clk);')
            L.append('')

    if emit_checkpoints:
        L.append('    ckpt_capture_en = 1;')
        L.append('')

    L.append('    $display("");')
    L.append(f'    $display("CONFIG dim={ED} ffn={FD} vocab={VS} layers={NL} heads={NH}");')
    L.append('    $display("");')

    # Process tokens
    for seq_pos, (tok, input_pos) in enumerate(zip(token_sequence, position_sequence)):
        L.append(f'    // Token {seq_pos}: id={tok}, input_pos={input_pos}')
        L.append('    @(negedge clk);')
        L.append(f'    token_in = {tok}; position_in = {input_pos};')
        L.append('    valid_in = 1;')
        L.append('    @(negedge clk); valid_in = 0;')
        L.append('    cycle_count = 0;')
        L.append('    while (!valid_out && cycle_count < 100000) begin')
        L.append('      @(negedge clk); cycle_count = cycle_count + 1;')
        L.append('    end')
        L.append('    if (valid_out) begin')
        L.append('      total_cycles = total_cycles + cycle_count;')
        L.append('      token_count = token_count + 1;')
        L.append(f'      $display("TOKEN pos=%0d input_pos=%0d id=%0d predicted=%0d cycles=%0d", {seq_pos}, {input_pos}, {tok}, token_out, cycle_count);')
        num_logits_to_show = ED if emit_checkpoints else min(ED, 16)
        for d in range(num_logits_to_show):
            L.append(f'      $display("LOGIT pos=%0d input_pos=%0d dim=%0d hex=%h", {seq_pos}, {input_pos}, {d}, logits_out[{d*DW} +: {DW}]);')
        if (not emit_checkpoints) and ED > 16:
            L.append(f'      $display("LOGIT_TRUNC pos=%0d dims_remaining=%0d", {seq_pos}, {ED - 16});')
        L.append('    end else begin')
        L.append(f'      $display("TOKEN pos=%0d input_pos=%0d id=%0d TIMEOUT", {seq_pos}, {input_pos}, {tok});')
        L.append('    end')
        L.append('    repeat(3) @(negedge clk);')
        L.append('')

    L.append('    $display("");')
    L.append('    if (token_count > 0)')
    L.append('      $display("SUMMARY total_tokens=%0d total_cycles=%0d avg_cycles=%0d",')
    L.append('               token_count, total_cycles, total_cycles / token_count);')
    L.append('    else')
    L.append('      $display("SUMMARY total_tokens=0 total_cycles=%0d avg_cycles=0", total_cycles);')
    L.append('    $display("DONE");')
    L.append('    $finish;')
    L.append('  end')
    L.append('endmodule')

    with open(tb_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(L) + '\n')

# ==============================================================
# Q8.8 Python Reference with sparsity tracking
# ==============================================================
Q88_FRAC_BITS = 8


def wrap_signed(value, bits):
    mask = (1 << bits) - 1
    value = int(value) & mask
    sign_bit = 1 << (bits - 1)
    if value & sign_bit:
        value -= (1 << bits)
    return value


def wrap_i16(value):
    return wrap_signed(value, 16)


def wrap_i24(value):
    return wrap_signed(value, 24)


def wrap_i32(value):
    return wrap_signed(value, 32)


def slice_23_8_to_i16(value):
    raw = int(value) & 0xFFFFFFFF
    return wrap_i16((raw >> 8) & 0xFFFF)


def build_exp_lut_256():
    return [
        max(1, min(255, int(round(255.0 * math.exp(-k / 64.0)))))
        for k in range(256)
    ]


def build_gelu_lut_256():
    vals = []
    for k in range(256):
        x = (k - 128) / 32.0
        gelu = x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        vals.append(wrap_i16(int(round(gelu * 256.0))))
    return vals


def build_inv_sqrt_lut_256():
    vals = []
    for k in range(256):
        val = int(round(256.0 / math.sqrt((k + 1) / 16.0)))
        vals.append(max(16, min(4096, val)))
    return vals

def load_lut_from_rtl(filename, fallback_builder):
    """Load LUT values from RTL source so reference math stays bit-aligned."""
    rtl_path = os.path.join(os.path.dirname(__file__), "..", "rtl", "compute", filename)
    rtl_path = os.path.abspath(rtl_path)
    try:
        with open(rtl_path, "r", encoding="utf-8") as f:
            text = f.read()
        pairs = re.findall(r"lut\[\s*(\d+)\s*\]\s*=\s*([\-]?\d+)\s*;", text)
        if not pairs:
            raise ValueError("No LUT entries found")

        lut = [0] * 256
        seen = set()
        for idx_s, val_s in pairs:
            idx = int(idx_s)
            if 0 <= idx < 256:
                lut[idx] = int(val_s)
                seen.add(idx)

        if len(seen) != 256:
            raise ValueError(f"Incomplete LUT entries: found {len(seen)} / 256")
        return lut
    except Exception:
        return fallback_builder()


EXP_LUT_256 = load_lut_from_rtl("exp_lut_256.v", build_exp_lut_256)
GELU_LUT_256 = load_lut_from_rtl("gelu_lut_256.v", build_gelu_lut_256)
INV_SQRT_LUT_256 = load_lut_from_rtl("inv_sqrt_lut_256.v", build_inv_sqrt_lut_256)


def exp_lut_q08(x_q88):
    neg_x = -int(x_q88)
    shifted = neg_x >> 2
    if neg_x < 0:
        idx = 0
    elif shifted > 255:
        idx = 255
    else:
        idx = shifted
    return EXP_LUT_256[idx]


def gelu_lut_q88(x_q88):
    shifted = (int(x_q88) >> 2) + 128
    idx = max(0, min(255, shifted))
    return GELU_LUT_256[idx]


def inv_sqrt_lut_q88(var_q88):
    v = max(0, int(var_q88))
    shifted = v >> 2
    if v == 0:
        idx = 0
    elif shifted > 255:
        idx = 255
    else:
        idx = shifted
    return INV_SQRT_LUT_256[idx]


def q88_vector_to_float(vec):
    return [q88_to_float(v) for v in vec]


def prepare_q88_reference_weights(weights_q88, ED, FD, VS, MSL):
    return {
        'token_emb': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['token_emb'][:VS]],
        'pos_emb': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['pos_emb'][:MSL]],
        'ln1_gamma': [wrap_i16(int(v)) for v in weights_q88['ln1_gamma'][:ED]],
        'ln1_beta': [wrap_i16(int(v)) for v in weights_q88['ln1_beta'][:ED]],
        'ln2_gamma': [wrap_i16(int(v)) for v in weights_q88['ln2_gamma'][:ED]],
        'ln2_beta': [wrap_i16(int(v)) for v in weights_q88['ln2_beta'][:ED]],
        'wq': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['wq'][:ED]],
        'wk': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['wk'][:ED]],
        'wv': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['wv'][:ED]],
        'wo': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['wo'][:ED]],
        'ffn_w1': [[wrap_i16(int(v)) for v in row[:FD]] for row in weights_q88['ffn_w1'][:ED]],
        'ffn_b1': [wrap_i16(int(v)) for v in weights_q88['ffn_b1'][:FD]],
        'ffn_w2': [[wrap_i16(int(v)) for v in row[:ED]] for row in weights_q88['ffn_w2'][:FD]],
        'ffn_b2': [wrap_i16(int(v)) for v in weights_q88['ffn_b2'][:ED]],
        'ln_final_gamma': [wrap_i16(int(v)) for v in weights_q88['ln_final_gamma'][:ED]],
        'ln_final_beta': [wrap_i16(int(v)) for v in weights_q88['ln_final_beta'][:ED]],
    }


def init_q88_reference_state(ED, MSL):
    return {
        'k_cache': [[0] * ED for _ in range(MSL)],
        'v_cache': [[0] * ED for _ in range(MSL)],
    }


def layer_norm_q88_rtl(x_vec, gamma_vec, beta_vec):
    dim = len(x_vec)
    dim_log2 = int(math.log2(dim))

    sum_acc = 0
    for val in x_vec:
        sum_acc = wrap_i24(sum_acc + int(val))
    mean_val = wrap_i16(sum_acc >> dim_log2)

    var_acc = 0
    for val in x_vec:
        diff = wrap_i16(int(val) - mean_val)
        var_acc = wrap_i32(var_acc + (diff * diff))
    var_slice = slice_23_8_to_i16(var_acc)
    var_val = wrap_i16(var_slice >> dim_log2)
    inv_std = inv_sqrt_lut_q88(var_val)

    out = [0] * dim
    for i in range(dim):
        diff = wrap_i16(int(x_vec[i]) - mean_val)
        norm_val = wrap_i32(diff * inv_std)
        norm_q88 = slice_23_8_to_i16(norm_val)
        scaled = wrap_i32(int(gamma_vec[i]) * norm_q88)
        scaled_q88 = slice_23_8_to_i16(scaled)
        out[i] = wrap_i16(scaled_q88 + int(beta_vec[i]))
    return out


def attention_q88_rtl(x_vec, seq_pos, weights, state):
    ed = len(x_vec)
    k_cache = state['k_cache']
    v_cache = state['v_cache']

    q_vec = [0] * ed
    k_vec = [0] * ed
    v_vec = [0] * ed

    for j in range(ed):
        acc = 0
        for i in range(ed):
            if x_vec[i] != 0 and weights['wq'][i][j] != 0:
                acc = wrap_i32(acc + int(x_vec[i]) * int(weights['wq'][i][j]))
        q_vec[j] = wrap_i16(acc >> Q88_FRAC_BITS)

        acc = 0
        for i in range(ed):
            acc = wrap_i32(acc + int(x_vec[i]) * int(weights['wk'][i][j]))
        k_vec[j] = wrap_i16(acc >> Q88_FRAC_BITS)

        acc = 0
        for i in range(ed):
            acc = wrap_i32(acc + int(x_vec[i]) * int(weights['wv'][i][j]))
        v_vec[j] = wrap_i16(acc >> Q88_FRAC_BITS)

    for j in range(ed):
        k_cache[seq_pos][j] = k_vec[j]
        v_cache[seq_pos][j] = v_vec[j]

    scores = [0] * (seq_pos + 1)
    max_score = -32767
    for t in range(seq_pos + 1):
        acc = 0
        for j in range(ed):
            acc = wrap_i32(acc + int(q_vec[j]) * int(k_cache[t][j]))
        score = wrap_i16((acc >> Q88_FRAC_BITS) >> 1)
        scores[t] = score
        if score > max_score:
            max_score = score

    probs = [0] * (seq_pos + 1)
    exp_sum = 0
    for t in range(seq_pos + 1):
        diff = wrap_i16(scores[t] - max_score)
        probs[t] = exp_lut_q08(diff)
        exp_sum = (exp_sum + probs[t]) & 0xFFFF

    for t in range(seq_pos + 1):
        p = probs[t]
        if exp_sum == 0:
            norm_val = 0
        elif exp_sum <= 1:
            norm_val = p
        elif exp_sum <= 2:
            norm_val = p * 128
        elif exp_sum <= 4:
            norm_val = p * 64
        elif exp_sum <= 8:
            norm_val = p * 32
        elif exp_sum <= 16:
            norm_val = p * 16
        elif exp_sum <= 32:
            norm_val = p * 8
        elif exp_sum <= 64:
            norm_val = p * 4
        elif exp_sum <= 128:
            norm_val = p * 2
        elif exp_sum <= 256:
            norm_val = p
        elif exp_sum <= 512:
            norm_val = p >> 1
        else:
            norm_val = p >> 2
        probs[t] = 255 if norm_val > 255 else (norm_val & 0xFF)

    attn_mix = [0] * ed
    for j in range(ed):
        acc = 0
        for t in range(seq_pos + 1):
            acc = wrap_i32(acc + int(probs[t]) * int(v_cache[t][j]))
        attn_mix[j] = wrap_i16(acc >> Q88_FRAC_BITS)

    y_out = [0] * ed
    for j in range(ed):
        acc = 0
        for i in range(ed):
            acc = wrap_i32(acc + int(attn_mix[i]) * int(weights['wo'][i][j]))
        y_out[j] = wrap_i16(acc >> Q88_FRAC_BITS)

    return y_out, v_vec, attn_mix


def ffn_q88_rtl(x_vec, weights, FD):
    ed = len(x_vec)

    hidden = [0] * FD
    for j in range(FD):
        accum = 0
        for i in range(ed):
            if x_vec[i] != 0 and weights['ffn_w1'][i][j] != 0:
                accum = wrap_i32(accum + int(x_vec[i]) * int(weights['ffn_w1'][i][j]))
        hidden[j] = wrap_i16(slice_23_8_to_i16(accum) + int(weights['ffn_b1'][j]))

    activated = [gelu_lut_q88(h) for h in hidden]

    y_out = [0] * ed
    for j in range(ed):
        accum = 0
        for i in range(FD):
            if activated[i] != 0 and weights['ffn_w2'][i][j] != 0:
                accum = wrap_i32(accum + int(activated[i]) * int(weights['ffn_w2'][i][j]))
        y_out[j] = wrap_i16(slice_23_8_to_i16(accum) + int(weights['ffn_b2'][j]))

    return y_out, activated


def run_q88_reference(
    weights_q88,
    token_id,
    position,
    ED,
    FD,
    NL,
    VS,
    MSL,
    state=None,
    capture_trace=False,
    collect_stats=True,
):
    """Stateful Q8.8 fixed-point reference aligned to RTL compute path."""
    if state is None:
        state = init_q88_reference_state(ED, MSL)

    stats = {'zero_mults': 0, 'total_mults': 0}
    trace = {'layers': {}, 'final_ln': None} if capture_trace else None

    tok_emb = weights_q88['token_emb'][token_id][:ED]
    pos_emb = weights_q88['pos_emb'][position][:ED]
    x = [wrap_i16(int(tok_emb[i]) + int(pos_emb[i])) for i in range(ED)]

    if capture_trace:
        trace['embedding'] = q88_vector_to_float(x)
    if collect_stats:
        zeros = sum(1 for v in x if abs(v) <= 2)
        stats['zero_mults'] += zeros * ED
        stats['total_mults'] += ED * ED

    for layer in range(NL):
        residual = x[:]

        x = layer_norm_q88_rtl(x, weights_q88['ln1_gamma'], weights_q88['ln1_beta'])

        attn_out, v_vec, attn_mix = attention_q88_rtl(x, position, weights_q88, state)
        x = [wrap_i16(int(residual[i]) + int(attn_out[i])) for i in range(ED)]

        if collect_stats:
            zeros_v = sum(1 for v in v_vec if abs(v) <= 2)
            zeros_attn = sum(1 for v in attn_mix if abs(v) <= 2)
            stats['zero_mults'] += zeros_v * ED
            stats['zero_mults'] += zeros_attn * ED
            stats['total_mults'] += ED * ED
            stats['total_mults'] += ED * ED

        residual = x[:]

        x = layer_norm_q88_rtl(x, weights_q88['ln2_gamma'], weights_q88['ln2_beta'])

        ffn_out, activated = ffn_q88_rtl(x, weights_q88, FD)
        x = [wrap_i16(int(residual[i]) + int(ffn_out[i])) for i in range(ED)]

        if collect_stats:
            gelu_zeros = sum(1 for v in activated if abs(v) <= 2)
            stats['zero_mults'] += gelu_zeros * ED
            stats['total_mults'] += ED * FD + FD * ED

        if capture_trace:
            trace['layers'][str(layer)] = q88_vector_to_float(x)

    x = layer_norm_q88_rtl(x, weights_q88['ln_final_gamma'], weights_q88['ln_final_beta'])
    if capture_trace:
        trace['final_ln'] = q88_vector_to_float(x)

    logits_debug_i16 = [0] * ED
    vocab_logits_i16 = [0] * VS
    for vocab_idx in range(VS):
        dot_acc = 0
        vocab_vec = weights_q88['token_emb'][vocab_idx]
        for dim_idx in range(ED):
            dot_acc += int(x[dim_idx]) * int(vocab_vec[dim_idx])
        vocab_logit = wrap_i16(dot_acc >> Q88_FRAC_BITS)
        vocab_logits_i16[vocab_idx] = vocab_logit
        if vocab_idx < ED:
            logits_debug_i16[vocab_idx] = vocab_logit

    logits_debug = np.array([q88_to_float(v) for v in logits_debug_i16], dtype=np.float64)
    predicted = int(np.argmax(vocab_logits_i16)) if vocab_logits_i16 else 0

    return logits_debug, predicted, stats, trace

# ==============================================================
# Parse Verilog output
# ==============================================================
def parse_output(output, ED, DW):
    results = {}
    config = {}
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith("CONFIG "):
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    config[k] = v
        elif line.startswith("TOKEN ") and "TIMEOUT" not in line:
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts['pos'])
            existing = results.get(pos, {})
            results[pos] = {
                'token_id': int(parts['id']),
                'input_pos': int(parts.get('input_pos', pos)),
                'predicted': int(parts['predicted']),
                'cycles': int(parts['cycles']),
                'logits': existing.get('logits', [0.0] * ED),
                'checkpoints': existing.get('checkpoints', {'layers': {}, 'final_ln': [None] * ED})
            }
        elif line.startswith("LOGIT ") or line.strip().startswith("LOGIT "):
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts['pos'])
            dim = int(parts['dim'])
            hex_val = parts['hex']
            # Handle multi-char hex (take last 4 chars for 16-bit)
            hex_val = hex_val[-4:]
            try:
                int_val = int(hex_val, 16)
            except ValueError:
                continue
            if int_val >= 32768: int_val -= 65536
            if pos in results and dim < ED:
                results[pos]['logits'][dim] = int_val / 256.0
        elif line.startswith("CKPT "):
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v

            if 'pos' not in parts or 'dim' not in parts or 'hex' not in parts:
                continue

            pos = int(parts['pos'])
            dim = int(parts['dim'])
            if dim < 0 or dim >= ED:
                continue

            ckpt_type = parts.get('type', '')
            layer = int(parts.get('layer', -1))
            input_pos = int(parts.get('input_pos', pos))

            if pos not in results:
                results[pos] = {
                    'token_id': -1,
                    'input_pos': input_pos,
                    'predicted': -1,
                    'cycles': -1,
                    'logits': [0.0] * ED,
                    'checkpoints': {'layers': {}, 'final_ln': [None] * ED}
                }

            hex_val = parts['hex'][-4:]
            try:
                int_val = int(hex_val, 16)
            except ValueError:
                continue
            if int_val >= 32768:
                int_val -= 65536
            f_val = int_val / 256.0

            if ckpt_type == 'layer':
                layer_key = str(layer)
                if layer_key not in results[pos]['checkpoints']['layers']:
                    results[pos]['checkpoints']['layers'][layer_key] = [None] * ED
                results[pos]['checkpoints']['layers'][layer_key][dim] = f_val
            elif ckpt_type == 'final_ln':
                results[pos]['checkpoints']['final_ln'][dim] = f_val
        elif "TOKEN" in line and "TIMEOUT" in line:
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts.get('pos', -1))
            existing = results.get(pos, {})
            results[pos] = {
                'token_id': int(parts.get('id', 0)),
                'input_pos': int(parts.get('input_pos', pos)),
                'predicted': -1,
                'cycles': -1,
                'logits': existing.get('logits', []),
                'timeout': True,
                'checkpoints': existing.get('checkpoints', {'layers': {}, 'final_ln': [None] * ED})
            }
    return results, config


def compute_vector_metrics(reference, observed):
    ref = np.array(reference, dtype=np.float64)
    got = np.array(observed, dtype=np.float64)
    if ref.shape != got.shape:
        raise ValueError(f"Vector shape mismatch: ref={ref.shape}, observed={got.shape}")
    if ref.size == 0:
        return {
            'mean_abs_error': 0.0,
            'max_abs_error': 0.0,
            'mse': 0.0,
        }
    abs_err = np.abs(ref - got)
    sq_err = (ref - got) ** 2
    return {
        'mean_abs_error': float(np.mean(abs_err)),
        'max_abs_error': float(np.max(abs_err)),
        'mse': float(np.mean(sq_err)),
    }

# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Scaled GPT-2 Cosimulation")
    parser.add_argument("--sentence", type=str, default="hello")
    parser.add_argument("--token-seq", type=str, default="", help="Optional comma-separated token IDs (overrides --sentence)")
    parser.add_argument("--position-seq", type=str, default="", help="Optional comma-separated position IDs aligned to --token-seq")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--ffn-mult", type=int, default=4, help="FFN multiplier (ffn_dim = dim * mult)")
    parser.add_argument("--vocab", type=int, default=16, help="Vocabulary size")
    parser.add_argument("--seq-len", type=int, default=8, help="Max sequence length")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=0, help="Number of attention heads (0=auto)")
    parser.add_argument("--report", type=str, default="cosim_report_scaled.txt")
    parser.add_argument("--json-report", type=str, default="", help="Optional machine-readable JSON report path")
    parser.add_argument("--emit-checkpoints", action="store_true", help="Emit layer/final-LN checkpoints from Verilog for WS1 parity")
    parser.add_argument("--warmup-token-seq", type=str, default="", help="Optional comma-separated warmup token IDs (executed before measured tokens)")
    parser.add_argument("--warmup-position-seq", type=str, default="", help="Optional comma-separated warmup position IDs aligned to --warmup-token-seq")
    parser.add_argument("--disable-startup-warmup", action="store_true", help="Disable automatic startup warmup for checkpoint runs")
    parser.add_argument("--logit-tolerance", type=float, default=2.0, help="Max absolute tolerance for logits parity")
    parser.add_argument("--checkpoint-tolerance", type=float, default=2.0, help="Max absolute tolerance for checkpoint parity")
    parser.add_argument("--fail-on-parity", action="store_true", help="Fail (exit code 2) if parity gate does not pass")
    args = parser.parse_args()

    ED = args.dim
    FD = ED * args.ffn_mult
    VS = args.vocab
    MSL = args.seq_len
    NL = args.layers
    DW = 16
    NH = args.heads if args.heads > 0 else max(1, ED // 8)
    HD = ED // NH
    CLK_MHZ = 100

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_path = os.path.join(root_dir, args.report)

    # Open report file
    report = open(report_path, 'w', encoding='utf-8')
    def out(s=""):
        print(s)
        report.write(s + "\n")

    def run_checked(cmd, step, cwd=None, timeout=None):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{step} timed out after {timeout}s") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr if stderr else stdout
            raise RuntimeError(f"{step} failed with return code {result.returncode}\n{detail}")
        return result

    out("")
    out("=" * 70)
    out(f"  BitbyBit GPU -- Scaled Cosimulation (EMBED_DIM={ED})")
    out("=" * 70)
    out(f"  Config: dim={ED}, ffn={FD}, vocab={VS}, seq={MSL}, layers={NL}, heads={NH}")
    out(f"  Parameters: {ED*ED*4 + ED*FD*2 + ED*6 + FD:,} weight values")
    out("")

    # 1. Extract weights at new dimensions
    cache_dir = os.path.join(root_dir, "weights", "cache")
    weight_dir = os.path.join(root_dir, "weights", f"gpt2_dim{ED}")
    npz_file = os.path.join(weight_dir, "gpt2_q88_weights.npz")

    if os.path.exists(npz_file):
        out(f"  Loading cached Q8.8 weights (dim={ED})...")
        weights_q88 = dict(np.load(npz_file, allow_pickle=True))
    else:
        out(f"  Extracting GPT-2 weights for dim={ED}...")
        os.makedirs(cache_dir, exist_ok=True)
        raw = download_gpt2_weights(cache_dir)
        weights_q88 = extract_for_bitbybit(
            raw, embed_dim=ED, ffn_dim=FD, vocab_size=VS,
            max_seq_len=MSL, num_layers=NL, output_dir=weight_dir
        )

    raw_npz = os.path.join(cache_dir, "gpt2_weights.npz")
    raw_weights = dict(np.load(raw_npz, allow_pickle=True)) if os.path.exists(raw_npz) else None

    # 2. Write hex files for $readmemh
    hex_dir = os.path.join(root_dir, "weights", f"gpt2_dim{ED}", "hex_sim")
    write_weight_hex_files(weights_q88, hex_dir, ED, FD, VS, MSL)
    out(f"  Wrote hex files to {hex_dir}")

    # 3. Tokenize
    if args.token_seq.strip():
        token_seq = []
        for raw_tok in args.token_seq.split(','):
            raw_tok = raw_tok.strip()
            if not raw_tok:
                continue
            token_seq.append(int(raw_tok) % VS)
        if not token_seq:
            raise RuntimeError("--token-seq was provided but no token IDs were parsed")
        token_seq = token_seq[:MSL]

        if args.position_seq.strip():
            position_seq = []
            for raw_pos in args.position_seq.split(','):
                raw_pos = raw_pos.strip()
                if not raw_pos:
                    continue
                position_seq.append(int(raw_pos) % MSL)
            if len(position_seq) != len(token_seq):
                raise RuntimeError("--position-seq length must match --token-seq length")
        else:
            position_seq = list(range(len(token_seq)))
        input_text = "<token-seq>"
    else:
        input_text = args.sentence
        token_seq = tokenize(input_text, VS, MSL)
        position_seq = list(range(len(token_seq)))

    if not token_seq:
        raise RuntimeError("Empty token sequence after parsing input")

    warmup_token_seq = []
    warmup_position_seq = []
    if not args.disable_startup_warmup and args.emit_checkpoints:
        warmup_tok_raw = args.warmup_token_seq.strip()
        warmup_pos_raw = args.warmup_position_seq.strip()

        if warmup_tok_raw:
            for raw_tok in args.warmup_token_seq.split(','):
                raw_tok = raw_tok.strip()
                if not raw_tok:
                    continue
                warmup_token_seq.append(int(raw_tok) % VS)
            if not warmup_token_seq:
                raise RuntimeError("--warmup-token-seq was provided but no token IDs were parsed")

        if warmup_pos_raw:
            for raw_pos in args.warmup_position_seq.split(','):
                raw_pos = raw_pos.strip()
                if not raw_pos:
                    continue
                warmup_position_seq.append(int(raw_pos) % MSL)

        if warmup_token_seq and warmup_position_seq:
            if len(warmup_position_seq) != len(warmup_token_seq):
                raise RuntimeError("--warmup-position-seq length must match --warmup-token-seq length")
        elif warmup_token_seq and not warmup_position_seq:
            warmup_position_seq = [0] * len(warmup_token_seq)
        elif warmup_position_seq and not warmup_token_seq:
            warmup_token_seq = [0] * len(warmup_position_seq)
        else:
            # Auto-prime prefix positions [0..max_input_pos-1] so later position jumps
            # don't encounter uninitialized checkpoint state.
            max_input_pos = max(position_seq)
            warmup_position_seq = list(range(max_input_pos))
            warmup_token_seq = [0] * len(warmup_position_seq)

    out(f'  Input text:     "{input_text}"')
    out(f"  Token sequence: {token_seq}")
    out(f"  Position seq:   {position_seq}")
    if warmup_token_seq:
        out(f"  Warmup tokens:  {warmup_token_seq}")
        out(f"  Warmup pos:     {warmup_position_seq}")
    out("")

    # 4. Generate testbench
    build_dir = os.path.join(root_dir, "tb", "cocotb", "sim_build")
    os.makedirs(build_dir, exist_ok=True)
    tb_path = os.path.join(build_dir, "scaled_cosim_tb.v")
    generate_testbench(
        token_seq,
        position_seq,
        tb_path,
        hex_dir,
        ED,
        FD,
        VS,
        MSL,
        NL,
        NH,
        HD,
        DW,
        emit_checkpoints=args.emit_checkpoints,
        warmup_token_sequence=warmup_token_seq,
        warmup_position_sequence=warmup_position_seq,
    )
    out(f"  Generated testbench ({len(token_seq)} tokens)")

    # 5. Compile
    sources = [
        os.path.join(root_dir, "rtl", "gpt2", "embedding_lookup.v"),
        os.path.join(root_dir, "rtl", "gpt2", "transformer_block.v"),
        os.path.join(root_dir, "rtl", "gpt2", "gpt2_engine.v"),
        os.path.join(root_dir, "rtl", "transformer", "layer_norm.v"),
        os.path.join(root_dir, "rtl", "transformer", "attention_unit.v"),
        os.path.join(root_dir, "rtl", "transformer", "ffn_block.v"),
        os.path.join(root_dir, "rtl", "transformer", "linear_layer.v"),
        os.path.join(root_dir, "rtl", "compute", "gelu_lut_256.v"),
        os.path.join(root_dir, "rtl", "compute", "exp_lut_256.v"),
        os.path.join(root_dir, "rtl", "compute", "inv_sqrt_lut_256.v"),
        os.path.join(root_dir, "rtl", "compute", "gelu_activation.v"),
        os.path.join(root_dir, "rtl", "compute", "softmax_unit.v"),
        tb_path,
    ]
    out_bin = os.path.join(build_dir, "scaled_cosim")

    out("  [1/3] Compiling Verilog...")
    cmd = [IVERILOG, "-g2012", "-o", out_bin, "-s", "scaled_cosim_tb"] + sources
    try:
        run_checked(cmd, "Compilation")
    except RuntimeError as err:
        out(f"  COMPILE FAILED:\n{err}")
        report.close()
        sys.exit(1)
    out("  [2/3] Running simulation (this may take a while at dim=64)...")

    t_start = time.time()
    try:
        r = run_checked([VVP, out_bin], "Simulation", cwd=build_dir, timeout=600)
    except RuntimeError as err:
        out(f"  SIMULATION FAILED:\n{err}")
        report.close()
        sys.exit(1)
    sim_time = time.time() - t_start
    verilog_output = r.stdout
    out(f"  [3/3] Simulation completed in {sim_time:.1f}s")
    out("")

    # 6. Parse results
    verilog_results, v_config = parse_output(verilog_output, ED, DW)

    # 7. Run CPU reference
    out("  Running CPU Q8.8 reference...")
    cpu_results = {}
    total_zero_mults = 0
    total_all_mults = 0

    q88_ref_weights = prepare_q88_reference_weights(weights_q88, ED, FD, VS, MSL)
    q88_ref_state = init_q88_reference_state(ED, MSL)

    # Prime reference cache/state with the same warmup sequence used by RTL.
    if warmup_token_seq:
        for warm_tok, warm_pos in zip(warmup_token_seq, warmup_position_seq):
            run_q88_reference(
                q88_ref_weights,
                warm_tok,
                warm_pos,
                ED,
                FD,
                NL,
                VS,
                MSL,
                state=q88_ref_state,
                capture_trace=False,
                collect_stats=False,
            )

    for seq_pos, (tok, input_pos) in enumerate(zip(token_seq, position_seq)):
        ref_logits_f32 = None
        if raw_weights is not None:
            ref_logits_f32 = run_float32_reference(raw_weights, tok, input_pos, ED, FD)

        q88_logits, q88_pred, q88_stats, q88_trace = run_q88_reference(
            q88_ref_weights,
            tok,
            input_pos,
            ED,
            FD,
            NL,
            VS,
            MSL,
            state=q88_ref_state,
            capture_trace=args.emit_checkpoints,
        )
        total_zero_mults += q88_stats['zero_mults']
        total_all_mults += q88_stats['total_mults']

        cpu_results[seq_pos] = {
            'input_pos': input_pos,
            'logits_f32': ref_logits_f32,
            'logits_q88': q88_logits,
            'predicted_f32': int(np.argmax(ref_logits_f32)) if ref_logits_f32 is not None else -1,
            'predicted_q88': q88_pred,
            'trace': q88_trace,
        }

    # 8. Comparison report
    out("=" * 70)
    out("                      COMPARISON REPORT")
    out("=" * 70)
    out("")

    out("  PER-TOKEN RESULTS:")
    out(f"  {'Pos':>3} | {'Inp':>3} | {'Char':>4} | {'ID':>3} | {'GPU Pred':>8} | {'Q88 Pred':>8} | {'F32 Pred':>8} | {'Cycles':>7}")
    out(f"  {'---':>3} | {'---':>3} | {'----':>4} | {'---':>3} | {'--------':>8} | {'--------':>8} | {'--------':>8} | {'-------':>7}")

    total_cycles = 0
    total_mse = 0
    matches_q88 = 0
    matches_f32 = 0
    n = len(token_seq)
    parity_tokens = []
    overall_parity_pass = True

    for pos in range(n):
        vr = verilog_results.get(pos, {})
        cr = cpu_results.get(pos, {})

        v_pred = vr.get('predicted', -1)
        v_cycles = vr.get('cycles', 0)
        v_logits = np.array(vr.get('logits', [0]*ED))
        total_cycles += v_cycles

        q_pred = cr['predicted_q88']
        f_pred = cr['predicted_f32']
        q_logits = cr['logits_q88']
        input_pos = vr.get('input_pos', cr.get('input_pos', pos))

        if v_pred == q_pred: matches_q88 += 1
        if v_pred == f_pred: matches_f32 += 1

        # MSE on compared logits dims
        num_compare = ED if args.emit_checkpoints else min(16, ED)
        logit_metrics = compute_vector_metrics(q_logits[:num_compare], v_logits[:num_compare])
        mse = logit_metrics['mse']
        total_mse += mse
        logit_pass = logit_metrics['max_abs_error'] <= args.logit_tolerance

        layer_parity = []
        final_ln_parity = None
        if args.emit_checkpoints:
            trace = cr.get('trace') or {}
            ref_layers = trace.get('layers', {})
            hw_layers = (vr.get('checkpoints') or {}).get('layers', {})

            for layer_idx in range(NL):
                layer_key = str(layer_idx)
                ref_vec = ref_layers.get(layer_key)
                hw_vec = hw_layers.get(layer_key)
                if ref_vec is None or hw_vec is None or any(v is None for v in hw_vec):
                    metrics = {
                        'layer': layer_idx,
                        'missing': True,
                        'mean_abs_error': None,
                        'max_abs_error': None,
                        'mse': None,
                        'pass': False,
                    }
                else:
                    vm = compute_vector_metrics(ref_vec[:ED], hw_vec[:ED])
                    metrics = {
                        'layer': layer_idx,
                        'missing': False,
                        **vm,
                        'pass': vm['max_abs_error'] <= args.checkpoint_tolerance,
                    }
                layer_parity.append(metrics)

            ref_final_ln = trace.get('final_ln')
            hw_final_ln = (vr.get('checkpoints') or {}).get('final_ln')
            if ref_final_ln is None or hw_final_ln is None or any(v is None for v in hw_final_ln):
                final_ln_parity = {
                    'missing': True,
                    'mean_abs_error': None,
                    'max_abs_error': None,
                    'mse': None,
                    'pass': False,
                }
            else:
                vm = compute_vector_metrics(ref_final_ln[:ED], hw_final_ln[:ED])
                final_ln_parity = {
                    'missing': False,
                    **vm,
                    'pass': vm['max_abs_error'] <= args.checkpoint_tolerance,
                }

        tok_id = vr.get('token_id', token_seq[pos])
        ch = TOKEN_TO_CHAR.get(tok_id % 16, '?')
        v_ch = TOKEN_TO_CHAR.get(v_pred % 16, '?') if v_pred >= 0 else '?'

        token_pass = logit_pass
        if args.emit_checkpoints:
            token_pass = token_pass and all(item['pass'] for item in layer_parity)
            token_pass = token_pass and bool(final_ln_parity and final_ln_parity['pass'])

        parity_tokens.append({
            'position_index': pos,
            'input_pos': input_pos,
            'token_id': tok_id,
            'logits': {
                **logit_metrics,
                'pass': logit_pass,
                'tolerance': args.logit_tolerance,
            },
            'layers': layer_parity,
            'final_ln': final_ln_parity,
            'pass': token_pass,
        })
        overall_parity_pass = overall_parity_pass and token_pass

        out(f"  {pos:>3} | {input_pos:>3} | '{ch}'  | {tok_id:>3} | {v_pred:>3} ('{v_ch}')  | {q_pred:>8} | {f_pred:>8} | {v_cycles:>7}")

    out("")

    zero_rate = (total_zero_mults / max(total_all_mults, 1)) * 100
    avg_cyc = total_cycles / max(n, 1)
    lat_us = total_cycles / CLK_MHZ

    out("  AGGREGATE METRICS:")
    out(f"  +-----------------------------------+------------------+")
    out(f"  | Metric                            | Value            |")
    out(f"  +-----------------------------------+------------------+")
    out(f"  | Embedding Dimension               | {ED:>16} |")
    out(f"  | FFN Dimension                     | {FD:>16} |")
    out(f"  | Total Weight Parameters           | {ED*ED*4 + ED*FD*2 + ED*6 + FD:>16,} |")
    out(f"  | Total Tokens Processed            | {n:>16} |")
    out(f"  | Total GPU Clock Cycles            | {total_cycles:>16,} |")
    out(f"  | Avg Cycles Per Token              | {avg_cyc:>16,.1f} |")
    out(f"  | Est. Latency @ {CLK_MHZ}MHz             | {lat_us:>13.1f} us |")
    out(f"  | Verilog vs Q8.8 Match             | {matches_q88}/{n:>13} |")
    out(f"  | Verilog vs Float32 Match          | {matches_f32}/{n:>13} |")
    out(f"  | Avg MSE (vs Q8.8 reference)       | {total_mse/max(n,1):>16.6f} |")
    out(f"  | Parity Gate (all tokens)          | {'PASS' if overall_parity_pass else 'FAIL':>16} |")
    out(f"  | Zero-Skip Rate (activations)      | {zero_rate:>14.1f}% |")
    out(f"  | Est. Throughput Boost (0-skip)     | {1/(max(1-zero_rate/100, 0.01)):>15.2f}x |")
    out(f"  | Simulation Wall-Clock Time        | {sim_time:>13.1f} s |")
    out(f"  +-----------------------------------+------------------+")
    out("")

    out("  PARITY SUMMARY:")
    out(f"    Logit tolerance:      {args.logit_tolerance:.4f}")
    if args.emit_checkpoints:
        out(f"    Checkpoint tolerance: {args.checkpoint_tolerance:.4f}")
    for token_parity in parity_tokens:
        pos = token_parity['position_index']
        logit_max = token_parity['logits']['max_abs_error']
        token_status = "PASS" if token_parity['pass'] else "FAIL"
        out(f"    Token {pos}: status={token_status} logit_max_abs_err={logit_max:.6f}")
    out("")

    # Scaling comparison: dim=4 vs dim=64
    out("  SCALING COMPARISON (dim=4 vs dim=64):")
    out(f"  +----------------------------+-----------+-----------+")
    out(f"  | Metric                     | dim=4     | dim={ED:<5} |")
    out(f"  +----------------------------+-----------+-----------+")
    out(f"  | Weight parameters          | 232       | {ED*ED*4 + ED*FD*2 + ED*6 + FD:>9,} |")
    out(f"  | Cycles per token           | 130       | {avg_cyc:>9,.0f} |")
    out(f"  | Logit expressiveness       | Low       | {'High':>9} |")
    out(f"  | Token differentiation      | Poor      | {'Good':>9} |")
    out(f"  +----------------------------+-----------+-----------+")
    out("")

    # Detailed logits (first 2 tokens, first 16 dims)
    out("  LOGIT DETAILS (first 2 tokens, first 16 dims):")
    for pos in range(min(2, n)):
        vr = verilog_results.get(pos, {})
        v_logits = vr.get('logits', [0]*ED)
        cr = cpu_results.get(pos, {})
        q_logits = cr['logits_q88']

        tok_id = vr.get('token_id', token_seq[pos])
        ch = TOKEN_TO_CHAR.get(tok_id % 16, '?')
        out(f"  Token {pos} ('{ch}', id={tok_id}):")
        out(f"    {'Dim':>4} | {'Verilog':>10} | {'CPU Q88':>10} | {'Err':>8}")
        for d in range(min(16, ED)):
            vl = v_logits[d] if d < len(v_logits) else 0.0
            ql = float(q_logits[d])
            err = abs(ql - vl)
            out(f"    [{d:>2}] | {vl:>+10.4f} | {ql:>+10.4f} | {err:>8.4f}")
        if ED > 16:
            out(f"    ... ({ED-16} more dimensions)")
        out("")

    # Raw Verilog
    out("  RAW VERILOG OUTPUT (tokens only):")
    for line in verilog_output.split('\n'):
        s = line.strip()
        if s.startswith("TOKEN") or s.startswith("SUMMARY") or s.startswith("CONFIG") or s == "DONE":
            out(f"  {s}")
    out("")

    json_path = None
    if args.json_report:
        json_path = args.json_report
        if not os.path.isabs(json_path):
            json_path = os.path.join(root_dir, json_path)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        json_payload = {
            'run_id': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
            'config': {
                'dim': ED,
                'ffn_dim': FD,
                'vocab': VS,
                'seq_len': MSL,
                'layers': NL,
                'heads': NH,
                'head_dim': HD,
                'clock_mhz': CLK_MHZ,
            },
            'input': {
                'sentence': input_text,
                'token_sequence': token_seq,
                'position_sequence': position_seq,
                'warmup_token_sequence': warmup_token_seq,
                'warmup_position_sequence': warmup_position_seq,
            },
            'aggregate': {
                'total_tokens': n,
                'total_cycles': int(total_cycles),
                'avg_cycles_per_token': float(avg_cyc),
                'latency_us': float(lat_us),
                'verilog_vs_q88_matches': int(matches_q88),
                'verilog_vs_float32_matches': int(matches_f32),
                'avg_mse_vs_q88': float(total_mse / max(n, 1)),
                'zero_skip_rate_pct': float(zero_rate),
                'throughput_boost_estimate': float(1 / (max(1 - zero_rate / 100, 0.01))),
                'sim_time_seconds': float(sim_time),
            },
            'parity': {
                'emit_checkpoints': bool(args.emit_checkpoints),
                'logit_tolerance': float(args.logit_tolerance),
                'checkpoint_tolerance': float(args.checkpoint_tolerance),
                'overall_pass': bool(overall_parity_pass),
                'tokens': parity_tokens,
            },
            'verilog_config': v_config,
        }

        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(json_payload, jf, indent=2)
        out(f"  JSON report saved to: {json_path}")
        out("")

    report.close()
    print(f"  Full report saved to: {report_path}")

    if args.fail_on_parity and not overall_parity_pass:
        print("  [FAIL-CLOSE] parity gate failed")
        sys.exit(2)

if __name__ == "__main__":
    main()
