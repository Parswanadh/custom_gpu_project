#!/usr/bin/env python3
"""
run_sentence_cosim.py -- End-to-End GPT-2 Sentence Inference Cosimulation

Loads REAL GPT-2 weights into the custom BitbyBit Verilog GPU, processes a
full sentence token-by-token, and generates a detailed comparison report against
a CPU float32 reference. Monitors:
  - Per-token clock cycles, logits, predictions
  - Quantization error (MSE) per token
  - Zero-skip statistics (sparsity of activations)
  - ALU utilization estimates
  - Total latency comparison (custom GPU vs estimated CPU baseline)

Usage:
    python scripts/run_sentence_cosim.py
    python scripts/run_sentence_cosim.py --sentence "Hello world"
    python scripts/run_sentence_cosim.py --generate 5
"""

import numpy as np
import subprocess
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from extract_gpt2_weights import (
    float_to_q88, q88_to_float, q88_hex,
    download_gpt2_weights, extract_for_bitbybit,
    run_float32_reference, layer_norm_f32, gelu_f32, extract_ln
)

# ==========================================================================
# Configuration
# ==========================================================================

IVERILOG = r"D:\Tools\iverilog\bin\iverilog.exe"
VVP      = r"D:\Tools\iverilog\bin\vvp.exe"

EMBED_DIM   = 4
FFN_DIM     = 8
VOCAB_SIZE  = 16
MAX_SEQ_LEN = 8
NUM_LAYERS  = 2
DATA_WIDTH  = 16
NUM_HEADS   = 2
HEAD_DIM    = 2
CLK_FREQ_MHZ = 100  # Assumed target clock for estimates

# ==========================================================================
# Simple tokenizer mapping (our GPU has VOCAB_SIZE=16)
# Maps characters/words to token IDs 0-15
# ==========================================================================

CHAR_TO_TOKEN = {
    ' ':  0,  'a':  1,  'b':  2,  'c':  3,
    'd':  4,  'e':  5,  'f':  6,  'g':  7,
    'h':  8,  'i':  9,  'j': 10,  'k': 11,
    'l': 12,  'm': 13,  'n': 14,  'o': 15,
}

TOKEN_TO_CHAR = {v: k for k, v in CHAR_TO_TOKEN.items()}

def tokenize(text, max_len=MAX_SEQ_LEN):
    """Map characters to token IDs 0-15 (wrapping for unknown chars)."""
    tokens = []
    for ch in text.lower()[:max_len]:
        if ch in CHAR_TO_TOKEN:
            tokens.append(CHAR_TO_TOKEN[ch])
        else:
            tokens.append(ord(ch) % VOCAB_SIZE)
    return tokens

def detokenize(token_ids):
    """Map token IDs back to characters."""
    return ''.join(TOKEN_TO_CHAR.get(t, '?') for t in token_ids)

# ==========================================================================
# Q8.8 float32 reference with sparsity tracking
# ==========================================================================

def run_q88_reference(weights_q88, token_id, position, raw_weights):
    """
    Run Q8.8 quantized inference in Python (matching Verilog precision).
    Also tracks zero counts for sparsity analysis.
    """
    ED, FD = EMBED_DIM, FFN_DIM
    stats = {'zero_mults': 0, 'total_mults': 0, 'activations': []}

    # Embedding
    tok_emb = np.array([q88_to_float(int(v)) for v in weights_q88['token_emb'][token_id]])
    pos_emb = np.array([q88_to_float(int(v)) for v in weights_q88['pos_emb'][position]])
    x = tok_emb + pos_emb
    stats['activations'].append(('embedding', x.copy()))

    # Count zeros in embedding output
    zeros = np.sum(np.abs(x) < 0.01)
    stats['zero_mults'] += int(zeros) * ED  # Each zero activation skips ED multiplies
    stats['total_mults'] += ED * ED  # Matrix multiply size

    for layer in range(NUM_LAYERS):
        residual = x.copy()

        # LayerNorm 1
        gamma = np.array([q88_to_float(int(v)) for v in weights_q88['ln1_gamma']])
        beta  = np.array([q88_to_float(int(v)) for v in weights_q88['ln1_beta']])
        x = layer_norm_f32(x, gamma, beta)
        stats['activations'].append((f'ln1_L{layer}', x.copy()))

        # Attention: Q, K, V projections
        wq = np.array([[q88_to_float(int(weights_q88['wq'][r, c])) for c in range(ED)] for r in range(ED)])
        wv = np.array([[q88_to_float(int(weights_q88['wv'][r, c])) for c in range(ED)] for r in range(ED)])
        wo = np.array([[q88_to_float(int(weights_q88['wo'][r, c])) for c in range(ED)] for r in range(ED)])

        q_vec = wq.T @ x
        v_vec = wv.T @ x
        attn_out = wo.T @ v_vec
        x = residual + attn_out
        stats['activations'].append((f'attn_L{layer}', attn_out.copy()))

        # Count zero elements in attention intermediates
        for vec in [q_vec, v_vec, attn_out]:
            z = np.sum(np.abs(vec) < 0.01)
            stats['zero_mults'] += int(z) * ED
            stats['total_mults'] += ED * ED

        residual = x.copy()

        # LayerNorm 2
        gamma2 = np.array([q88_to_float(int(v)) for v in weights_q88['ln2_gamma']])
        beta2  = np.array([q88_to_float(int(v)) for v in weights_q88['ln2_beta']])
        x = layer_norm_f32(x, gamma2, beta2)

        # FFN
        w1 = np.array([[q88_to_float(int(weights_q88['ffn_w1'][r, c])) for c in range(FD)] for r in range(ED)])
        b1 = np.array([q88_to_float(int(v)) for v in weights_q88['ffn_b1']])
        w2 = np.array([[q88_to_float(int(weights_q88['ffn_w2'][r, c])) for c in range(ED)] for r in range(FD)])
        b2 = np.array([q88_to_float(int(v)) for v in weights_q88['ffn_b2']])

        h = w1.T @ x + b1
        h_pre_gelu = h.copy()
        h = gelu_f32(h)
        stats['activations'].append((f'gelu_L{layer}', h.copy()))

        # Count zeros after GELU (this is where zero-skip helps most!)
        gelu_zeros = np.sum(np.abs(h) < 0.01)
        stats['zero_mults'] += int(gelu_zeros) * ED  # Each zero output skips ED mults in W2
        stats['total_mults'] += ED * FD + FD * ED  # W1 + W2

        ffn_out = w2.T @ h + b2
        x = residual + ffn_out
        stats['activations'].append((f'ffn_L{layer}', ffn_out.copy()))

    # Final LayerNorm
    gamma_f = np.array([q88_to_float(int(v)) for v in weights_q88['ln_final_gamma']])
    beta_f  = np.array([q88_to_float(int(v)) for v in weights_q88['ln_final_beta']])
    x = layer_norm_f32(x, gamma_f, beta_f)
    stats['activations'].append(('final_ln', x.copy()))

    return x, stats

# ==========================================================================
# Generate Verilog testbench (baked-in weights, sequential sentence tokens)
# ==========================================================================

def pack_hex(values):
    """Pack Q8.8 values into hex (MSB-first for Verilog literal)."""
    result = ""
    for val in reversed(values):
        result += f"{int(val) & 0xFFFF:04x}"
    return result

def generate_sentence_testbench(weights, token_sequence, output_path):
    """Generate testbench that processes a full sentence, token by token."""
    ED, FD, VS, MSL, DW = EMBED_DIM, FFN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, DATA_WIDTH
    tk_bits = max(1, int(np.ceil(np.log2(VS))))
    pos_bits = max(1, int(np.ceil(np.log2(MSL))))
    dim_bits = max(1, int(np.ceil(np.log2(ED))))

    lines = []
    lines.append('`timescale 1ns/1ps')
    lines.append('')
    lines.append('module sentence_cosim_tb;')
    lines.append(f'  parameter VOCAB_SIZE  = {VS};')
    lines.append(f'  parameter MAX_SEQ_LEN = {MSL};')
    lines.append(f'  parameter EMBED_DIM   = {ED};')
    lines.append(f'  parameter NUM_HEADS   = {NUM_HEADS};')
    lines.append(f'  parameter HEAD_DIM    = {HEAD_DIM};')
    lines.append(f'  parameter FFN_DIM     = {FD};')
    lines.append(f'  parameter NUM_LAYERS  = {NUM_LAYERS};')
    lines.append(f'  parameter DATA_WIDTH  = {DW};')
    lines.append('')

    # Declarations
    lines.append('  reg clk, rst;')
    lines.append('  reg valid_in;')
    lines.append(f'  reg [{tk_bits-1}:0] token_in;')
    lines.append(f'  reg [{pos_bits-1}:0] position_in;')
    lines.append('  reg load_token_emb, load_pos_emb;')
    lines.append(f'  reg [{tk_bits-1}:0] load_token_idx;')
    lines.append(f'  reg [{dim_bits-1}:0] load_dim_idx;')
    lines.append(f'  reg signed [{DW-1}:0] load_emb_data;')
    lines.append(f'  reg [{pos_bits-1}:0] load_pos_idx;')
    lines.append(f'  reg [{ED*DW-1}:0] ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;')
    lines.append(f'  reg [{ED*DW-1}:0] ln_final_gamma, ln_final_beta;')
    lines.append(f'  reg [{ED*ED*DW-1}:0] wq_flat, wk_flat, wv_flat, wo_flat;')
    lines.append(f'  reg [{ED*FD*DW-1}:0] ffn_w1_flat;')
    lines.append(f'  reg [{FD*DW-1}:0] ffn_b1_flat;')
    lines.append(f'  reg [{FD*ED*DW-1}:0] ffn_w2_flat;')
    lines.append(f'  reg [{ED*DW-1}:0] ffn_b2_flat;')
    lines.append(f'  wire [{tk_bits-1}:0] token_out;')
    lines.append(f'  wire [{ED*DW-1}:0] logits_out;')
    lines.append('  wire valid_out;')
    lines.append('  integer cycle_count;')
    lines.append('  integer total_cycles;')
    lines.append('  integer token_count;')
    lines.append('')

    # DUT
    lines.append('  gpt2_engine #(')
    lines.append('    .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),')
    lines.append('    .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),')
    lines.append('    .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),')
    lines.append('    .NUM_LAYERS(NUM_LAYERS), .DATA_WIDTH(DATA_WIDTH)')
    lines.append('  ) dut (')
    lines.append('    .clk(clk), .rst(rst),')
    lines.append('    .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),')
    lines.append('    .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),')
    lines.append('    .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),')
    lines.append('    .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),')
    lines.append('    .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),')
    lines.append('    .wq_flat(wq_flat), .wk_flat(wk_flat),')
    lines.append('    .wv_flat(wv_flat), .wo_flat(wo_flat),')
    lines.append('    .ffn_w1_flat(ffn_w1_flat), .ffn_b1_flat(ffn_b1_flat),')
    lines.append('    .ffn_w2_flat(ffn_w2_flat), .ffn_b2_flat(ffn_b2_flat),')
    lines.append('    .ln_final_gamma(ln_final_gamma), .ln_final_beta(ln_final_beta),')
    lines.append('    .valid_in(valid_in), .token_in(token_in),')
    lines.append('    .position_in(position_in),')
    lines.append('    .token_out(token_out), .logits_out(logits_out),')
    lines.append('    .valid_out(valid_out)')
    lines.append('  );')
    lines.append('')
    lines.append('  always #5 clk = ~clk;')
    lines.append('')
    lines.append('  initial begin')
    lines.append('    $dumpfile("sentence_cosim.vcd");')
    lines.append('    $dumpvars(0, sentence_cosim_tb);')
    lines.append('  end')
    lines.append('')

    # Main test
    lines.append('  initial begin')
    lines.append('    clk = 0; rst = 1;')
    lines.append('    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;')
    lines.append('    token_in = 0; position_in = 0;')
    lines.append('    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;')
    lines.append('    load_pos_idx = 0;')
    lines.append('    total_cycles = 0; token_count = 0;')
    lines.append('')

    # Set weight buses
    lines.append('    // ===== REAL GPT-2 WEIGHTS (Q8.8) =====')
    lines.append(f'    ln1_gamma = {ED*DW}\'h{pack_hex(weights["ln1_gamma"][:ED])};')
    lines.append(f'    ln1_beta  = {ED*DW}\'h{pack_hex(weights["ln1_beta"][:ED])};')
    lines.append(f'    ln2_gamma = {ED*DW}\'h{pack_hex(weights["ln2_gamma"][:ED])};')
    lines.append(f'    ln2_beta  = {ED*DW}\'h{pack_hex(weights["ln2_beta"][:ED])};')
    lines.append(f'    ln_final_gamma = {ED*DW}\'h{pack_hex(weights["ln_final_gamma"][:ED])};')
    lines.append(f'    ln_final_beta  = {ED*DW}\'h{pack_hex(weights["ln_final_beta"][:ED])};')
    lines.append(f'    wq_flat  = {ED*ED*DW}\'h{pack_hex(weights["wq"][:ED,:ED].flatten())};')
    lines.append(f'    wk_flat  = {ED*ED*DW}\'h{pack_hex(weights["wk"][:ED,:ED].flatten())};')
    lines.append(f'    wv_flat  = {ED*ED*DW}\'h{pack_hex(weights["wv"][:ED,:ED].flatten())};')
    lines.append(f'    wo_flat  = {ED*ED*DW}\'h{pack_hex(weights["wo"][:ED,:ED].flatten())};')
    lines.append(f'    ffn_w1_flat = {ED*FD*DW}\'h{pack_hex(weights["ffn_w1"][:ED,:FD].flatten())};')
    lines.append(f'    ffn_b1_flat = {FD*DW}\'h{pack_hex(weights["ffn_b1"][:FD])};')
    lines.append(f'    ffn_w2_flat = {FD*ED*DW}\'h{pack_hex(weights["ffn_w2"][:FD,:ED].flatten())};')
    lines.append(f'    ffn_b2_flat = {ED*DW}\'h{pack_hex(weights["ffn_b2"][:ED])};')
    lines.append('')
    lines.append('    #35 rst = 0; #25;')
    lines.append('')

    # Load embeddings
    lines.append('    // ===== LOAD EMBEDDINGS =====')
    for tid in range(VS):
        for dim in range(ED):
            val = int(weights['token_emb'][tid, dim]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_token_emb = 1; load_token_idx = {tid}; load_dim_idx = {dim}; load_emb_data = 16\'h{val:04x};')
            lines.append(f'    @(negedge clk); load_token_emb = 0;')

    for pid in range(MSL):
        for dim in range(ED):
            val = int(weights['pos_emb'][pid, dim]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_pos_emb = 1; load_pos_idx = {pid}; load_dim_idx = {dim}; load_emb_data = 16\'h{val:04x};')
            lines.append(f'    @(negedge clk); load_pos_emb = 0;')

    lines.append('    #20;')
    lines.append('')

    # Banner
    seq_str = ','.join(str(t) for t in token_sequence)
    lines.append('    $display("");')
    lines.append('    $display("+=========================================================+");')
    lines.append('    $display("|  BitbyBit GPU -- Sentence Processing Cosimulation        |");')
    lines.append('    $display("|  Model: Real GPT-2 (Q8.8 quantized)                     |");')
    lines.append(f'    $display("|  Tokens: [{seq_str}]");')
    lines.append(f'    $display("|  Sequence length: {len(token_sequence)} tokens                                 |");')
    lines.append('    $display("+=========================================================+");')
    lines.append('    $display("");')
    lines.append('')

    # Process each token
    for pos, token_id in enumerate(token_sequence):
        lines.append(f'    // ===== TOKEN {pos}: id={token_id} =====')
        lines.append(f'    @(negedge clk);')
        lines.append(f'    token_in = {token_id}; position_in = {pos};')
        lines.append(f'    valid_in = 1;')
        lines.append(f'    @(negedge clk); valid_in = 0;')
        lines.append(f'    cycle_count = 0;')
        lines.append(f'    while (!valid_out && cycle_count < 1000) begin')
        lines.append(f'      @(negedge clk); cycle_count = cycle_count + 1;')
        lines.append(f'    end')
        lines.append('')
        lines.append(f'    if (valid_out) begin')
        lines.append(f'      total_cycles = total_cycles + cycle_count;')
        lines.append(f'      token_count = token_count + 1;')
        lines.append(f'      $display("TOKEN pos=%0d id=%0d predicted=%0d cycles=%0d logits=%h",')
        lines.append(f'               {pos}, {token_id}, token_out, cycle_count, logits_out);')

        # Print each logit dimension
        for d in range(ED):
            lines.append(f'      $display("  LOGIT pos=%0d dim=%0d hex=%h",')
            lines.append(f'               {pos}, {d}, logits_out[{d*DW} +: {DW}]);')

        lines.append(f'    end else begin')
        lines.append(f'      $display("TOKEN pos=%0d id=%0d TIMEOUT", {pos}, {token_id});')
        lines.append(f'    end')
        lines.append(f'    repeat(5) @(negedge clk);')
        lines.append('')

    # Summary
    lines.append('    $display("");')
    lines.append('    $display("SUMMARY total_tokens=%0d total_cycles=%0d", token_count, total_cycles);')
    lines.append('    $display("DONE");')
    lines.append('    $finish;')
    lines.append('  end')
    lines.append('endmodule')

    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(lines) + '\n')

# ==========================================================================
# Compile, run, parse
# ==========================================================================

def compile_and_run(tb_path, root_dir):
    """Compile and run with Icarus Verilog."""
    sources = [
        os.path.join(root_dir, "rtl", "gpt2", "embedding_lookup.v"),
        os.path.join(root_dir, "rtl", "gpt2", "transformer_block.v"),
        os.path.join(root_dir, "rtl", "gpt2", "gpt2_engine.v"),
        os.path.join(root_dir, "rtl", "transformer", "layer_norm.v"),
        os.path.join(root_dir, "rtl", "transformer", "attention_unit.v"),
        os.path.join(root_dir, "rtl", "transformer", "ffn_block.v"),
        os.path.join(root_dir, "rtl", "transformer", "linear_layer.v"),
        os.path.join(root_dir, "rtl", "compute", "gelu_activation.v"),
        os.path.join(root_dir, "rtl", "compute", "softmax_unit.v"),
        tb_path,
    ]
    build_dir = os.path.join(root_dir, "tb", "cocotb", "sim_build")
    os.makedirs(build_dir, exist_ok=True)
    out_bin = os.path.join(build_dir, "sentence_cosim")

    print("  [1/3] Compiling Verilog...")
    cmd = [IVERILOG, "-o", out_bin, "-s", "sentence_cosim_tb"] + sources
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  COMPILE FAILED:")
        print(r.stderr)
        return None
    print("  [2/3] Running Verilog simulation...")
    r = subprocess.run([VVP, out_bin], capture_output=True, text=True, timeout=120, cwd=build_dir)
    return r.stdout

def parse_output(output):
    """Parse TOKEN and LOGIT lines."""
    results = {}
    summary = {}
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith("TOKEN ") and "TIMEOUT" not in line:
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts['pos'])
            results[pos] = {
                'token_id': int(parts['id']),
                'predicted': int(parts['predicted']),
                'cycles': int(parts['cycles']),
                'logits': []
            }
        elif line.startswith("LOGIT "):
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts['pos'])
            hex_val = parts['hex']
            int_val = int(hex_val, 16)
            if int_val >= 32768: int_val -= 65536
            if pos in results:
                results[pos]['logits'].append(int_val / 256.0)
        elif line.startswith("SUMMARY"):
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            summary = parts
        elif "TOKEN" in line and "TIMEOUT" in line:
            parts = {}
            for p in line.split()[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    parts[k] = v
            pos = int(parts.get('pos', -1))
            results[pos] = {'token_id': int(parts.get('id', 0)), 'predicted': -1, 'cycles': -1, 'logits': [], 'timeout': True}
    return results, summary

# ==========================================================================
# Main
# ==========================================================================

class TeeWriter:
    """Write to both console and file simultaneously."""
    def __init__(self, filepath):
        self._file = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout
    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Sentence Cosimulation")
    parser.add_argument("--sentence", type=str, default="hello",
                        help="Sentence to process (chars mapped to tokens)")
    parser.add_argument("--generate", type=int, default=0,
                        help="Number of tokens to auto-generate after input")
    parser.add_argument("--report", type=str, default="cosim_report.txt",
                        help="Output report file path")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Tee output to both console and report file
    report_path = os.path.join(root_dir, args.report)
    tee = TeeWriter(report_path)
    sys.stdout = tee

    print()
    print("=" * 64)
    print("  BitbyBit GPU -- End-to-End Sentence Processing Cosimulation")
    print("=" * 64)
    print()

    # ------ Load weights ------
    cache_dir = os.path.join(root_dir, "weights", "cache")
    weight_dir = os.path.join(root_dir, "weights", "gpt2_real")
    npz_file = os.path.join(weight_dir, "gpt2_q88_weights.npz")

    if os.path.exists(npz_file):
        print("  Loading cached Q8.8 weights...")
        weights_q88 = dict(np.load(npz_file, allow_pickle=True))
    else:
        print("  Extracting weights...")
        raw = download_gpt2_weights(cache_dir)
        weights_q88 = extract_for_bitbybit(raw, output_dir=weight_dir)

    raw_npz = os.path.join(cache_dir, "gpt2_weights.npz")
    raw_weights = dict(np.load(raw_npz, allow_pickle=True)) if os.path.exists(raw_npz) else None

    # ------ Tokenize ------
    input_text = args.sentence
    token_sequence = tokenize(input_text)
    print(f'  Input text:     "{input_text}"')
    print(f"  Token sequence: {token_sequence}")
    print(f"  Token mapping:  {' '.join(f'{ch}->{t}' for ch, t in zip(input_text.lower(), token_sequence))}")
    print()

    # ------ Run Verilog GPU ------
    print("  ---- VERILOG GPU (Custom BitbyBit) ----")
    tb_path = os.path.join(root_dir, "tb", "cocotb", "sim_build", "sentence_cosim_tb.v")
    generate_sentence_testbench(weights_q88, token_sequence, tb_path)
    print(f"  Generated testbench: {len(token_sequence)} tokens")

    verilog_output = compile_and_run(tb_path, root_dir)
    if verilog_output is None:
        print("  Simulation failed!")
        sys.exit(1)

    verilog_results, v_summary = parse_output(verilog_output)
    print(f"  [3/3] Simulation completed!")
    print()

    # ------ Run CPU float32 reference ------
    print("  ---- CPU REFERENCE (Float32 Python) ----")
    cpu_results = {}
    cpu_q88_results = {}

    for pos, token_id in enumerate(token_sequence):
        # Full float32
        t0 = time.perf_counter_ns()
        if raw_weights is not None:
            ref_logits = run_float32_reference(raw_weights, token_id, pos, EMBED_DIM, FFN_DIM)
        else:
            ref_logits = np.zeros(EMBED_DIM)
        cpu_ns = time.perf_counter_ns() - t0

        ref_pred = int(np.argmax(ref_logits))
        cpu_results[pos] = {
            'token_id': token_id,
            'logits': ref_logits,
            'predicted': ref_pred,
            'time_ns': cpu_ns
        }

        # Q8.8 reference (matches Verilog precision)
        q88_logits, q88_stats = run_q88_reference(weights_q88, token_id, pos, raw_weights)
        q88_pred = int(np.argmax(q88_logits))
        cpu_q88_results[pos] = {
            'logits': q88_logits,
            'predicted': q88_pred,
            'stats': q88_stats
        }

    print("  Float32 and Q8.8 reference inference complete!")
    print()

    # =================================================================
    # COMPARISON REPORT
    # =================================================================
    print("=" * 64)
    print("                    COMPARISON REPORT")
    print("=" * 64)
    print()

    # Per-token results
    print("  PER-TOKEN RESULTS:")
    print(f"  {'Pos':>3} | {'Input':>5} | {'Verilog':>7} | {'CPU f32':>7} | {'CPU Q88':>7} | {'Cycles':>6} | {'MSE':>10}")
    print(f"  {'---':>3} | {'-----':>5} | {'-------':>7} | {'-------':>7} | {'-------':>7} | {'------':>6} | {'----------':>10}")

    total_cycles = 0
    total_mse_f32 = 0
    total_mse_q88 = 0
    matches_f32 = 0
    matches_q88 = 0
    total_zero_mults = 0
    total_all_mults = 0

    for pos in range(len(token_sequence)):
        vr = verilog_results.get(pos, {})
        cr = cpu_results.get(pos, {})
        qr = cpu_q88_results.get(pos, {})

        v_pred = vr.get('predicted', -1)
        v_logits = np.array(vr.get('logits', [0]*EMBED_DIM))
        v_cycles = vr.get('cycles', 0)
        total_cycles += v_cycles

        c_pred = cr.get('predicted', -1)
        c_logits = cr.get('logits', np.zeros(EMBED_DIM))
        q_pred = qr.get('predicted', -1)
        q_logits = qr.get('logits', np.zeros(EMBED_DIM))

        mse_f32 = float(np.mean((c_logits - v_logits) ** 2))
        mse_q88 = float(np.mean((q_logits - v_logits) ** 2))
        total_mse_f32 += mse_f32
        total_mse_q88 += mse_q88

        if v_pred == c_pred: matches_f32 += 1
        if v_pred == q_pred: matches_q88 += 1

        stats = qr.get('stats', {})
        total_zero_mults += stats.get('zero_mults', 0)
        total_all_mults += stats.get('total_mults', 1)

        token_char = TOKEN_TO_CHAR.get(vr.get('token_id', 0), '?')
        pred_char = TOKEN_TO_CHAR.get(v_pred, '?')

        print(f"  {pos:>3} | '{token_char}' ={vr.get('token_id',0):>2} | {v_pred:>3} ('{pred_char}') | {c_pred:>3}     | {q_pred:>3}     | {v_cycles:>6} | {mse_f32:>10.6f}")

    print()
    n = len(token_sequence)

    # Summary table
    print("  AGGREGATE METRICS:")
    print(f"  +-------------------------------+----------------+")
    print(f"  | Metric                        | Value          |")
    print(f"  +-------------------------------+----------------+")
    print(f"  | Total tokens processed        | {n:>14} |")
    print(f"  | Total GPU clock cycles        | {total_cycles:>14} |")
    print(f"  | Avg cycles per token          | {total_cycles/max(n,1):>14.1f} |")
    if CLK_FREQ_MHZ > 0:
        latency_us = total_cycles / CLK_FREQ_MHZ
        print(f"  | Est. latency @ {CLK_FREQ_MHZ}MHz        | {latency_us:>11.1f} us |")
    print(f"  | Verilog vs Float32 match      | {matches_f32}/{n:>11} |")
    print(f"  | Verilog vs Q8.8 ref match     | {matches_q88}/{n:>11} |")
    print(f"  | Avg MSE (vs Float32)          | {total_mse_f32/max(n,1):>14.6f} |")
    print(f"  | Avg MSE (vs Q8.8 ref)         | {total_mse_q88/max(n,1):>14.6f} |")
    zero_rate = (total_zero_mults / max(total_all_mults, 1)) * 100
    print(f"  | Zero-skip rate (activations)  | {zero_rate:>12.1f}% |")
    saved_cycles = int(total_cycles * zero_rate / 100)
    print(f"  | Est. cycles saved by 0-skip   | {saved_cycles:>14} |")
    print(f"  | Effective throughput boost     | {1/(1-zero_rate/100):>13.2f}x |")
    print(f"  +-------------------------------+----------------+")
    print()

    # Detailed logit comparison
    print("  DETAILED LOGIT COMPARISON (per token):")
    print()
    for pos in range(len(token_sequence)):
        vr = verilog_results.get(pos, {})
        cr = cpu_results.get(pos, {})
        qr = cpu_q88_results.get(pos, {})

        v_logits = vr.get('logits', [0]*EMBED_DIM)
        c_logits = cr.get('logits', np.zeros(EMBED_DIM))
        q_logits = qr.get('logits', np.zeros(EMBED_DIM))

        tok_id = vr.get('token_id', token_sequence[pos] if pos < len(token_sequence) else 0)
        token_char = TOKEN_TO_CHAR.get(tok_id, '?')

        print(f"  Token {pos} ('{token_char}', id={tok_id}):")
        print(f"    {'Dim':>3} | {'Verilog Q88':>11} | {'CPU Float32':>11} | {'CPU Q88':>11} | {'Err (f32)':>9}")
        for d in range(EMBED_DIM):
            vl = v_logits[d] if d < len(v_logits) else 0.0
            cl = float(c_logits[d])
            ql = float(q_logits[d])
            err = abs(cl - vl)
            print(f"    [{d}] | {vl:>+11.4f} | {cl:>+11.4f} | {ql:>+11.4f} | {err:>9.4f}")
        print()

    # Activation sparsity analysis
    print("  ACTIVATION SPARSITY ANALYSIS:")
    print("  (Shows how many values are near-zero at each pipeline stage)")
    print()
    for pos in range(min(2, len(token_sequence))):  # Show first 2 tokens
        qr = cpu_q88_results.get(pos, {})
        stats = qr.get('stats', {})
        acts = stats.get('activations', [])
        tok_id = token_sequence[pos]
        print(f"  Token {pos} (id={tok_id}):")
        for name, act in acts:
            near_zero = np.sum(np.abs(act) < 0.05)
            total = len(act)
            pct = near_zero / total * 100
            bar = '#' * int(pct / 5) + '.' * (20 - int(pct / 5))
            print(f"    {name:>12}: [{bar}] {near_zero}/{total} ({pct:.0f}% sparse)")
        print()

    # GPU vs CPU comparison
    print("  GPU vs CPU EXECUTION COMPARISON:")
    print()
    total_cpu_ns = sum(cr['time_ns'] for cr in cpu_results.values())
    total_cpu_us = total_cpu_ns / 1000
    gpu_us = total_cycles / CLK_FREQ_MHZ

    print(f"  Custom GPU (BitbyBit):")
    print(f"    Total cycles:      {total_cycles}")
    print(f"    Clock frequency:   {CLK_FREQ_MHZ} MHz (target)")
    print(f"    Latency:           {gpu_us:.1f} us")
    print(f"    Optimizations:     Zero-skip ({zero_rate:.0f}%), INT8 parallel (2x), pipelined")
    print()
    print(f"  CPU Reference (Python float32):")
    print(f"    Wall-clock time:   {total_cpu_us:.0f} us")
    print(f"    (Note: Python is ~100-1000x slower than C/CUDA)")
    print()
    print(f"  Estimated comparison vs CPU baseline:")
    print(f"    GPU @ {CLK_FREQ_MHZ}MHz:        {gpu_us:>8.1f} us / {n} tokens")
    print(f"    ARM Cortex-M4 est: {n * 50:>8} us / {n} tokens (est. 50us/token)")
    print(f"    Speedup vs MCU:    {n * 50 / max(gpu_us, 0.1):>8.1f}x")
    print()

    # Print raw Verilog output
    print("=" * 64)
    print("  RAW VERILOG OUTPUT:")
    print("=" * 64)
    for line in verilog_output.split('\n'):
        s = line.strip()
        if s and ('TOKEN' in s or 'SUMMARY' in s or 'BitbyBit' in s or '+===' in s or 'DONE' in s):
            print(f"  {s}")
    print()

    # Restore stdout and report
    sys.stdout = tee._stdout
    tee.close()
    print(f"  Full report saved to: {report_path}")
    print()

if __name__ == "__main__":
    main()
