#!/usr/bin/env python3
"""
run_cosim.py — Standalone Python-Verilog Cosimulation

Loads real GPT-2 weights (Q8.8), generates a Verilog testbench that
feeds them into the gpt2_engine, runs it with Icarus Verilog,
parses the output, and compares against a float32 reference.

No cocotb, no make, no VPI — just Python + iverilog + vvp.

Usage:
    python scripts/run_cosim.py
    python scripts/run_cosim.py --token 5
"""

import numpy as np
import subprocess
import os
import sys
import json
import tempfile
import argparse
import shutil
from pathlib import Path

# Add scripts dir
sys.path.insert(0, os.path.dirname(__file__))
from extract_gpt2_weights import (
    float_to_q88, q88_to_float, q88_hex,
    download_gpt2_weights, extract_for_bitbybit,
    run_float32_reference
)

# ============================================================================
# Settings
# ============================================================================

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
VVP = resolve_executable("BITBYBIT_VVP", r"D:\Tools\iverilog\bin\vvp.exe", "vvp")

EMBED_DIM = 4
FFN_DIM = 8
VOCAB_SIZE = 16
MAX_SEQ_LEN = 8
NUM_LAYERS = 2
DATA_WIDTH = 16
NUM_HEADS = 2
HEAD_DIM = 2

# ============================================================================
# Generate Verilog testbench with real weights baked in.
# ============================================================================

def generate_cosim_testbench(weights, test_tokens, output_path):
    """Generate a complete Verilog testbench with real GPT-2 weights."""
    
    ED = EMBED_DIM
    FD = FFN_DIM
    VS = VOCAB_SIZE
    MSL = MAX_SEQ_LEN
    DW = DATA_WIDTH
    ffn_bits = max(1, int(np.ceil(np.log2(max(FD, ED)))))
    layer_bits = max(1, int(np.ceil(np.log2(NUM_LAYERS + 1))))
    
    lines = []
    lines.append(f'`timescale 1ns/1ps')
    lines.append(f'')
    lines.append(f'module cosim_tb;')
    lines.append(f'')
    lines.append(f'  parameter VOCAB_SIZE  = {VS};')
    lines.append(f'  parameter MAX_SEQ_LEN = {MSL};')
    lines.append(f'  parameter EMBED_DIM   = {ED};')
    lines.append(f'  parameter NUM_HEADS   = {NUM_HEADS};')
    lines.append(f'  parameter HEAD_DIM    = {HEAD_DIM};')
    lines.append(f'  parameter FFN_DIM     = {FD};')
    lines.append(f'  parameter NUM_LAYERS  = {NUM_LAYERS};')
    lines.append(f'  parameter DATA_WIDTH  = {DW};')
    lines.append(f'')
    lines.append(f'  reg clk, rst;')
    lines.append(f'  reg valid_in;')
    lines.append(f'  reg [{int(np.ceil(np.log2(VS)))-1}:0] token_in;')
    lines.append(f'  reg [{int(np.ceil(np.log2(MSL)))-1}:0] position_in;')
    lines.append(f'')
    lines.append(f'  // Embedding loading')
    lines.append(f'  reg load_token_emb, load_pos_emb;')
    lines.append(f'  reg [{int(np.ceil(np.log2(VS)))-1}:0] load_token_idx;')
    lines.append(f'  reg [{int(np.ceil(np.log2(ED)))-1}:0] load_dim_idx;')
    lines.append(f'  reg signed [{DW-1}:0] load_emb_data;')
    lines.append(f'  reg [{int(np.ceil(np.log2(MSL)))-1}:0] load_pos_idx;')
    lines.append(f'')
    lines.append(f'  // Load-based transformer weight interface')
    lines.append(f'  reg load_ln_en;')
    lines.append(f'  reg [{layer_bits-1}:0] load_layer_idx;')
    lines.append(f'  reg load_ln_sel, load_ln_is_gamma;')
    lines.append(f'  reg [{int(np.ceil(np.log2(ED)))-1}:0] load_ln_dim;')
    lines.append(f'  reg signed [{DW-1}:0] load_ln_data;')
    lines.append(f'  reg load_attn_weight_en;')
    lines.append(f'  reg [1:0] load_attn_matrix_sel;')
    lines.append(f'  reg [{int(np.ceil(np.log2(ED)))-1}:0] load_attn_row, load_attn_col;')
    lines.append(f'  reg signed [{DW-1}:0] load_attn_data;')
    lines.append(f'  reg load_ffn_weight_en;')
    lines.append(f'  reg load_ffn_layer_sel, load_ffn_is_bias;')
    lines.append(f'  reg [{ffn_bits-1}:0] load_ffn_row, load_ffn_col;')
    lines.append(f'  reg signed [{DW-1}:0] load_ffn_data;')
    lines.append(f'')
    lines.append(f'  // Output')
    lines.append(f'  wire [{int(np.ceil(np.log2(VS)))-1}:0] token_out;')
    lines.append(f'  wire [{ED*DW-1}:0] logits_out;')
    lines.append(f'  wire valid_out;')
    lines.append(f'')
    lines.append(f'  integer cycle_count;')
    lines.append(f'  integer i;')
    lines.append(f'')
    
    # DUT instantiation
    lines.append(f'  gpt2_engine #(')
    lines.append(f'    .VOCAB_SIZE(VOCAB_SIZE), .MAX_SEQ_LEN(MAX_SEQ_LEN),')
    lines.append(f'    .EMBED_DIM(EMBED_DIM), .NUM_HEADS(NUM_HEADS),')
    lines.append(f'    .HEAD_DIM(HEAD_DIM), .FFN_DIM(FFN_DIM),')
    lines.append(f'    .NUM_LAYERS(NUM_LAYERS), .DATA_WIDTH(DATA_WIDTH)')
    lines.append(f'  ) dut (')
    lines.append(f'    .clk(clk), .rst(rst),')
    lines.append(f'    .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),')
    lines.append(f'    .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),')
    lines.append(f'    .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),')
    lines.append(f'    .load_ln_en(load_ln_en), .load_layer_idx(load_layer_idx),')
    lines.append(f'    .load_ln_sel(load_ln_sel), .load_ln_is_gamma(load_ln_is_gamma),')
    lines.append(f'    .load_ln_dim(load_ln_dim), .load_ln_data(load_ln_data),')
    lines.append(f'    .load_attn_weight_en(load_attn_weight_en),')
    lines.append(f'    .load_attn_matrix_sel(load_attn_matrix_sel),')
    lines.append(f'    .load_attn_row(load_attn_row), .load_attn_col(load_attn_col),')
    lines.append(f'    .load_attn_data(load_attn_data),')
    lines.append(f'    .load_ffn_weight_en(load_ffn_weight_en),')
    lines.append(f'    .load_ffn_layer_sel(load_ffn_layer_sel),')
    lines.append(f'    .load_ffn_is_bias(load_ffn_is_bias),')
    lines.append(f'    .load_ffn_row(load_ffn_row), .load_ffn_col(load_ffn_col),')
    lines.append(f'    .load_ffn_data(load_ffn_data),')
    lines.append(f'    .valid_in(valid_in), .token_in(token_in),')
    lines.append(f'    .position_in(position_in),')
    lines.append(f'    .token_out(token_out), .logits_out(logits_out),')
    lines.append(f'    .valid_out(valid_out),')
    lines.append(f'    .total_zero_skips(), .total_cycles()')
    lines.append(f'  );')
    lines.append(f'')
    
    # Clock
    lines.append(f'  always #5 clk = ~clk;')
    lines.append(f'')
    
    # Dump waveform
    lines.append(f'  initial begin')
    lines.append(f'    $dumpfile("cosim.vcd");')
    lines.append(f'    $dumpvars(0, cosim_tb);')
    lines.append(f'  end')
    lines.append(f'')
    
    # Main test
    lines.append(f'  initial begin')
    lines.append(f'    clk = 0; rst = 1;')
    lines.append(f'    valid_in = 0; load_token_emb = 0; load_pos_emb = 0;')
    lines.append(f'    token_in = 0; position_in = 0;')
    lines.append(f'    load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0;')
    lines.append(f'    load_pos_idx = 0;')
    lines.append(f'    load_ln_en = 0; load_layer_idx = 0; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = 0; load_ln_data = 0;')
    lines.append(f'    load_attn_weight_en = 0; load_attn_matrix_sel = 0; load_attn_row = 0; load_attn_col = 0; load_attn_data = 0;')
    lines.append(f'    load_ffn_weight_en = 0; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = 0; load_ffn_col = 0; load_ffn_data = 0;')
    lines.append(f'')
    
    # Load weights (using real weights)
    lines.append(f'    // ===== REAL GPT-2 WEIGHTS (Q8.8) =====')
    lines.append(f'')
    
    # Reset
    lines.append(f'    #35 rst = 0;')
    lines.append(f'    #25;')
    lines.append(f'')

    # Load LayerNorm banks for each layer
    for layer in range(NUM_LAYERS):
        for dim in range(ED):
            ln1g = int(weights["ln1_gamma"][dim]) & 0xFFFF
            ln1b = int(weights["ln1_beta"][dim]) & 0xFFFF
            ln2g = int(weights["ln2_gamma"][dim]) & 0xFFFF
            ln2b = int(weights["ln2_beta"][dim]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {layer}; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = {dim}; load_ln_data = 16\'h{ln1g:04x};')
            lines.append(f'    @(negedge clk); load_ln_en = 0;')
            lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {layer}; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = {dim}; load_ln_data = 16\'h{ln1b:04x};')
            lines.append(f'    @(negedge clk); load_ln_en = 0;')
            lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {layer}; load_ln_sel = 1; load_ln_is_gamma = 1; load_ln_dim = {dim}; load_ln_data = 16\'h{ln2g:04x};')
            lines.append(f'    @(negedge clk); load_ln_en = 0;')
            lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {layer}; load_ln_sel = 1; load_ln_is_gamma = 0; load_ln_dim = {dim}; load_ln_data = 16\'h{ln2b:04x};')
            lines.append(f'    @(negedge clk); load_ln_en = 0;')

    # Load final LayerNorm (layer index == NUM_LAYERS)
    for dim in range(ED):
        lnfg = int(weights["ln_final_gamma"][dim]) & 0xFFFF
        lnfb = int(weights["ln_final_beta"][dim]) & 0xFFFF
        lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {NUM_LAYERS}; load_ln_sel = 0; load_ln_is_gamma = 1; load_ln_dim = {dim}; load_ln_data = 16\'h{lnfg:04x};')
        lines.append(f'    @(negedge clk); load_ln_en = 0;')
        lines.append(f'    @(negedge clk); load_ln_en = 1; load_layer_idx = {NUM_LAYERS}; load_ln_sel = 0; load_ln_is_gamma = 0; load_ln_dim = {dim}; load_ln_data = 16\'h{lnfb:04x};')
        lines.append(f'    @(negedge clk); load_ln_en = 0;')

    # Load attention matrices
    for row in range(ED):
        for col in range(ED):
            wq = int(weights["wq"][row, col]) & 0xFFFF
            wk = int(weights["wk"][row, col]) & 0xFFFF
            wv = int(weights["wv"][row, col]) & 0xFFFF
            wo = int(weights["wo"][row, col]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d0; load_attn_row = {row}; load_attn_col = {col}; load_attn_data = 16\'h{wq:04x};')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 0;')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d1; load_attn_row = {row}; load_attn_col = {col}; load_attn_data = 16\'h{wk:04x};')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 0;')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d2; load_attn_row = {row}; load_attn_col = {col}; load_attn_data = 16\'h{wv:04x};')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 0;')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 1; load_attn_matrix_sel = 2\'d3; load_attn_row = {row}; load_attn_col = {col}; load_attn_data = 16\'h{wo:04x};')
            lines.append(f'    @(negedge clk); load_attn_weight_en = 0;')

    # Load FFN W1 + b1
    for row in range(ED):
        for col in range(FD):
            fw1 = int(weights["ffn_w1"][row, col]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 0; load_ffn_row = {row}; load_ffn_col = {col}; load_ffn_data = 16\'h{fw1:04x};')
            lines.append(f'    @(negedge clk); load_ffn_weight_en = 0;')
    for col in range(FD):
        fb1 = int(weights["ffn_b1"][col]) & 0xFFFF
        lines.append(f'    @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 0; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = {col}; load_ffn_data = 16\'h{fb1:04x};')
        lines.append(f'    @(negedge clk); load_ffn_weight_en = 0;')

    # Load FFN W2 + b2
    for row in range(FD):
        for col in range(ED):
            fw2 = int(weights["ffn_w2"][row, col]) & 0xFFFF
            lines.append(f'    @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 0; load_ffn_row = {row}; load_ffn_col = {col}; load_ffn_data = 16\'h{fw2:04x};')
            lines.append(f'    @(negedge clk); load_ffn_weight_en = 0;')
    for col in range(ED):
        fb2 = int(weights["ffn_b2"][col]) & 0xFFFF
        lines.append(f'    @(negedge clk); load_ffn_weight_en = 1; load_ffn_layer_sel = 1; load_ffn_is_bias = 1; load_ffn_row = 0; load_ffn_col = {col}; load_ffn_data = 16\'h{fb2:04x};')
        lines.append(f'    @(negedge clk); load_ffn_weight_en = 0;')

    lines.append(f'')
    
    # Load token embeddings
    lines.append(f'    // ===== LOAD TOKEN EMBEDDINGS =====')
    for tid in range(VS):
        for dim in range(ED):
            val = int(weights['token_emb'][tid, dim]) & 0xFFFF
            lines.append(f'    @(negedge clk);')
            lines.append(f'    load_token_emb = 1; load_token_idx = {tid}; load_dim_idx = {dim}; load_emb_data = 16\'h{val:04x};')
            lines.append(f'    @(negedge clk);')
            lines.append(f'    load_token_emb = 0;')
    
    lines.append(f'')
    
    # Load position embeddings
    lines.append(f'    // ===== LOAD POSITION EMBEDDINGS =====')
    for pid in range(MSL):
        for dim in range(ED):
            val = int(weights['pos_emb'][pid, dim]) & 0xFFFF
            lines.append(f'    @(negedge clk);')
            lines.append(f'    load_pos_emb = 1; load_pos_idx = {pid}; load_dim_idx = {dim}; load_emb_data = 16\'h{val:04x};')
            lines.append(f'    @(negedge clk);')
            lines.append(f'    load_pos_emb = 0;')
    
    lines.append(f'')
    lines.append(f'    #20;')
    lines.append(f'')
    
    # Banner
    lines.append(f'    $display("");')
    lines.append(f'    $display("+=========================================================+");')
    lines.append(f'    $display("|  BitbyBit GPU -- Real GPT-2 Weight Cosimulation          |");')
    lines.append(f'    $display("|  Source: HuggingFace openai-community/gpt2               |");')
    lines.append(f'    $display("|  Weights: Q8.8 quantized (first {ED} dims extracted)       |");')
    lines.append(f'    $display("+=========================================================+");')
    lines.append(f'    $display("");');
    lines.append(f'')
    
    # Run inference for each test token
    for token_id in test_tokens:
        lines.append(f'    // ===== INFERENCE: Token {token_id} =====')
        lines.append(f'    @(negedge clk);')
        lines.append(f'    token_in = {token_id}; position_in = 0;')
        lines.append(f'    valid_in = 1;')
        lines.append(f'    @(negedge clk);')
        lines.append(f'    valid_in = 0;')
        lines.append(f'')
        lines.append(f'    cycle_count = 0;')
        lines.append(f'    while (!valid_out && cycle_count < 500) begin')
        lines.append(f'      @(negedge clk);')
        lines.append(f'      cycle_count = cycle_count + 1;')
        lines.append(f'    end')
        lines.append(f'')
        lines.append(f'    if (valid_out) begin')
        lines.append(f'      $display("COSIM_RESULT token={token_id} predicted=%0d cycles=%0d logits_hex=%h",')
        lines.append(f'               token_out, cycle_count, logits_out);')
        
        # Print individual logit values
        for i in range(ED):
            lines.append(f'      $display("COSIM_LOGIT token={token_id} dim={i} value=%h",')
            lines.append(f'               logits_out[{i*DW} +: {DW}]);')
        
        lines.append(f'    end else begin')
        lines.append(f'      $display("COSIM_ERROR token={token_id} TIMEOUT");')
        lines.append(f'    end')
        lines.append(f'')
        lines.append(f'    // Wait between inferences')
        lines.append(f'    repeat(10) @(negedge clk);')
        lines.append(f'')
    
    lines.append(f'    $display("");')
    lines.append(f'    $display("COSIM_DONE");')
    lines.append(f'    $finish;')
    lines.append(f'  end')
    lines.append(f'')
    lines.append(f'endmodule')
    
    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write('\n'.join(lines) + '\n')
    
    return output_path

def pack_hex(values):
    """Pack Q8.8 values into a hex string (MSB-first for Verilog literal)."""
    result = ""
    for val in reversed(values):  # MSB first 
        result += f"{int(val) & 0xFFFF:04x}"
    return result

# ============================================================================
# Run Verilog simulation
# ============================================================================

def compile_and_run(tb_path, root_dir):
    """Compile with iverilog and run with vvp."""
    
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
    
    build_dir = os.path.join(root_dir, "tb", "cocotb", "sim_build")
    os.makedirs(build_dir, exist_ok=True)
    out_bin = os.path.join(build_dir, "cosim_tb")

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
    
    # Compile
    print("  Compiling GPU + cosimulation testbench...")
    cmd = [IVERILOG, "-g2012", "-o", out_bin, "-s", "cosim_tb"] + sources
    run_checked(cmd, "Compilation")
    print("  Compilation successful!")
    
    # Run
    print("  Running Verilog simulation...")
    result = run_checked([VVP, out_bin], "Simulation", cwd=build_dir, timeout=60)
    return result.stdout

def parse_verilog_output(output):
    """Parse COSIM_RESULT and COSIM_LOGIT lines from Verilog output."""
    results = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        if line.startswith("COSIM_RESULT"):
            parts = {}
            for part in line.split()[1:]:
                key, val = part.split('=')
                parts[key] = val
            
            token = int(parts['token'])
            predicted = int(parts['predicted'])
            cycles = int(parts['cycles'])
            results[token] = {'predicted': predicted, 'cycles': cycles, 'logits': []}
        
        elif line.startswith("COSIM_LOGIT"):
            parts = {}
            for part in line.split()[1:]:
                key, val = part.split('=')
                parts[key] = val
            
            token = int(parts['token'])
            dim = int(parts['dim'])
            hex_val = parts['value']
            
            # Convert hex to Q8.8 float
            int_val = int(hex_val, 16)
            if int_val >= 32768:
                int_val -= 65536
            float_val = int_val / 256.0
            
            if token in results:
                results[token]['logits'].append(float_val)
    
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run GPT-2 cosimulation")
    parser.add_argument("--tokens", type=int, nargs='+', default=[0, 3, 5, 7, 10],
                        help="Token IDs to test")
    parser.add_argument("--embed-dim", type=int, default=4)
    args = parser.parse_args()
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    print()
    print("+" + "=" * 58 + "+")
    print("|  BitbyBit GPU -- Python-Verilog Cosimulation              |")
    print("|  Loading REAL GPT-2 weights into custom Verilog GPU      |")
    print("+" + "=" * 58 + "+")
    print()
    
    # Step 1: Extract weights
    cache_dir = os.path.join(root_dir, "weights", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    weight_dir = os.path.join(root_dir, "weights", "gpt2_real")
    npz_file = os.path.join(weight_dir, "gpt2_q88_weights.npz")
    
    if os.path.exists(npz_file):
        print("  Loading cached Q8.8 weights...")
        weights = dict(np.load(npz_file, allow_pickle=True))
    else:
        print("  Extracting fresh weights...")
        raw = download_gpt2_weights(cache_dir)
        weights = extract_for_bitbybit(raw, output_dir=weight_dir)
    
    # Step 2: Generate testbench
    tb_path = os.path.join(root_dir, "tb", "cocotb", "sim_build", "cosim_tb.v")
    os.makedirs(os.path.dirname(tb_path), exist_ok=True)
    
    print(f"  Generating testbench for tokens: {args.tokens}")
    generate_cosim_testbench(weights, args.tokens, tb_path)
    print(f"  Generated {tb_path}")
    
    # Step 3: Compile and run Verilog
    try:
        output = compile_and_run(tb_path, root_dir)
    except RuntimeError as err:
        print(f"  Simulation failed: {err}")
        sys.exit(1)
    
    # Step 4: Parse results
    verilog_results = parse_verilog_output(output)
    
    # Step 5: Run float32 reference
    print()
    print("=" * 60)
    print("                  COSIMULATION RESULTS")
    print("=" * 60)
    print()
    print(f"  {'Token':>5} | {'Verilog Pred':>12} | {'Float32 Pred':>12} | {'Match':>5} | {'MSE':>8} | {'Cycles':>6}")
    print(f"  {'-'*5:>5} | {'-'*12:>12} | {'-'*12:>12} | {'-'*5:>5} | {'-'*8:>8} | {'-'*6:>6}")
    
    # Load raw weights for float32 reference
    raw_npz = os.path.join(cache_dir, "gpt2_weights.npz")
    raw_weights = dict(np.load(raw_npz, allow_pickle=True)) if os.path.exists(raw_npz) else None
    
    total_mse = 0
    matches = 0
    total = 0
    
    for token_id in args.tokens:
        if token_id not in verilog_results:
            print(f"  {token_id:>5} | {'TIMEOUT':>12} | {'-':>12} | {'':>5} | {'':>8} | {'-':>6}")
            continue
        
        vr = verilog_results[token_id]
        v_logits = np.array(vr['logits'])
        v_pred = vr['predicted']
        cycles = vr['cycles']
        
        if raw_weights is not None:
            ref_logits = run_float32_reference(raw_weights, token_id, 0,
                                                EMBED_DIM, FFN_DIM)
            ref_pred = int(np.argmax(ref_logits))
            mse = np.mean((ref_logits - v_logits) ** 2)
            match = "YES" if v_pred == ref_pred else "NO"
            if v_pred == ref_pred:
                matches += 1
            total_mse += mse
        else:
            ref_pred = "-"
            mse = "-"
            match = "-"
        
        total += 1
        
        mse_str = f"{mse:.4f}" if isinstance(mse, float) else mse
        print(f"  {token_id:>5} | {v_pred:>12} | {str(ref_pred):>12} | {match:>5} | {mse_str:>8} | {cycles:>6}")
    
    print(f"  {'-'*5:>5} | {'-'*12:>12} | {'-'*12:>12} | {'-'*5:>5} | {'-'*8:>8} | {'-'*6:>6}")
    
    if total > 0 and raw_weights is not None:
        avg_mse = total_mse / total
        print(f"  {'':>5} | {'':>12} | {'':>12} | {matches}/{total:>3} | {avg_mse:>8.4f} |")
    
    print()
    
    # Detailed logit comparison
    print("  DETAILED LOGIT COMPARISON:")
    print()
    for token_id in args.tokens:
        if token_id not in verilog_results:
            continue
        vr = verilog_results[token_id]
        v_logits = vr['logits']
        
        if raw_weights is not None:
            ref_logits = run_float32_reference(raw_weights, token_id, 0,
                                                EMBED_DIM, FFN_DIM)
            print(f"  Token {token_id}:")
            for i in range(EMBED_DIM):
                diff = abs(v_logits[i] - ref_logits[i])
                print(f"    dim[{i}]: Verilog={v_logits[i]:+8.4f}  Float32={ref_logits[i]:+8.4f}  diff={diff:.4f}")
            print()
    
    # Print raw Verilog output
    print("=" * 60)
    print("  RAW VERILOG OUTPUT:")
    print("=" * 60)
    for line in output.split('\n'):
        if line.strip():
            print(f"  {line.rstrip()}")
    print()

if __name__ == "__main__":
    main()
