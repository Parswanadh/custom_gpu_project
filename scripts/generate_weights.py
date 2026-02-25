#!/usr/bin/env python3
"""
generate_weights.py — Generate weights for the custom GPU GPT-2 engine.

Creates Verilog-compatible weight files in Q8.8 fixed-point format.
Can generate:
  - Random weights (for testing)
  - Identity weights (for pipeline verification)
  - Weights converted from a PyTorch GPT-2 model (requires transformers)

Output: Verilog $readmemh-compatible hex files.
"""
import struct
import numpy as np
import argparse
import os

def float_to_q88(val):
    """Convert float to Q8.8 fixed-point (16-bit signed)."""
    q = int(round(val * 256))
    q = max(-32768, min(32767, q))  # Clamp to 16-bit signed
    if q < 0:
        q = q + 65536  # Two's complement for hex output
    return q

def q88_to_hex(val):
    """Convert Q8.8 value to 4-digit hex string."""
    return f"{val:04x}"

def generate_identity_weights(embed_dim, ffn_dim, num_layers, output_dir):
    """Generate identity-like weights for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Identity matrix (EMBED_DIM × EMBED_DIM) for Q/K/V/O projections
    identity = np.eye(embed_dim)
    
    for layer in range(num_layers):
        prefix = f"layer{layer}"
        
        # Attention weights: Wq, Wk, Wv, Wo — all identity
        for name in ['wq', 'wk', 'wv', 'wo']:
            with open(os.path.join(output_dir, f"{prefix}_{name}.hex"), 'w') as f:
                for i in range(embed_dim):
                    for j in range(embed_dim):
                        f.write(q88_to_hex(float_to_q88(identity[i][j])) + '\n')
        
        # LayerNorm gamma=1, beta=0
        for ln_name in ['ln1', 'ln2']:
            with open(os.path.join(output_dir, f"{prefix}_{ln_name}_gamma.hex"), 'w') as f:
                for i in range(embed_dim):
                    f.write(q88_to_hex(float_to_q88(1.0)) + '\n')
            with open(os.path.join(output_dir, f"{prefix}_{ln_name}_beta.hex"), 'w') as f:
                for i in range(embed_dim):
                    f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
        
        # FFN W1: EMBED_DIM × FFN_DIM (identity in first EMBED_DIM cols)
        with open(os.path.join(output_dir, f"{prefix}_ffn_w1.hex"), 'w') as f:
            for i in range(embed_dim):
                for j in range(ffn_dim):
                    val = 1.0 if i == j else 0.0
                    f.write(q88_to_hex(float_to_q88(val)) + '\n')
        
        # FFN B1: zeros
        with open(os.path.join(output_dir, f"{prefix}_ffn_b1.hex"), 'w') as f:
            for j in range(ffn_dim):
                f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
        
        # FFN W2: FFN_DIM × EMBED_DIM (identity in first EMBED_DIM rows)
        with open(os.path.join(output_dir, f"{prefix}_ffn_w2.hex"), 'w') as f:
            for i in range(ffn_dim):
                for j in range(embed_dim):
                    val = 1.0 if i == j else 0.0
                    f.write(q88_to_hex(float_to_q88(val)) + '\n')
        
        # FFN B2: zeros
        with open(os.path.join(output_dir, f"{prefix}_ffn_b2.hex"), 'w') as f:
            for j in range(embed_dim):
                f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
    
    # Final layer norm
    with open(os.path.join(output_dir, "ln_final_gamma.hex"), 'w') as f:
        for i in range(embed_dim):
            f.write(q88_to_hex(float_to_q88(1.0)) + '\n')
    with open(os.path.join(output_dir, "ln_final_beta.hex"), 'w') as f:
        for i in range(embed_dim):
            f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
    
    print(f"Generated identity weights in {output_dir}/")


def generate_random_weights(embed_dim, ffn_dim, vocab_size, num_layers, output_dir):
    """Generate random weights scaled for Q8.8."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    scale = 0.02  # Small initialization like GPT-2
    
    # Token embeddings
    with open(os.path.join(output_dir, "token_embeddings.hex"), 'w') as f:
        for token_id in range(vocab_size):
            for dim in range(embed_dim):
                val = np.random.randn() * scale
                f.write(q88_to_hex(float_to_q88(val)) + '\n')
    
    # Position embeddings
    with open(os.path.join(output_dir, "position_embeddings.hex"), 'w') as f:
        for pos in range(1024):  # Max sequence length
            for dim in range(embed_dim):
                val = np.random.randn() * scale
                f.write(q88_to_hex(float_to_q88(val)) + '\n')
    
    for layer in range(num_layers):
        prefix = f"layer{layer}"
        
        for name in ['wq', 'wk', 'wv', 'wo']:
            with open(os.path.join(output_dir, f"{prefix}_{name}.hex"), 'w') as f:
                for i in range(embed_dim * embed_dim):
                    val = np.random.randn() * scale
                    f.write(q88_to_hex(float_to_q88(val)) + '\n')
        
        for ln_name in ['ln1', 'ln2']:
            with open(os.path.join(output_dir, f"{prefix}_{ln_name}_gamma.hex"), 'w') as f:
                for i in range(embed_dim):
                    f.write(q88_to_hex(float_to_q88(1.0)) + '\n')
            with open(os.path.join(output_dir, f"{prefix}_{ln_name}_beta.hex"), 'w') as f:
                for i in range(embed_dim):
                    f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
        
        with open(os.path.join(output_dir, f"{prefix}_ffn_w1.hex"), 'w') as f:
            for i in range(embed_dim * ffn_dim):
                val = np.random.randn() * scale
                f.write(q88_to_hex(float_to_q88(val)) + '\n')
        
        with open(os.path.join(output_dir, f"{prefix}_ffn_b1.hex"), 'w') as f:
            for j in range(ffn_dim):
                f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
        
        with open(os.path.join(output_dir, f"{prefix}_ffn_w2.hex"), 'w') as f:
            for i in range(ffn_dim * embed_dim):
                val = np.random.randn() * scale
                f.write(q88_to_hex(float_to_q88(val)) + '\n')
        
        with open(os.path.join(output_dir, f"{prefix}_ffn_b2.hex"), 'w') as f:
            for j in range(embed_dim):
                f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
    
    with open(os.path.join(output_dir, "ln_final_gamma.hex"), 'w') as f:
        for i in range(embed_dim):
            f.write(q88_to_hex(float_to_q88(1.0)) + '\n')
    with open(os.path.join(output_dir, "ln_final_beta.hex"), 'w') as f:
        for i in range(embed_dim):
            f.write(q88_to_hex(float_to_q88(0.0)) + '\n')
    
    print(f"Generated random weights in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights for custom GPU GPT-2")
    parser.add_argument("--mode", choices=["identity", "random"], default="identity")
    parser.add_argument("--embed-dim", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="weights")
    args = parser.parse_args()
    
    if args.mode == "identity":
        generate_identity_weights(args.embed_dim, args.ffn_dim, args.num_layers, args.output_dir)
    else:
        generate_random_weights(args.embed_dim, args.ffn_dim, args.vocab_size, args.num_layers, args.output_dir)
