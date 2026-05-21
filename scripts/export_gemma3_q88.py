#!/usr/bin/env python3
"""
Export Gemma-3 270M (stub) weights to Q8.8 format.
This script loads a dummy/stub Gemma-3 model using transformers,
iterates over the state_dict, quantizes weights to Q8.8,
and saves them to binary files.
"""

import os
import struct
import numpy as np

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print("PyTorch and/or transformers not found. Please install them:")
    print("pip install torch transformers")
    exit(1)

def float_to_q88(val):
    """Convert float to signed Q8.8 fixed-point integer."""
    q = int(round(float(val) * 256.0))
    return max(-32768, min(32767, q))

def q88_hex(val):
    """Format as 4-character hex string for Verilog $readmemh."""
    return f"{val & 0xFFFF:04x}"

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    weights_dir = os.path.join(project_dir, 'weights', 'gemma3_q88')
    os.makedirs(weights_dir, exist_ok=True)
    
    print("Initializing dummy/stub Gemma-3 270M model...")
    # Use Gemma config as a stub (approximate 270M params)
    try:
        config = AutoConfig.from_pretrained("google/gemma-2b")
    except Exception:
        # Fallback config if not found
        config = AutoConfig.for_model("llama")
        
    # Scale down config for ~270M parameters to match the stub needs
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.num_hidden_layers = 12
    config.vocab_size = 256000 # Typical large vocab size for Gemma
    
    # Initialize dummy weights
    model = AutoModelForCausalLM.from_config(config)
    state_dict = model.state_dict()
    
    print(f"Exporting quantized weights to {weights_dir} ...")
    
    total_sq_error = 0.0
    total_elements = 0
    
    # Save state dict
    for name, tensor in state_dict.items():
        # Cast to float32 before converting to numpy to handle bfloat16
        flat_tensor = tensor.detach().cpu().to(torch.float32).flatten().numpy()
        
        # Clean up name for file paths
        safe_name = name.replace('.', '_')
        bin_path = os.path.join(weights_dir, f"{safe_name}.bin")
        hex_path = os.path.join(weights_dir, f"{safe_name}.hex")
        
        with open(bin_path, "wb") as f_bin, open(hex_path, "w") as f_hex:
            for val in flat_tensor:
                q_val = float_to_q88(val)
                # Pack as little-endian 16-bit signed integer
                f_bin.write(struct.pack('<h', q_val))
                # Write hex string for Verilog $readmemh
                f_hex.write(q88_hex(q_val) + "\n")
                
                # Accumulate MSE
                dequantized_val = q_val / 256.0
                total_sq_error += (val - dequantized_val) ** 2
                total_elements += 1
                
    if total_elements > 0:
        mse = total_sq_error / total_elements
        print(f"Quantization Mean Squared Error (MSE): {mse:.8f}")
    
    print("Export complete. Binary and Hex files generated.")

if __name__ == "__main__":
    main()
