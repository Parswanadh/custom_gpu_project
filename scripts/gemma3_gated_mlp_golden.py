import os
import numpy as np
import math

# Q8.8 fixed point config
FRACTIONAL_BITS = 8
SCALE = 1 << FRACTIONAL_BITS

def float_to_q88(val):
    # Clamp to 16-bit signed integer range for Q8.8
    q = int(round(val * SCALE))
    if q > 32767: q = 32767
    if q < -32768: q = -32768
    return q

def gelu(x):
    # Standard GELU function
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

def main():
    np.random.seed(42)

    N = 4 # Batch size or Sequence length
    D_in = 8
    D_hidden = 16
    D_out = 8

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Inputs in Q8.8 (using range -1.0 to 1.0)
    x_float = np.random.uniform(-1.0, 1.0, (N, D_in))
    x_q88 = np.vectorize(float_to_q88)(x_float)
    x = x_q88 / SCALE

    # Gate Weights: strictly ternary (-1, 0, 1)
    gate_w = np.random.choice([-1, 0, 1], size=(D_in, D_hidden))
    
    # Up Weights: INT8 (-127 to 127)
    up_w = np.random.randint(-127, 128, size=(D_in, D_hidden))
    
    # Down Weights: INT8 (-127 to 127)
    down_w = np.random.randint(-127, 128, size=(D_hidden, D_out))

    # MLP Computation
    # 1. Gate Projection
    gate_proj = np.matmul(x, gate_w)
    
    # 2. GELU on Gate
    gate_gelu = np.vectorize(gelu)(gate_proj)
    
    # 3. Up Projection
    up_proj = np.matmul(x, up_w)
    
    # 4. Element-wise multiplication of GELU(Gate) and Up
    gated_up = gate_gelu * up_proj
    
    # 5. Down Projection
    output = np.matmul(gated_up, down_w)

    # Convert Output to Q8.8
    out_q88 = np.vectorize(float_to_q88)(output)

    # Save to Hex files
    input_hex_path = os.path.join(script_dir, 'mlp_input.hex')
    with open(input_hex_path, 'w') as f:
        for row in x_q88:
            for val in row:
                f.write(f"{(val & 0xFFFF):04x}\n")

    weights_hex_path = os.path.join(script_dir, 'mlp_weights.hex')
    with open(weights_hex_path, 'w') as f:
        # Gate weights
        for row in gate_w:
            for val in row:
                f.write(f"{(int(val) & 0xFF):02x}\n")
        # Up weights
        for row in up_w:
            for val in row:
                f.write(f"{(int(val) & 0xFF):02x}\n")
        # Down weights
        for row in down_w:
            for val in row:
                f.write(f"{(int(val) & 0xFF):02x}\n")

    output_hex_path = os.path.join(script_dir, 'mlp_output.hex')
    with open(output_hex_path, 'w') as f:
        for row in out_q88:
            for val in row:
                f.write(f"{(val & 0xFFFF):04x}\n")

    print("PASS: Gemma-3 Gated MLP golden model generated hex files.")
    print(f"Generated: {input_hex_path}")
    print(f"Generated: {weights_hex_path}")
    print(f"Generated: {output_hex_path}")

if __name__ == '__main__':
    main()