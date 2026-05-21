import os
import numpy as np

def float_to_q8_8(val):
    q = int(round(val * 256.0))
    q = max(-32768, min(32767, q))
    return q

def q8_8_to_hex(q):
    return f"{(q & 0xFFFF):04x}"

def main():
    np.random.seed(42)
    
    # 8-dimensional vector
    N = 8
    
    # Generate random values, keeping them in a range that doesn't easily overflow Q8.8 when squared
    x_float = np.random.uniform(-4.0, 4.0, N)
    
    # Quantize input to Q8.8
    x_q = [float_to_q8_8(v) for v in x_float]
    
    # Float equivalent of the Q8.8 input
    x_dq = np.array(x_q) / 256.0
    
    # Compute RMSNorm
    epsilon = 1e-6
    variance = np.mean(x_dq**2)
    rms = np.sqrt(variance + epsilon)
    
    x_out_float = x_dq / rms
    
    # Quantize output
    x_out_q = [float_to_q8_8(v) for v in x_out_float]
    
    # Determine output directory (same directory as script, or where it's run)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "rmsnorm_input.hex")
    output_file = os.path.join(script_dir, "rmsnorm_output.hex")
    
    with open(input_file, "w") as f_in, open(output_file, "w") as f_out:
        for xi, xo in zip(x_q, x_out_q):
            f_in.write(f"{q8_8_to_hex(xi)}\n")
            f_out.write(f"{q8_8_to_hex(xo)}\n")
            
    print(f"Generated {input_file}")
    print(f"Generated {output_file}")
    print("PASS: RMSNorm golden vectors generated successfully.")

if __name__ == "__main__":
    main()
