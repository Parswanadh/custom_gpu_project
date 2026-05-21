import os
import math

def float_to_q88(val):
    scaled = int(round(val * 256.0))
    scaled = max(-32768, min(32767, scaled))
    return f"{scaled & 0xFFFF:04x}"

def main():
    dim = 8
    positions = 4
    theta_base = 10000.0
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_qk_hex = []
    output_q_hex = []
    output_k_hex = []

    for pos in range(positions):
        # We need values that fit into Q8.8 safely (range -128.0 to 127.99)
        q = [float((pos + i) * 0.5) for i in range(dim)]
        k = [float((pos - i) * 0.5) for i in range(dim)]
        
        q_out = [0.0] * dim
        k_out = [0.0] * dim
        
        for i in range(0, dim, 2):
            power = i / dim
            theta_i = pos / (theta_base ** power)
            cos_th = math.cos(theta_i)
            sin_th = math.sin(theta_i)
            
            q_out[i]   = q[i] * cos_th - q[i+1] * sin_th
            q_out[i+1] = q[i] * sin_th + q[i+1] * cos_th
            
            k_out[i]   = k[i] * cos_th - k[i+1] * sin_th
            k_out[i+1] = k[i] * sin_th + k[i+1] * cos_th
            
        for i in range(dim):
            # Combine Q and K into a single 32-bit hex word if needed,
            # or just write them as Q8.8. Let's write them concatenated for easier 32-bit readmemh
            # Format: upper 16 bits = Q, lower 16 bits = K
            input_qk_hex.append(f"{float_to_q88(q[i])}{float_to_q88(k[i])}")
            output_q_hex.append(float_to_q88(q_out[i]))
            output_k_hex.append(float_to_q88(k_out[i]))

    with open(os.path.join(out_dir, "rope_input_qk.hex"), "w") as f:
        f.write("\n".join(input_qk_hex) + "\n")

    with open(os.path.join(out_dir, "rope_output_q.hex"), "w") as f:
        f.write("\n".join(output_q_hex) + "\n")

    with open(os.path.join(out_dir, "rope_output_k.hex"), "w") as f:
        f.write("\n".join(output_k_hex) + "\n")
        
    print("PASS: Generated Golden RoPE test vectors successfully.")

if __name__ == "__main__":
    main()
