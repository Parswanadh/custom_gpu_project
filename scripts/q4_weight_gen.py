#!/usr/bin/env python3
"""Q4_0 Weight Quantization and Test Vector Generator.

Implements GGML Q4_0 quantization as a golden reference and generates
test vectors for the 4x4 systolic array, including Verilog $readmemh files.
"""

import os
import sys
import math
import struct

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Q4_0 quantization (golden reference)
# ---------------------------------------------------------------------------

def quantize_q4_0(weights_float, block_size=32):
    """Quantize float weights to Q4_0 format.

    For each block of *block_size* weights:
        scale      = max(abs(weights)) / 7          (maps to [-8, 7])
        zero_point = 8                               (symmetric-ish)
        q_weight   = clamp(round(w / scale) + 8, 0, 15)

    Dequant: w_approx = (q_weight - 8) * scale

    Returns
    -------
    list of dict
        Each dict: {'scale': float, 'zero_point': int,
                    'quants': list[int], 'packed': list[int]}
        *packed* stores pairs of nibbles in one byte (low nibble first).
    """
    flat = _flatten(weights_float)
    # Pad to multiple of block_size
    while len(flat) % block_size != 0:
        flat.append(0.0)

    blocks = []
    for i in range(0, len(flat), block_size):
        block = flat[i:i + block_size]
        amax = max(abs(v) for v in block)
        scale = amax / 7.0 if amax != 0.0 else 1.0
        zero_point = 8
        quants = [_clamp(round(v / scale) + 8, 0, 15) for v in block]
        # Pack two 4-bit values per byte (low nibble first)
        packed = []
        for j in range(0, len(quants), 2):
            lo = quants[j]
            hi = quants[j + 1] if j + 1 < len(quants) else 0
            packed.append((hi << 4) | lo)
        blocks.append({
            'scale': scale,
            'zero_point': zero_point,
            'quants': quants,
            'packed': packed,
        })
    return blocks


def dequantize_q4_0(blocks):
    """Dequantize Q4_0 blocks back to floats."""
    result = []
    for b in blocks:
        scale = b['scale']
        for q in b['quants']:
            result.append((q - 8) * scale)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(m):
    """Flatten a nested list / 2-D matrix into a 1-D list of floats."""
    flat = []
    for row in m:
        if isinstance(row, (list, tuple)):
            flat.extend(float(v) for v in row)
        else:
            flat.append(float(row))
    return flat


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _matmul_4x4(a, b):
    """Multiply 4x4 matrix *a* by 4x4 matrix *b* (lists of lists)."""
    c = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c


def _matvec_4x4(mat, vec):
    """Multiply 4x4 matrix by 4-element vector."""
    result = [0.0] * 4
    for i in range(4):
        for j in range(4):
            result[i] += mat[i][j] * vec[j]
    return result


def _reshape_4x4(flat):
    """Reshape a flat list of >=16 values into a 4x4 matrix."""
    return [flat[i * 4:(i + 1) * 4] for i in range(4)]


def _snr_db(original, reconstructed):
    """Signal-to-noise ratio in dB."""
    sig = sum(v * v for v in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    if noise == 0:
        return float('inf')
    if sig == 0:
        return 0.0
    return 10.0 * math.log10(sig / noise)


def _float_to_fp16_hex(val):
    """Convert a Python float to a 16-bit half-precision hex string."""
    packed = struct.pack('>e', val)
    return f"{packed[0]:02x}{packed[1]:02x}"


def _float_to_fixed16_hex(val, frac_bits=8):
    """Convert float to signed 16-bit fixed-point hex (Qn.frac_bits)."""
    scaled = int(round(val * (1 << frac_bits)))
    if scaled < -32768:
        scaled = -32768
    elif scaled > 32767:
        scaled = 32767
    return f"{scaled & 0xFFFF:04x}"


# ---------------------------------------------------------------------------
# Test matrices
# ---------------------------------------------------------------------------

def build_test_matrices():
    """Return dict of name -> 4x4 float matrix."""
    matrices = {}

    # 1. Identity
    matrices['identity_4x4'] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    # 2. Random in [-1, 1] (deterministic seed)
    if HAS_NUMPY:
        rng = np.random.RandomState(42)
        r = rng.uniform(-1.0, 1.0, (4, 4))
        matrices['random_4x4'] = [[float(r[i][j]) for j in range(4)] for i in range(4)]
    else:
        import random
        random.seed(42)
        matrices['random_4x4'] = [[random.uniform(-1.0, 1.0) for _ in range(4)] for _ in range(4)]

    # 3. Diagonal
    diag_vals = [0.5, -0.3, 0.7, -0.1]
    matrices['diagonal_4x4'] = [[diag_vals[i] if i == j else 0.0 for j in range(4)] for i in range(4)]

    # 4. Known sequential scaled to Q4 range
    # Original [[1..16]], scale into roughly [-1, 1]
    raw = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    max_val = 16.0
    matrices['sequential_4x4'] = [[v / max_val for v in row] for row in raw]

    return matrices


def build_test_activations():
    """Return list of (name, 4-element vector) test activations."""
    activations = [
        ('ones', [1.0, 1.0, 1.0, 1.0]),
        ('unit_0', [1.0, 0.0, 0.0, 0.0]),
        ('ascending', [0.25, 0.5, 0.75, 1.0]),
        ('mixed', [1.0, -0.5, 0.3, -0.7]),
    ]
    return activations


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

def write_hex_file(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def generate_verilog_files(all_results, sim_dir):
    """Write $readmemh files into *sim_dir*."""
    weight_lines = []
    scale_lines = []
    activation_lines = []
    expected_lines = []

    for res in all_results:
        # -- weights: pack 4 nibbles into a 16-bit word per row --
        mat_q = res['quant_matrix']
        for row in mat_q:
            # 4 nibbles -> 16-bit word (MSB first: q[0] in bits[15:12])
            word = 0
            for j, q in enumerate(row):
                word |= (q & 0xF) << (4 * (3 - j))
            weight_lines.append(f"{word:04x}")

        # -- scales: fp16 per block (one block per matrix for 4x4) --
        scale_lines.append(_float_to_fp16_hex(res['scale']))

        # -- activations & expected results --
        for act_name, act_vec, exp_vec in res['test_vectors']:
            for v in act_vec:
                activation_lines.append(_float_to_fixed16_hex(v))
            for v in exp_vec:
                expected_lines.append(_float_to_fixed16_hex(v))

    write_hex_file(os.path.join(sim_dir, 'q4_weights.hex'), weight_lines)
    write_hex_file(os.path.join(sim_dir, 'q4_scales.hex'), scale_lines)
    write_hex_file(os.path.join(sim_dir, 'q4_activations.hex'), activation_lines)
    write_hex_file(os.path.join(sim_dir, 'q4_expected.hex'), expected_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    sim_dir = os.path.join(project_dir, 'sim')

    matrices = build_test_matrices()
    activations = build_test_activations()

    all_results = []

    print("=" * 60)
    print("Q4_0 Quantization Summary:")
    print("=" * 60)

    for name, mat in matrices.items():
        flat_orig = _flatten(mat)
        # Quantize (block_size >= 16 so one block covers the 4x4)
        blocks = quantize_q4_0(mat, block_size=32)
        flat_deq = dequantize_q4_0(blocks)[:16]  # only first 16 values

        deq_mat = _reshape_4x4(flat_deq)
        quant_values = blocks[0]['quants'][:16]
        quant_mat = _reshape_4x4(quant_values)
        scale = blocks[0]['scale']
        zero_point = blocks[0]['zero_point']

        # Error metrics
        errors = [abs(a - b) for a, b in zip(flat_orig, flat_deq)]
        max_err = max(errors)
        mean_err = sum(errors) / len(errors)
        snr = _snr_db(flat_orig, flat_deq)

        print(f"\nMatrix: {name}")
        print(f"  Scale: {scale:.6f}  Zero point: {zero_point}")
        print(f"  Original floats:")
        for row in mat:
            print(f"    {['%8.4f' % v for v in row]}")
        print(f"  Q4 values (0-15):")
        for row in quant_mat:
            print(f"    {row}")
        print(f"  Dequantized floats:")
        for row in deq_mat:
            print(f"    {['%8.4f' % v for v in row]}")
        print(f"  Quantization error per element:")
        err_mat = _reshape_4x4(errors)
        for row in err_mat:
            print(f"    {['%8.5f' % v for v in row]}")
        print(f"  Max abs error: {max_err:.6f}")
        print(f"  Mean abs error: {mean_err:.6f}")
        snr_str = f"{snr:.1f} dB" if snr != float('inf') else "inf dB"
        print(f"  SNR: {snr_str}")

        # Test vectors: matmul with each activation using dequantized weights
        test_vectors = []
        print(f"  Test vectors (dequantized weight @ activation):")
        for act_name, act_vec in activations:
            expected = _matvec_4x4(deq_mat, act_vec)
            test_vectors.append((act_name, act_vec, expected))
            print(f"    {act_name}: act={['%6.3f' % v for v in act_vec]}  "
                  f"=> {['%8.4f' % v for v in expected]}")

        all_results.append({
            'name': name,
            'original': mat,
            'scale': scale,
            'zero_point': zero_point,
            'quant_matrix': quant_mat,
            'deq_matrix': deq_mat,
            'errors': errors,
            'max_err': max_err,
            'mean_err': mean_err,
            'snr': snr,
            'test_vectors': test_vectors,
        })

    # Write Verilog hex files
    generate_verilog_files(all_results, sim_dir)
    print("\n" + "=" * 60)
    print("Generated Verilog $readmemh files:")
    for fname in ['q4_weights.hex', 'q4_scales.hex',
                   'q4_activations.hex', 'q4_expected.hex']:
        fpath = os.path.join(sim_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  {fpath}  ({size} bytes)")
    print("=" * 60)


if __name__ == '__main__':
    main()
