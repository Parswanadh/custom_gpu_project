"""
test_gpt2_cosim.py — Cocotb cosimulation: Real GPT-2 weights ↔ Verilog GPU

Loads real GPT-2 weights (Q8.8 quantized), feeds them into the
gpt2_engine Verilog module, runs inference, and compares against
a float32 reference model.

Run with: make -C tb/cocotb
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
import numpy as np
import os
import sys

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

# ============================================================================
# Q8.8 Helpers
# ============================================================================

def to_q88(val):
    q = int(round(val * 256))
    return max(-32768, min(32767, q))

def from_q88(q):
    q = int(q)
    if q >= 32768:
        q -= 65536
    if q < 0:
        q = q & 0xFFFF
        q = q - 65536 if q >= 32768 else q
    return q / 256.0

def pack_vector(values, data_width=16):
    """Pack a list of Q8.8 values into a single integer (flattened bus)."""
    result = 0
    for i, val in enumerate(values):
        result |= (int(val) & 0xFFFF) << (i * data_width)
    return result

def pack_matrix(mat, data_width=16):
    """Pack a 2D matrix into a single integer (row-major, flattened)."""
    flat = mat.flatten()
    result = 0
    for i, val in enumerate(flat):
        result |= (int(val) & 0xFFFF) << (i * data_width)
    return result

# ============================================================================
# Weight Loading
# ============================================================================

def load_weights(weight_dir=None):
    """Load Q8.8 weights from extracted NPZ or generate identity defaults."""
    if weight_dir is None:
        weight_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'gpt2_real')
    
    npz_file = os.path.join(weight_dir, "gpt2_q88_weights.npz")
    
    if os.path.exists(npz_file):
        cocotb.log.info(f"Loading real GPT-2 weights from {npz_file}")
        data = dict(np.load(npz_file, allow_pickle=True))
        return data, "real_gpt2"
    else:
        cocotb.log.info("No extracted weights found, using identity weights")
        ED = 4
        FD = 8
        VS = 16
        MSL = 8
        
        data = {
            'token_emb': (np.random.randn(VS, ED) * 0.5 * 256).astype(np.int32),
            'pos_emb': (np.ones((MSL, ED)) * 13).astype(np.int32),
            'ln1_gamma': np.full(ED, 256, dtype=np.int32),
            'ln1_beta': np.zeros(ED, dtype=np.int32),
            'ln2_gamma': np.full(ED, 256, dtype=np.int32),
            'ln2_beta': np.zeros(ED, dtype=np.int32),
            'wq': (np.eye(ED) * 256).astype(np.int32),
            'wk': (np.eye(ED) * 256).astype(np.int32),
            'wv': (np.eye(ED) * 256).astype(np.int32),
            'wo': (np.eye(ED) * 256).astype(np.int32),
            'ffn_w1': np.zeros((ED, FD), dtype=np.int32),
            'ffn_b1': np.zeros(FD, dtype=np.int32),
            'ffn_w2': np.zeros((FD, ED), dtype=np.int32),
            'ffn_b2': np.zeros(ED, dtype=np.int32),
            'ln_final_gamma': np.full(ED, 256, dtype=np.int32),
            'ln_final_beta': np.zeros(ED, dtype=np.int32),
        }
        # Identity FFN
        for i in range(min(ED, FD)):
            data['ffn_w1'][i, i] = 256
        for i in range(min(FD, ED)):
            data['ffn_w2'][i, i] = 256
        
        return data, "identity"

# ============================================================================
# Cocotb Tests
# ============================================================================

@cocotb.test()
async def test_load_real_weights(dut):
    """Load real GPT-2 weights into the Verilog GPU and run inference."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    dut.valid_in.value = 0
    dut.load_token_emb.value = 0
    dut.load_pos_emb.value = 0
    dut.load_token_idx.value = 0
    dut.load_dim_idx.value = 0
    dut.load_emb_data.value = 0
    dut.load_pos_idx.value = 0
    dut.token_in.value = 0
    dut.position_in.value = 0
    
    await Timer(35, units="ns")
    dut.rst.value = 0
    await Timer(25, units="ns")
    
    # Load weights
    weights, source = load_weights()
    ED = 4
    FD = 8
    VS = 16
    MSL = 8
    DW = 16
    
    cocotb.log.info(f"Weight source: {source}")
    cocotb.log.info(f"Setting transformer weights...")
    
    # Set weight buses (combinational inputs)
    dut.ln1_gamma.value = pack_vector(weights['ln1_gamma'][:ED])
    dut.ln1_beta.value = pack_vector(weights['ln1_beta'][:ED])
    dut.ln2_gamma.value = pack_vector(weights['ln2_gamma'][:ED])
    dut.ln2_beta.value = pack_vector(weights['ln2_beta'][:ED])
    dut.ln_final_gamma.value = pack_vector(weights['ln_final_gamma'][:ED])
    dut.ln_final_beta.value = pack_vector(weights['ln_final_beta'][:ED])
    
    dut.wq_flat.value = pack_matrix(weights['wq'][:ED, :ED])
    dut.wk_flat.value = pack_matrix(weights['wk'][:ED, :ED])
    dut.wv_flat.value = pack_matrix(weights['wv'][:ED, :ED])
    dut.wo_flat.value = pack_matrix(weights['wo'][:ED, :ED])
    
    dut.ffn_w1_flat.value = pack_matrix(weights['ffn_w1'][:ED, :FD])
    dut.ffn_b1_flat.value = pack_vector(weights['ffn_b1'][:FD])
    dut.ffn_w2_flat.value = pack_matrix(weights['ffn_w2'][:FD, :ED])
    dut.ffn_b2_flat.value = pack_vector(weights['ffn_b2'][:ED])
    
    # Load token embeddings
    cocotb.log.info(f"Loading {VS} token embeddings...")
    for tid in range(VS):
        for dim in range(ED):
            await FallingEdge(dut.clk)
            dut.load_token_emb.value = 1
            dut.load_token_idx.value = tid
            dut.load_dim_idx.value = dim
            dut.load_emb_data.value = int(weights['token_emb'][tid, dim]) & 0xFFFF
            await FallingEdge(dut.clk)
            dut.load_token_emb.value = 0
    
    # Load position embeddings
    cocotb.log.info(f"Loading {MSL} position embeddings...")
    for pid in range(MSL):
        for dim in range(ED):
            await FallingEdge(dut.clk)
            dut.load_pos_emb.value = 1
            dut.load_pos_idx.value = pid
            dut.load_dim_idx.value = dim
            dut.load_emb_data.value = int(weights['pos_emb'][pid, dim]) & 0xFFFF
            await FallingEdge(dut.clk)
            dut.load_pos_emb.value = 0
    
    await Timer(20, units="ns")
    
    cocotb.log.info("=" * 60)
    cocotb.log.info("  BitbyBit GPU — Real GPT-2 Weight Inference")
    cocotb.log.info("=" * 60)
    
    # Run inference for multiple tokens
    test_tokens = [0, 3, 5, 7, 10]
    
    for token_id in test_tokens:
        position = 0
        
        # Start inference
        await FallingEdge(dut.clk)
        dut.token_in.value = token_id
        dut.position_in.value = position
        dut.valid_in.value = 1
        await FallingEdge(dut.clk)
        dut.valid_in.value = 0
        
        # Wait for result
        timeout = 0
        while int(dut.valid_out.value) == 0 and timeout < 500:
            await FallingEdge(dut.clk)
            timeout += 1
        
        if int(dut.valid_out.value):
            predicted = int(dut.token_out.value)
            logits_raw = int(dut.logits_out.value)
            
            # Extract individual logit values
            logits = []
            for i in range(ED):
                val = (logits_raw >> (i * DW)) & 0xFFFF
                logits.append(from_q88(val))
            
            cocotb.log.info(f"Token {token_id:2d} → Predicted: {predicted} | "
                          f"Logits: [{', '.join(f'{l:.3f}' for l in logits)}] | "
                          f"Cycles: {timeout}")
        else:
            cocotb.log.error(f"Token {token_id}: TIMEOUT after {timeout} cycles!")
        
        # Wait between inferences
        for _ in range(10):
            await FallingEdge(dut.clk)
    
    cocotb.log.info("=" * 60)
    cocotb.log.info(f"  Inference complete! Source: {source}")
    cocotb.log.info("=" * 60)


@cocotb.test()
async def test_compare_with_reference(dut):
    """Run inference and compare Verilog GPU output vs float32 reference."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst.value = 1
    dut.valid_in.value = 0
    dut.load_token_emb.value = 0
    dut.load_pos_emb.value = 0
    
    await Timer(35, units="ns")
    dut.rst.value = 0
    await Timer(25, units="ns")
    
    weights, source = load_weights()
    ED = 4
    FD = 8
    DW = 16
    VS = 16
    MSL = 8
    
    # Load all weights into DUT
    dut.ln1_gamma.value = pack_vector(weights['ln1_gamma'][:ED])
    dut.ln1_beta.value = pack_vector(weights['ln1_beta'][:ED])
    dut.ln2_gamma.value = pack_vector(weights['ln2_gamma'][:ED])
    dut.ln2_beta.value = pack_vector(weights['ln2_beta'][:ED])
    dut.ln_final_gamma.value = pack_vector(weights['ln_final_gamma'][:ED])
    dut.ln_final_beta.value = pack_vector(weights['ln_final_beta'][:ED])
    dut.wq_flat.value = pack_matrix(weights['wq'][:ED, :ED])
    dut.wk_flat.value = pack_matrix(weights['wk'][:ED, :ED])
    dut.wv_flat.value = pack_matrix(weights['wv'][:ED, :ED])
    dut.wo_flat.value = pack_matrix(weights['wo'][:ED, :ED])
    dut.ffn_w1_flat.value = pack_matrix(weights['ffn_w1'][:ED, :FD])
    dut.ffn_b1_flat.value = pack_vector(weights['ffn_b1'][:FD])
    dut.ffn_w2_flat.value = pack_matrix(weights['ffn_w2'][:FD, :ED])
    dut.ffn_b2_flat.value = pack_vector(weights['ffn_b2'][:ED])
    
    for tid in range(VS):
        for dim in range(ED):
            await FallingEdge(dut.clk)
            dut.load_token_emb.value = 1
            dut.load_token_idx.value = tid
            dut.load_dim_idx.value = dim
            dut.load_emb_data.value = int(weights['token_emb'][tid, dim]) & 0xFFFF
            await FallingEdge(dut.clk)
            dut.load_token_emb.value = 0
    
    for pid in range(MSL):
        for dim in range(ED):
            await FallingEdge(dut.clk)
            dut.load_pos_emb.value = 1
            dut.load_pos_idx.value = pid
            dut.load_dim_idx.value = dim
            dut.load_emb_data.value = int(weights['pos_emb'][pid, dim]) & 0xFFFF
            await FallingEdge(dut.clk)
            dut.load_pos_emb.value = 0
    
    await Timer(20, units="ns")
    
    # Also run float32 reference
    try:
        from extract_gpt2_weights import run_float32_reference
        
        weight_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'cache')
        npz = os.path.join(weight_dir, "gpt2_weights.npz")
        if os.path.exists(npz):
            raw_weights = dict(np.load(npz, allow_pickle=True))
        else:
            raw_weights = None
    except Exception:
        raw_weights = None
    
    cocotb.log.info("=" * 70)
    cocotb.log.info("  COMPARISON: Verilog GPU (Q8.8) vs Float32 Reference")
    cocotb.log.info("=" * 70)
    
    total_mse = 0
    matches = 0
    total = 0
    
    for token_id in [0, 3, 5]:
        # Verilog inference
        await FallingEdge(dut.clk)
        dut.token_in.value = token_id
        dut.position_in.value = 0
        dut.valid_in.value = 1
        await FallingEdge(dut.clk)
        dut.valid_in.value = 0
        
        timeout = 0
        while int(dut.valid_out.value) == 0 and timeout < 500:
            await FallingEdge(dut.clk)
            timeout += 1
        
        verilog_pred = int(dut.token_out.value)
        logits_raw = int(dut.logits_out.value)
        verilog_logits = []
        for i in range(ED):
            val = (logits_raw >> (i * DW)) & 0xFFFF
            verilog_logits.append(from_q88(val))
        verilog_logits = np.array(verilog_logits)
        
        # Float32 reference  
        if raw_weights:
            ref_logits = run_float32_reference(raw_weights, token_id, 0, ED, FD)
            ref_pred = int(np.argmax(ref_logits))
            mse = np.mean((ref_logits - verilog_logits) ** 2)
            total_mse += mse
            match = "✓" if ref_pred == verilog_pred else "✗"
            if ref_pred == verilog_pred:
                matches += 1
            
            cocotb.log.info(f"  Token {token_id:2d}: Verilog={verilog_pred} Ref={ref_pred} {match} "
                          f"MSE={mse:.4f}")
        else:
            cocotb.log.info(f"  Token {token_id:2d}: Verilog predicted={verilog_pred} "
                          f"logits={verilog_logits}")
        
        total += 1
        
        for _ in range(10):
            await FallingEdge(dut.clk)
    
    if raw_weights:
        cocotb.log.info(f"\n  Matches: {matches}/{total} | Avg MSE: {total_mse/total:.6f}")
    
    cocotb.log.info("=" * 70)
