#!/usr/bin/env python3
"""Quick test: compare zero-skip rates across modes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

cache_dir = os.path.join('weights', 'opt125m')
npz_path = os.path.join(cache_dir, 'opt125m_weights.npz')
raw = np.load(npz_path)
weights = {k: raw[k] for k in raw.files}

# Import the engine classes from chat_opt
import importlib.util
spec = importlib.util.spec_from_file_location("chat_opt", os.path.join(os.path.dirname(__file__), "chat_opt.py"))
mod = importlib.util.module_from_spec(spec)

# We need to prevent main() from running
import types
old_main = None
spec.loader.exec_module(mod)

OPTEngine = mod.OPTEngine
tokens = [2, 464, 499]  # </s>, 'The', 'future'

configs = [
    ("FFN ReLU + Q8.8 (default)", False, True, True),
    ("ReLU EVERYWHERE + Q8.8",    True,  True, True),
    ("No ReLU, Q8.8 only",        False, False, True),
    ("FFN ReLU, no Q8.8",         False, True, False),
]

for label, relu_all, ffn_relu, q88 in configs:
    w = {k: v.copy() for k, v in weights.items()}
    engine = OPTEngine(w, relu_everywhere=relu_all, use_ffn_relu=ffn_relu, use_q88=q88)
    engine.reset()
    for i, tid in enumerate(tokens):
        logits = engine.forward(tid, i)
    s = engine.stats
    relu_pct = s['relu_zeros'] / max(s['relu_total'], 1) * 100
    q88_pct = s['q88_zeros'] / max(s['q88_total'], 1) * 100
    skip_pct = s['zero_mult_skipped'] / max(s['total_mults'], 1) * 100
    boost = 1 / max(1 - skip_pct / 100, 0.01)
    print(f"=== {label} ===")
    print(f"  ReLU sparsity:    {relu_pct:.1f}%")
    print(f"  Q8.8 zeros:       {q88_pct:.1f}%")
    print(f"  Overall zero-skip: {skip_pct:.1f}%")
    print(f"  Boost:             {boost:.2f}x")
    print()
