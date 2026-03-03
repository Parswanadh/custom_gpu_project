#!/usr/bin/env python3
"""
run_tests.py - Run all Verilog testbenches for the Custom GPU
Usage: python scripts/run_tests.py
"""
import subprocess, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IVERILOG = r"D:\Tools\iverilog\bin\iverilog.exe"
VVP = r"D:\Tools\iverilog\bin\vvp.exe"

# Check iverilog exists
if not os.path.exists(IVERILOG):
    # Try PATH
    try:
        subprocess.run(["iverilog", "--version"], capture_output=True, check=True)
        IVERILOG = "iverilog"
        VVP = "vvp"
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"[ERROR] iverilog not found at {IVERILOG} or in PATH")
        sys.exit(1)

os.makedirs(os.path.join(ROOT, "sim", "waveforms"), exist_ok=True)

# Test definitions: (phase, name, output_bin, [sources], testbench)
TESTS = [
    # Phase 1: Core Compute Primitives
    ("P1", "zero_detect_mult", "zdm_test",
     ["rtl/primitives/zero_detect_mult.v"], "tb/primitives/zero_detect_mult_tb.v"),
    ("P1", "variable_precision_alu", "vpa_test",
     ["rtl/primitives/variable_precision_alu.v"], "tb/primitives/variable_precision_alu_tb.v"),
    ("P1", "sparse_memory_ctrl", "smc_test",
     ["rtl/primitives/sparse_memory_ctrl.v"], "tb/primitives/sparse_memory_ctrl_tb.v"),
    ("P1", "fused_dequantizer", "fd_test",
     ["rtl/primitives/fused_dequantizer.v"], "tb/primitives/fused_dequantizer_tb.v"),
    ("P1", "gpu_top", "gt_test",
     ["rtl/primitives/zero_detect_mult.v", "rtl/primitives/variable_precision_alu.v",
      "rtl/primitives/sparse_memory_ctrl.v", "rtl/primitives/fused_dequantizer.v",
      "rtl/primitives/gpu_top.v"], "tb/primitives/gpu_top_tb.v"),
    # Phase 2: Extended Compute Modules
    ("P2", "mac_unit", "mac_test",
     ["rtl/compute/mac_unit.v"], "tb/compute/mac_unit_tb.v"),
    ("P2", "systolic_array", "sa_test",
     ["rtl/compute/systolic_array.v"], "tb/compute/systolic_array_tb.v"),
    ("P2", "gelu_activation", "gelu_test",
     ["rtl/compute/gelu_lut_256.v", "rtl/compute/gelu_activation.v"], "tb/compute/gelu_activation_tb.v"),
    ("P2", "softmax_unit", "sm_test",
     ["rtl/compute/exp_lut_256.v", "rtl/compute/softmax_unit.v"], "tb/compute/softmax_unit_tb.v"),
    # Phase 3: Transformer Building Blocks
    ("P3", "layer_norm", "ln_test",
     ["rtl/compute/inv_sqrt_lut_256.v", "rtl/transformer/layer_norm.v"], "tb/transformer/layer_norm_tb.v"),
    ("P3", "linear_layer", "ll_test",
     ["rtl/transformer/linear_layer.v"], "tb/transformer/linear_layer_tb.v"),
    ("P3", "attention_unit", "au_test",
     ["rtl/compute/exp_lut_256.v", "rtl/transformer/attention_unit.v"], "tb/transformer/attention_unit_tb.v"),
    ("P3", "ffn_block", "ffn_test",
     ["rtl/compute/gelu_lut_256.v", "rtl/transformer/ffn_block.v"], "tb/transformer/ffn_block_tb.v"),
    # Phase 4: GPT-2 Full Pipeline
    ("P4", "embedding_lookup", "emb_test",
     ["rtl/gpt2/embedding_lookup.v"], "tb/gpt2/embedding_lookup_tb.v"),
    ("P4", "gpt2_engine_FULL", "gpt2_test",
     ["rtl/gpt2/embedding_lookup.v", "rtl/gpt2/transformer_block.v", "rtl/gpt2/gpt2_engine.v",
      "rtl/transformer/layer_norm.v", "rtl/transformer/attention_unit.v", "rtl/transformer/ffn_block.v",
      "rtl/transformer/linear_layer.v", "rtl/compute/gelu_activation.v", "rtl/compute/gelu_lut_256.v",
      "rtl/compute/softmax_unit.v", "rtl/compute/exp_lut_256.v", "rtl/compute/inv_sqrt_lut_256.v"],
     "tb/gpt2/gpt2_engine_tb.v"),
    ("P4", "accel_gpt2_engine", "gpt2_acc_test",
     ["rtl/primitives/zero_detect_mult.v", "rtl/primitives/fused_dequantizer.v",
      "rtl/primitives/gpu_core.v", "rtl/compute/exp_lut_256.v", "rtl/compute/inv_sqrt_lut_256.v",
      "rtl/transformer/layer_norm.v", "rtl/transformer/accelerated_attention.v",
      "rtl/transformer/accelerated_linear_layer.v", "rtl/transformer/accelerated_transformer_block.v",
      "rtl/gpt2/embedding_lookup.v", "rtl/gpt2/accelerated_gpt2_engine.v"],
     "tb/gpt2/accelerated_gpt2_engine_tb.v"),
    # Phase 5: Memory Interface
    ("P5", "axi_weight_memory", "axi_test",
     ["rtl/memory/axi_weight_memory.v"], "tb/memory/axi_weight_memory_tb.v"),
    ("P5", "dma_engine", "dma_test",
     ["rtl/memory/dma_engine.v"], "tb/memory/dma_engine_tb.v"),
    ("P5", "scratchpad", "sp_test",
     ["rtl/memory/scratchpad.v"], "tb/memory/scratchpad_tb.v"),
    # Phase 6: Top-Level Control
    ("P6", "command_processor", "cmd_test",
     ["rtl/top/command_processor.v"], "tb/top/command_processor_tb.v"),
    ("P6", "perf_counters", "perf_test",
     ["rtl/top/perf_counters.v"], "tb/top/perf_counters_tb.v"),
    ("P6", "gpu_config_regs", "cfg_test",
     ["rtl/top/gpu_config_regs.v"], "tb/top/gpu_config_regs_tb.v"),
    ("P6", "reset_synchronizer", "rst_test",
     ["rtl/top/reset_synchronizer.v"], "tb/top/reset_synchronizer_tb.v"),
    # Phase 7: System Integration
    ("P7", "gpu_system_top", "sys_test",
     ["rtl/top/reset_synchronizer.v", "rtl/top/gpu_config_regs.v", "rtl/top/command_processor.v",
      "rtl/top/perf_counters.v", "rtl/memory/scratchpad.v", "rtl/memory/dma_engine.v",
      "rtl/top/gpu_system_top.v"],
     "tb/top/gpu_system_top_tb.v"),
    # Phase 8: Q4 Quantization Support
    ("P8", "block_dequantizer", "bdq_test",
     ["rtl/compute/block_dequantizer.v"], "tb/compute/block_dequantizer_tb.v"),
    ("P8", "systolic_array_q4", "sa_q4_test",
     ["rtl/compute/systolic_array.v"], "tb/compute/systolic_array_q4_tb.v"),
    ("P8", "nanogpt_q4_e2e", "nanogpt_q4_test",
     ["rtl/compute/gelu_lut_256.v", "rtl/compute/exp_lut_256.v",
      "rtl/compute/inv_sqrt_lut_256.v",
      "rtl/transformer/layer_norm.v", "rtl/transformer/attention_unit.v",
      "rtl/transformer/ffn_block.v",
      "rtl/gpt2/transformer_block.v", "rtl/gpt2/embedding_lookup.v",
      "rtl/gpt2/gpt2_engine.v"],
     "tb/gpt2/nanogpt_q4_tb.v"),
]

total_pass, total_fail, total_modules = 0, 0, 0
results = []
current_phase = ""

for phase, name, outbin, sources, tb in TESTS:
    if phase != current_phase:
        if current_phase:
            print()
        print(f"--- {phase}: ---")
        current_phase = phase

    total_modules += 1
    all_files = [os.path.join(ROOT, f) for f in sources + [tb]]
    out_path = os.path.join(ROOT, "sim", outbin)

    print(f"  [{phase}] {name} ... ", end="", flush=True)

    # Compile
    comp = subprocess.run(
        [IVERILOG, "-o", out_path] + all_files,
        capture_output=True, text=True
    )
    if comp.returncode != 0:
        print("COMPILE FAIL")
        print(f"    {comp.stderr[:200]}")
        total_fail += 1
        results.append((phase, name, "COMPILE FAIL", "-"))
        continue

    # Simulate
    sim = subprocess.run(
        [VVP, out_path], capture_output=True, text=True, timeout=60
    )
    output = sim.stdout + sim.stderr

    # Count PASS/FAIL — exclude summary lines like "16 [PASS], 0 [FAIL]"
    passes = len(re.findall(r'\[PASS\]', output))
    fails = len(re.findall(r'\[FAIL\]', output))
    # Subtract summary line matches: "N [PASS], N [FAIL]"
    summary = re.findall(r'(\d+)\s+\[PASS\],\s+(\d+)\s+\[FAIL\]', output)
    for sp, sf in summary:
        passes -= 1  # remove the summary [PASS] match
        fails -= 1   # remove the summary [FAIL] match
        # Use summary counts if no individual markers found
        if passes <= 1 and fails <= 0:
            passes = int(sp)
            fails = int(sf)

    # Fallback: "N PASSED, N FAILED" summary
    if passes == 0 and fails == 0:
        m = re.search(r'(\d+)\s+(?:PASSED|passed),\s+(\d+)\s+(?:FAILED|failed)', output)
        if m:
            passes, fails = int(m.group(1)), int(m.group(2))

    total_pass += passes
    total_fail += fails

    if fails == 0 and passes > 0:
        print(f"{passes}/{passes} PASS")
        results.append((phase, name, "PASS", f"{passes}/{passes}"))
    elif passes > 0:
        print(f"{passes} PASS, {fails} FAIL")
        results.append((phase, name, "PARTIAL", f"{passes}/{passes+fails}"))
    else:
        print("NO OUTPUT")
        results.append((phase, name, "NO OUTPUT", "0/0"))

    # Save output log
    with open(os.path.join(ROOT, "sim", f"{outbin}_output.log"), "w") as f:
        f.write(output)

# Summary
print()
print("=" * 60)
print("                  TEST RESULTS SUMMARY")
print("=" * 60)
print(f"  Modules tested : {total_modules}")
print(f"  Total PASS     : {total_pass}")
print(f"  Total FAIL     : {total_fail}")
print()
print(f"  {'Phase':<6} {'Module':<25} {'Status':<15} {'Tests':<10}")
print(f"  {'-'*6} {'-'*25} {'-'*15} {'-'*10}")
for phase, name, status, tests in results:
    print(f"  {phase:<6} {name:<25} {status:<15} {tests:<10}")
print()
if total_fail == 0 and total_pass > 0:
    print("  >>> ALL TESTS PASSED - GPU READY FOR DEPLOYMENT <<<")
elif total_pass == 0:
    print("  >>> NO TESTS RAN - CHECK IVERILOG INSTALLATION <<<")
else:
    print("  >>> SOME TESTS FAILED - REVIEW REQUIRED <<<")
