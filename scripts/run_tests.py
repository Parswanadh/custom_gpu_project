#!/usr/bin/env python3
"""
run_tests.py - Run all Verilog testbenches for the Custom GPU
Usage: python scripts/run_tests.py

Optional env overrides (useful for CI smoke checks):
  BITBYBIT_IVERILOG         Override iverilog executable path
  BITBYBIT_VVP              Override vvp executable path
  BITBYBIT_SIM_TIMEOUT_SEC  Per-test simulation timeout (default: 60)
  BITBYBIT_MAX_TESTS        Limit number of tests executed (default: 0 = all)
  BITBYBIT_WARN_INVALID_ENV Emit warnings for invalid numeric knobs (1/true/on/yes)
"""
import os
import re
import subprocess
import sys
from shutil import which

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IVERILOG = r"D:\Tools\iverilog\bin\iverilog.exe"
DEFAULT_VVP = r"D:\Tools\iverilog\bin\vvp.exe"
WARN_INVALID_ENV = os.environ.get("BITBYBIT_WARN_INVALID_ENV", "").strip().lower() in {"1", "true", "on", "yes"}


def _warn(message):
    if WARN_INVALID_ENV:
        print(f"[WARN] {message}", file=sys.stderr)


def _coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _resolve_executable(env_var, default_path, fallback_name):
    explicit = os.environ.get(env_var)
    if explicit is not None:
        candidate = explicit.strip()
        if not candidate:
            print(f"[ERROR] {env_var} is explicitly set but empty. Refusing PATH fallback.", file=sys.stderr)
            sys.exit(1)
        if os.path.isfile(candidate):
            return candidate
        resolved_explicit = which(candidate)
        if resolved_explicit:
            return resolved_explicit
        print(
            f"[ERROR] {env_var} is explicitly set to '{candidate}', but that target is invalid. Refusing PATH fallback.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.path.isfile(default_path):
        return default_path

    resolved_fallback = which(fallback_name)
    if resolved_fallback:
        return resolved_fallback

    print(
        f"[ERROR] {fallback_name} not found at '{default_path}' and not available in PATH.",
        file=sys.stderr,
    )
    sys.exit(1)


def _int_env(name, default, minimum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn(f"Ignoring invalid {name}={raw!r}; using default {default}.")
        return default
    if value < minimum:
        _warn(f"Ignoring {name}={raw!r}; expected >= {minimum}. Using default {default}.")
        return default
    return value


IVERILOG = _resolve_executable("BITBYBIT_IVERILOG", DEFAULT_IVERILOG, "iverilog")
VVP = _resolve_executable("BITBYBIT_VVP", DEFAULT_VVP, "vvp")
SIM_TIMEOUT_SEC = _int_env("BITBYBIT_SIM_TIMEOUT_SEC", 60, 1)
MAX_TESTS = _int_env("BITBYBIT_MAX_TESTS", 0, 0)

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
    # Phase 9: Unified Top-Level Integration
    ("P9", "gpu_system_top_v2", "sys_v2_test",
     ["rtl/top/reset_synchronizer.v", "rtl/top/gpu_config_regs.v", "rtl/top/command_processor.v",
       "rtl/top/perf_counters.v", "rtl/memory/scratchpad.v", "rtl/memory/dma_engine.v",
        "rtl/memory/imprinted_embedding_rom.v",
        "rtl/integration/imprinted_mini_transformer_core.v",
        "rtl/transformer/rope_encoder.v", "rtl/transformer/grouped_query_attention.v",
        "rtl/compute/parallel_softmax.v", "rtl/compute/exp_lut_256.v", "rtl/compute/recip_lut_256.v",
        "rtl/compute/gelu_lut_256.v", "rtl/compute/gelu_activation.v",
        "rtl/memory/kv_cache_quantizer.v", "rtl/compute/activation_compressor.v",
        "rtl/memory/prefetch_engine.v", "rtl/control/layer_pipeline_controller.v",
        "rtl/integration/optimized_transformer_layer.v", "rtl/top/gpu_system_top_v2.v"],
     "tb/top/gpu_system_top_v2_tb.v"),
    # Phase 10: Extended Throughput/Integration Coverage
    ("P10", "parallel_softmax", "parsm_test",
     ["rtl/compute/parallel_softmax.v", "rtl/compute/exp_lut_256.v", "rtl/compute/recip_lut_256.v"],
     "tb/compute/parallel_softmax_tb.v"),
    ("P10", "layer_pipeline_controller", "pipe_test",
     ["rtl/control/layer_pipeline_controller.v"], "tb/control/layer_pipeline_controller_tb.v"),
    ("P10", "prefetch_engine", "prefetch_test",
     ["rtl/memory/prefetch_engine.v"], "tb/memory/prefetch_engine_tb.v"),
    ("P10", "optimized_transformer_layer", "e2e_test",
     ["rtl/integration/optimized_transformer_layer.v",
      "rtl/transformer/rope_encoder.v", "rtl/transformer/grouped_query_attention.v",
      "rtl/compute/parallel_softmax.v", "rtl/compute/exp_lut_256.v", "rtl/compute/recip_lut_256.v",
      "rtl/compute/gelu_activation.v", "rtl/compute/gelu_lut_256.v",
      "rtl/memory/kv_cache_quantizer.v", "rtl/compute/activation_compressor.v"],
     "tb/integration/end_to_end_pipeline_tb.v"),
    ("P10", "full_model_inference", "full_model_test",
     ["rtl/gpt2/embedding_lookup.v",
      "rtl/integration/optimized_transformer_layer.v",
      "rtl/transformer/rope_encoder.v", "rtl/transformer/grouped_query_attention.v",
      "rtl/compute/parallel_softmax.v", "rtl/compute/exp_lut_256.v", "rtl/compute/recip_lut_256.v",
      "rtl/compute/gelu_activation.v", "rtl/compute/gelu_lut_256.v",
      "rtl/memory/kv_cache_quantizer.v", "rtl/compute/activation_compressor.v",
      "rtl/compute/medusa_head_predictor.v"],
     "tb/integration/full_model_inference_tb.v"),
    ("P10", "full_model_imprint", "full_model_imprint_test",
     ["rtl/memory/imprinted_embedding_rom.v",
      "rtl/integration/imprinted_mini_transformer_core.v",
      "rtl/compute/medusa_head_predictor.v"],
     "tb/integration/full_model_inference_imprint_tb.v"),
]

if MAX_TESTS > 0:
    TESTS = TESTS[:MAX_TESTS]

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
    log_path = os.path.join(ROOT, "sim", f"{outbin}_output.log")

    print(f"  [{phase}] {name} ... ", end="", flush=True)

    # Compile
    try:
        comp = subprocess.run(
            [IVERILOG, "-g2012", "-o", out_path] + all_files,
            capture_output=True, text=True
        )
    except OSError as e:
        error_text = f"Failed to launch iverilog '{IVERILOG}': {e}"
        print("COMPILE LAUNCH FAIL")
        print(f"    {error_text[:200]}")
        total_fail += 1
        results.append((phase, name, "COMPILE LAUNCH FAIL", "-"))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(error_text + "\n")
        continue

    if comp.returncode != 0:
        print("COMPILE FAIL")
        print(f"    {comp.stderr[:200]}")
        total_fail += 1
        results.append((phase, name, "COMPILE FAIL", "-"))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write((comp.stdout or "") + (comp.stderr or ""))
        continue

    # Simulate
    try:
        sim = subprocess.run(
            [VVP, out_path], capture_output=True, text=True, timeout=SIM_TIMEOUT_SEC
        )
    except subprocess.TimeoutExpired as e:
        output = _coerce_text(e.stdout) + _coerce_text(e.stderr)
        print("TIMEOUT")
        if output.strip():
            print(f"    {output[:200]}")
        total_fail += 1
        results.append((phase, name, "TIMEOUT", "-"))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output)
        continue
    except OSError as e:
        error_text = f"Failed to launch vvp '{VVP}': {e}"
        print("SIM LAUNCH FAIL")
        print(f"    {error_text[:200]}")
        total_fail += 1
        results.append((phase, name, "SIM LAUNCH FAIL", "-"))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(error_text + "\n")
        continue

    output = sim.stdout + sim.stderr
    if sim.returncode != 0:
        print("SIM FAIL")
        total_fail += 1
        results.append((phase, name, "SIM FAIL", "-"))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output)
        continue

    # Preferred parser: explicit machine-readable TB summary.
    tb_result = re.search(r'TB_RESULT\s+pass=(\d+)\s+fail=(\d+)', output, flags=re.IGNORECASE)
    if tb_result:
        passes = int(tb_result.group(1))
        fails = int(tb_result.group(2))
    else:
        # Backward-compatible fallback for legacy benches.
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
    elif passes > 0 and fails > 0:
        print(f"{passes} PASS, {fails} FAIL")
        results.append((phase, name, "PARTIAL", f"{passes}/{passes+fails}"))
    elif fails > 0:
        print(f"0 PASS, {fails} FAIL")
        results.append((phase, name, "FAIL", f"0/{fails}"))
    else:
        print("NO OUTPUT")
        total_fail += 1
        results.append((phase, name, "NO OUTPUT", "0/0"))

    # Save output log
    with open(log_path, "w", encoding="utf-8") as f:
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
if total_fail > 0:
    print("  >>> SOME TESTS FAILED - REVIEW REQUIRED <<<")
    exit_code = 1
elif total_pass > 0:
    print("  >>> ALL TESTS PASSED - GPU READY FOR DEPLOYMENT <<<")
    exit_code = 0
else:
    print("  >>> NO TESTS RAN - CHECK IVERILOG INSTALLATION <<<")
    exit_code = 2

sys.exit(exit_code)
