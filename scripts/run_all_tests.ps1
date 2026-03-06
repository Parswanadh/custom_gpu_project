# ============================================================================
# run_all_tests.ps1 - Run all Verilog testbenches for the Custom GPU
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$iverilog = "D:\Tools\iverilog\bin\iverilog.exe"
$vvp = "D:\Tools\iverilog\bin\vvp.exe"
$root = "D:\Projects\BitbyBit\custom_gpu_project"

# Create sim output directory if missing
$simDir = Join-Path $root "sim"
if (-not (Test-Path $simDir)) { New-Item -ItemType Directory -Path $simDir -Force | Out-Null }
$waveDir = Join-Path $simDir "waveforms"
if (-not (Test-Path $waveDir)) { New-Item -ItemType Directory -Path $waveDir -Force | Out-Null }

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "       CUSTOM GPU - FULL TEST SUITE                             " -ForegroundColor Cyan
Write-Host "       Designed for GPT-2 Inference on FPGA                     " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$totalPass = 0
$totalFail = 0
$totalModules = 0
$results = @()

function Run-Test {
    param(
        [string]$Phase,
        [string]$Name,
        [string]$OutputBin,
        [string[]]$Sources,
        [string]$Testbench
    )

    $allFiles = ($Sources + $Testbench) | ForEach-Object { Join-Path $root $_ }
    $outPath = Join-Path (Join-Path $root "sim") $OutputBin

    Write-Host "  [$Phase] $Name ... " -NoNewline

    # Compile
    $compileArgs = @("-o", $outPath) + $allFiles
    $compileResult = & $iverilog @compileArgs 2>&1

    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAIL" -ForegroundColor Red
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="COMPILE FAIL"; Tests="-"}
        return
    }

    # Simulate
    $simOutput = & $vvp $outPath 2>&1 | Out-String -Width 300

    # Count PASS/FAIL from output (match [PASS]/[FAIL] and also "N passed, N failed" summary)
    $passes = ([regex]::Matches($simOutput, '\[PASS\]')).Count
    $fails = ([regex]::Matches($simOutput, '\[FAIL\]')).Count

    # Fallback: if no [PASS]/[FAIL] markers, try to parse "Results: N passed, N failed"
    if ($passes -eq 0 -and $fails -eq 0) {
        $summaryMatch = [regex]::Match($simOutput, '(\d+)\s+(?:PASSED|passed),\s+(\d+)\s+(?:FAILED|failed)')
        if ($summaryMatch.Success) {
            $passes = [int]$summaryMatch.Groups[1].Value
            $fails = [int]$summaryMatch.Groups[2].Value
        }
    }

    $script:totalPass += $passes
    $script:totalFail += $fails
    $script:totalModules++

    if ($fails -eq 0 -and $passes -gt 0) {
        Write-Host "$passes/$passes PASS" -ForegroundColor Green
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="PASS"; Tests="$passes/$passes"}
    } elseif ($passes -gt 0) {
        $total = $passes + $fails
        Write-Host "$passes PASS, $fails FAIL" -ForegroundColor Yellow
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="PARTIAL"; Tests="$passes/$total"}
    } else {
        Write-Host "NO OUTPUT" -ForegroundColor Red
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="NO OUTPUT"; Tests="0/0"}
    }
}

# -- Phase 1: Core Compute Primitives --
Write-Host "--- Phase 1: Core Compute Primitives ---" -ForegroundColor Yellow
Run-Test "P1" "zero_detect_mult" "zdm_test" @("rtl/primitives/zero_detect_mult.v") "tb/primitives/zero_detect_mult_tb.v"
Run-Test "P1" "variable_precision_alu" "vpa_test" @("rtl/primitives/variable_precision_alu.v") "tb/primitives/variable_precision_alu_tb.v"
Run-Test "P1" "sparse_memory_ctrl" "smc_test" @("rtl/primitives/sparse_memory_ctrl.v") "tb/primitives/sparse_memory_ctrl_tb.v"
Run-Test "P1" "fused_dequantizer" "fd_test" @("rtl/primitives/fused_dequantizer.v") "tb/primitives/fused_dequantizer_tb.v"
Run-Test "P1" "gpu_top" "gt_test" @(
    "rtl/primitives/zero_detect_mult.v",
    "rtl/primitives/variable_precision_alu.v",
    "rtl/primitives/sparse_memory_ctrl.v",
    "rtl/primitives/fused_dequantizer.v",
    "rtl/primitives/gpu_top.v"
) "tb/primitives/gpu_top_tb.v"
Write-Host ""

# -- Phase 2: Extended Compute Modules --
Write-Host "--- Phase 2: Extended Compute Modules ---" -ForegroundColor Yellow
Run-Test "P2" "mac_unit" "mac_test" @("rtl/compute/mac_unit.v") "tb/compute/mac_unit_tb.v"
Run-Test "P2" "systolic_array" "sa_test" @("rtl/compute/systolic_array.v") "tb/compute/systolic_array_tb.v"
Run-Test "P2" "gelu_activation" "gelu_test" @("rtl/compute/gelu_lut_256.v", "rtl/compute/gelu_activation.v") "tb/compute/gelu_activation_tb.v"
Run-Test "P2" "softmax_unit" "sm_test" @("rtl/compute/exp_lut_256.v", "rtl/compute/softmax_unit.v") "tb/compute/softmax_unit_tb.v"
Write-Host ""

# -- Phase 3: Transformer Building Blocks --
Write-Host "--- Phase 3: Transformer Building Blocks ---" -ForegroundColor Yellow
Run-Test "P3" "layer_norm" "ln_test" @("rtl/compute/inv_sqrt_lut_256.v", "rtl/transformer/layer_norm.v") "tb/transformer/layer_norm_tb.v"
Run-Test "P3" "linear_layer" "ll_test" @("rtl/transformer/linear_layer.v") "tb/transformer/linear_layer_tb.v"
Run-Test "P3" "attention_unit" "au_test" @("rtl/compute/exp_lut_256.v", "rtl/transformer/attention_unit.v") "tb/transformer/attention_unit_tb.v"
Run-Test "P3" "ffn_block" "ffn_test" @("rtl/compute/gelu_lut_256.v", "rtl/transformer/ffn_block.v") "tb/transformer/ffn_block_tb.v"
Write-Host ""

# -- Phase 4: GPT-2 Full Pipeline --
Write-Host "--- Phase 4: GPT-2 Full Pipeline ---" -ForegroundColor Yellow
Run-Test "P4" "embedding_lookup" "emb_test" @("rtl/gpt2/embedding_lookup.v") "tb/gpt2/embedding_lookup_tb.v"
Run-Test "P4" "gpt2_engine_FULL" "gpt2_test" @(
    "rtl/gpt2/embedding_lookup.v",
    "rtl/gpt2/transformer_block.v",
    "rtl/gpt2/gpt2_engine.v",
    "rtl/transformer/layer_norm.v",
    "rtl/transformer/attention_unit.v",
    "rtl/transformer/ffn_block.v",
    "rtl/transformer/linear_layer.v",
    "rtl/compute/gelu_activation.v",
    "rtl/compute/gelu_lut_256.v",
    "rtl/compute/softmax_unit.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/inv_sqrt_lut_256.v"
) "tb/gpt2/gpt2_engine_tb.v"
Run-Test "P4" "accel_gpt2_engine" "gpt2_acc_test" @(
    "rtl/primitives/zero_detect_mult.v",
    "rtl/primitives/fused_dequantizer.v",
    "rtl/primitives/gpu_core.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/inv_sqrt_lut_256.v",
    "rtl/transformer/layer_norm.v",
    "rtl/transformer/accelerated_attention.v",
    "rtl/transformer/accelerated_linear_layer.v",
    "rtl/transformer/accelerated_transformer_block.v",
    "rtl/gpt2/embedding_lookup.v",
    "rtl/gpt2/accelerated_gpt2_engine.v"
) "tb/gpt2/accelerated_gpt2_engine_tb.v"
Write-Host ""

# -- Phase 5: Memory Interface --
Write-Host "--- Phase 5: Memory Interface ---" -ForegroundColor Yellow
Run-Test "P5" "axi_weight_memory" "axi_test" @("rtl/memory/axi_weight_memory.v") "tb/memory/axi_weight_memory_tb.v"
Run-Test "P5" "dma_engine" "dma_test" @("rtl/memory/dma_engine.v") "tb/memory/dma_engine_tb.v"
Run-Test "P5" "scratchpad" "sp_test" @("rtl/memory/scratchpad.v") "tb/memory/scratchpad_tb.v"
Write-Host ""

# -- Phase 6: Top-Level Control --
Write-Host "--- Phase 6: Top-Level Control ---" -ForegroundColor Yellow
Run-Test "P6" "command_processor" "cmd_test" @("rtl/top/command_processor.v") "tb/top/command_processor_tb.v"
Run-Test "P6" "perf_counters" "perf_test" @("rtl/top/perf_counters.v") "tb/top/perf_counters_tb.v"
Run-Test "P6" "gpu_config_regs" "cfg_test" @("rtl/top/gpu_config_regs.v") "tb/top/gpu_config_regs_tb.v"
Run-Test "P6" "reset_synchronizer" "rst_test" @("rtl/top/reset_synchronizer.v") "tb/top/reset_synchronizer_tb.v"
Write-Host ""

# -- Phase 7: System Integration --
Write-Host "--- Phase 7: System Integration ---" -ForegroundColor Yellow
Run-Test "P7" "gpu_system_top" "sys_test" @(
    "rtl/top/reset_synchronizer.v",
    "rtl/top/gpu_config_regs.v",
    "rtl/top/command_processor.v",
    "rtl/top/perf_counters.v",
    "rtl/memory/scratchpad.v",
    "rtl/memory/dma_engine.v",
    "rtl/top/gpu_system_top.v"
) "tb/top/gpu_system_top_tb.v"
Write-Host ""

# -- Phase 8: Architecture Improvements --
Write-Host "--- Phase 8: Architecture Improvements ---" -ForegroundColor Yellow
Run-Test "P8" "online_softmax" "osm_test" @("rtl/compute/exp_lut_256.v", "rtl/compute/recip_lut_256.v", "rtl/compute/online_softmax.v") "tb/compute/online_softmax_tb.v"
Run-Test "P8" "sparsity_decoder_2_4" "sp24_test" @("rtl/compute/sparsity_decoder_2_4.v") "tb/compute/sparsity_decoder_2_4_tb.v"
Run-Test "P8" "activation_compressor" "ac_test" @("rtl/compute/activation_compressor.v") "tb/compute/activation_compressor_tb.v"
Run-Test "P8" "weight_double_buffer" "wdb_test" @("rtl/memory/weight_double_buffer.v") "tb/memory/weight_double_buffer_tb.v"
Run-Test "P8" "parallel_attention" "pa_test" @("rtl/transformer/parallel_attention.v") "tb/transformer/parallel_attention_tb.v"
Run-Test "P8" "token_scheduler" "ts_test" @("rtl/top/token_scheduler.v") "tb/top/token_scheduler_tb.v"
Run-Test "P8" "power_management_unit" "pmu_test" @("rtl/top/power_management_unit.v") "tb/top/power_management_unit_tb.v"
Write-Host ""
# -- Phase 9: Advanced SOTA Integrations --
Write-Host "--- Phase 9: Advanced SOTA Integrations ---" -ForegroundColor Yellow
Run-Test "P9" "w4a8_decompressor" "w4a8_test" @("rtl/compute/w4a8_decompressor.v") "tb/compute/w4a8_decompressor_tb.v"
Run-Test "P9" "moe_router" "moe_test" @("rtl/compute/moe_router.v") "tb/compute/moe_router_tb.v"
Write-Host ""
# -- Phase 10: New Features --
Write-Host "--- Phase 10: New Features ---" -ForegroundColor Yellow
Run-Test "P10" "speculative_decode_engine" "spec_test" @("rtl/compute/speculative_decode_engine.v") "tb/compute/speculative_decode_engine_tb.v"
Run-Test "P10" "paged_attention_mmu" "pa_mmu_test" @("rtl/memory/paged_attention_mmu.v") "tb/memory/paged_attention_mmu_tb.v"
Write-Host ""
# -- Phase 11: Closing SOTA Gaps --
Write-Host "--- Phase 11: Closing SOTA Gaps ---" -ForegroundColor Yellow
Run-Test "P11" "flash_attention_unit" "flash_test" @("rtl/transformer/flash_attention_unit.v") "tb/transformer/flash_attention_unit_tb.v"
Run-Test "P11" "mixed_precision_decompressor" "mpd_test" @("rtl/compute/mixed_precision_decompressor.v") "tb/compute/mixed_precision_decompressor_tb.v"
Run-Test "P11" "q4_weight_pipeline" "q4_test" @("rtl/compute/w4a8_decompressor.v", "rtl/compute/mixed_precision_decompressor.v", "rtl/compute/q4_weight_pipeline.v") "tb/compute/q4_weight_pipeline_tb.v"
Write-Host ""

# -- Summary --
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "                    TEST RESULTS SUMMARY                        " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Modules tested : $totalModules"
Write-Host "  Total PASS     : $totalPass" -ForegroundColor Green
if ($totalFail -gt 0) {
    Write-Host "  Total FAIL     : $totalFail" -ForegroundColor Red
} else {
    Write-Host "  Total FAIL     : $totalFail" -ForegroundColor Green
}
Write-Host ""

$results | Format-Table -AutoSize

if ($totalFail -eq 0 -and $totalPass -gt 0) {
    Write-Host "  >>> ALL TESTS PASSED - GPU READY FOR DEPLOYMENT <<<" -ForegroundColor Green
} elseif ($totalPass -eq 0) {
    Write-Host "  >>> NO TESTS RAN - CHECK IVERILOG INSTALLATION <<<" -ForegroundColor Red
} else {
    Write-Host "  >>> SOME TESTS FAILED - REVIEW REQUIRED <<<" -ForegroundColor Red
}
Write-Host ""
