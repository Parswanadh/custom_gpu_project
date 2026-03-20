# ============================================================================
# run_all_tests.ps1 - Run all Verilog testbenches for the Custom GPU
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1
# ============================================================================

$ErrorActionPreference = "Continue"

function Resolve-ToolPath {
    param(
        [string]$EnvVar,
        [string]$DefaultPath,
        [string]$FallbackCommand
    )

    $fromEnv = [Environment]::GetEnvironmentVariable($EnvVar)
    if ($fromEnv) {
        if (Test-Path $fromEnv) { return $fromEnv }
        throw "$EnvVar is set but file does not exist: $fromEnv"
    }

    if (Test-Path $DefaultPath) { return $DefaultPath }

    $cmd = Get-Command $FallbackCommand -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    throw "Unable to locate $FallbackCommand. Set $EnvVar or install tool in PATH."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$iverilog = Resolve-ToolPath -EnvVar "BITBYBIT_IVERILOG" -DefaultPath "D:\Tools\iverilog\bin\iverilog.exe" -FallbackCommand "iverilog"
$vvp = Resolve-ToolPath -EnvVar "BITBYBIT_VVP" -DefaultPath "D:\Tools\iverilog\bin\vvp.exe" -FallbackCommand "vvp"

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
$simulationTimeoutSec = 60

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
    $logPath = Join-Path (Join-Path $root "sim") "${OutputBin}_output.log"

    Write-Host "  [$Phase] $Name ... " -NoNewline

    # Compile
    $compileArgs = @("-g2012", "-o", $outPath) + $allFiles
    $compileResult = ""
    $compileLaunchFailed = $false
    try {
        $compileResult = & $iverilog @compileArgs 2>&1
    } catch {
        $compileLaunchFailed = $true
        $compileResult = ($_ | Out-String)
    }

    if ($compileLaunchFailed) {
        Write-Host "COMPILE LAUNCH FAIL" -ForegroundColor Red
        Set-Content -Path $logPath -Value ($compileResult | Out-String)
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="COMPILE LAUNCH FAIL"; Tests="-"}
        return
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAIL" -ForegroundColor Red
        Set-Content -Path $logPath -Value ($compileResult | Out-String)
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="COMPILE FAIL"; Tests="-"}
        return
    }

    # Simulate with per-test timeout to prevent hangs
    $simOutput = ""
    $simExitCode = 1
    $simTimedOut = $false
    $simLaunchFailed = $false
    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    $simProcess = $null

    try {
        $simProcess = Start-Process -FilePath $vvp -ArgumentList @($outPath) -NoNewWindow -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

        $simTimedOut = -not $simProcess.WaitForExit($simulationTimeoutSec * 1000)
        if ($simTimedOut) {
            try {
                Stop-Process -Id $simProcess.Id -Force -ErrorAction Stop
            } catch {
                # Best effort process cleanup on timeout
            }
            try {
                $null = $simProcess.WaitForExit(5000)
            } catch {
                # Continue with timeout failure path even if wait throws
            }
        }

        if (-not $simTimedOut) {
            $simExitCode = [int]$simProcess.ExitCode
        }

        if (Test-Path $stdoutPath) {
            $simOutput += (Get-Content -Path $stdoutPath -Raw)
        }
        if (Test-Path $stderrPath) {
            $simOutput += (Get-Content -Path $stderrPath -Raw)
        }
    } catch {
        $simLaunchFailed = $true
        $simOutput += ($_ | Out-String)
    } finally {
        if ($simProcess -ne $null) {
            $simProcess.Dispose()
        }
        if (Test-Path $stdoutPath) {
            Remove-Item -Path $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrPath) {
            Remove-Item -Path $stderrPath -Force -ErrorAction SilentlyContinue
        }
    }

    if ($simTimedOut) {
        Write-Host "TIMEOUT" -ForegroundColor Red
        Set-Content -Path $logPath -Value $simOutput
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="TIMEOUT"; Tests="-"}
        return
    }

    if ($simLaunchFailed) {
        Write-Host "SIM LAUNCH FAIL" -ForegroundColor Red
        Set-Content -Path $logPath -Value $simOutput
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="SIM LAUNCH FAIL"; Tests="-"}
        return
    }

    if ($simExitCode -ne 0) {
        Write-Host "SIM FAIL" -ForegroundColor Red
        Set-Content -Path $logPath -Value $simOutput
        $script:totalFail++
        $script:totalModules++
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="SIM FAIL"; Tests="-"}
        return
    }

    # Preferred parser: explicit machine-readable TB summary.
    $tbResult = [regex]::Match($simOutput, 'TB_RESULT\s+pass=(\d+)\s+fail=(\d+)', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if ($tbResult.Success) {
        $passes = [int]$tbResult.Groups[1].Value
        $fails = [int]$tbResult.Groups[2].Value
    } else {
        # Backward-compatible fallback for legacy benches.
        $passes = ([regex]::Matches($simOutput, '\[PASS\]')).Count
        $fails = ([regex]::Matches($simOutput, '\[FAIL\]')).Count

        # Remove summary marker matches like "4 [PASS], 0 [FAIL]"
        $summaryMatches = [regex]::Matches($simOutput, '(\d+)\s+\[PASS\],\s+(\d+)\s+\[FAIL\]')
        foreach ($m in $summaryMatches) {
            if ($passes -gt 0) { $passes-- }
            if ($fails -gt 0) { $fails-- }
            if ($passes -le 1 -and $fails -le 0) {
                $passes = [int]$m.Groups[1].Value
                $fails = [int]$m.Groups[2].Value
            }
        }

        # Fallback: if no [PASS]/[FAIL] markers, try to parse "Results: N passed, N failed"
        if ($passes -eq 0 -and $fails -eq 0) {
            $summaryMatch = [regex]::Match($simOutput, '(\d+)\s+(?:PASSED|passed),\s+(\d+)\s+(?:FAILED|failed)')
            if ($summaryMatch.Success) {
                $passes = [int]$summaryMatch.Groups[1].Value
                $fails = [int]$summaryMatch.Groups[2].Value
            }
        }
    }

    $script:totalPass += $passes
    $script:totalFail += $fails
    $script:totalModules++

    if ($fails -eq 0 -and $passes -gt 0) {
        Write-Host "$passes/$passes PASS" -ForegroundColor Green
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="PASS"; Tests="$passes/$passes"}
    } elseif ($passes -gt 0 -and $fails -gt 0) {
        $total = $passes + $fails
        Write-Host "$passes PASS, $fails FAIL" -ForegroundColor Yellow
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="PARTIAL"; Tests="$passes/$total"}
    } elseif ($fails -gt 0) {
        Write-Host "0 PASS, $fails FAIL" -ForegroundColor Red
        $script:results += [PSCustomObject]@{Phase=$Phase; Module=$Name; Status="FAIL"; Tests="0/$fails"}
    } else {
        Write-Host "NO OUTPUT" -ForegroundColor Red
        $script:totalFail++
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
# -- Phase 13: Novel Breakthroughs (2024-2026 Papers) --
Write-Host "--- Phase 13: Novel Breakthrough Features ---" -ForegroundColor Yellow
Run-Test "P13" "ternary_mac_engine" "ternary_test" @("rtl/compute/ternary_mac_engine.v") "tb/compute/ternary_mac_engine_tb.v"
Run-Test "P13" "rope_encoder" "rope_test" @("rtl/transformer/rope_encoder.v") "tb/transformer/rope_encoder_tb.v"
Run-Test "P13" "grouped_query_attention" "gqa_test" @("rtl/transformer/grouped_query_attention.v") "tb/transformer/grouped_query_attention_tb.v"
Run-Test "P13" "kv_cache_quantizer" "kv_quant_test" @("rtl/memory/kv_cache_quantizer.v") "tb/memory/kv_cache_quantizer_tb.v"
Run-Test "P13" "medusa_head_predictor" "medusa_test" @("rtl/compute/medusa_head_predictor.v") "tb/compute/medusa_head_predictor_tb.v"
Run-Test "P13" "prefetch_engine" "prefetch_test" @("rtl/memory/prefetch_engine.v") "tb/memory/prefetch_engine_tb.v"
Write-Host ""
# -- Phase 14: Memory Bandwidth Solutions (3D Stacking + Near-Memory) --
Write-Host "--- Phase 14: Memory Bandwidth Solutions ---" -ForegroundColor Yellow
Run-Test "P14" "multibank_sram_controller" "sram_bank_test" @("rtl/memory/multibank_sram_controller.v") "tb/memory/multibank_sram_controller_tb.v"
Run-Test "P14" "compute_in_sram" "cis_test" @("rtl/memory/compute_in_sram.v") "tb/memory/compute_in_sram_tb.v"
Run-Test "P14" "hbm_controller" "hbm_test" @("rtl/memory/hbm_controller.v") "tb/memory/hbm_controller_tb.v"
Write-Host ""
# -- Phase 15: Speed Optimizations --
Write-Host "--- Phase 15: Speed Optimizations ---" -ForegroundColor Yellow
Run-Test "P15" "simd_ternary_engine" "simd_test" @("rtl/compute/simd_ternary_engine.v") "tb/compute/simd_ternary_engine_tb.v"
Run-Test "P15" "parallel_softmax" "parsm_test" @(
    "rtl/compute/parallel_softmax.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/recip_lut_256.v"
) "tb/compute/parallel_softmax_tb.v"
Run-Test "P15" "layer_pipeline_controller" "pipe_test" @("rtl/control/layer_pipeline_controller.v") "tb/control/layer_pipeline_controller_tb.v"
Write-Host ""
# -- Phase 16: End-to-End Integration --
Write-Host "--- Phase 16: End-to-End Pipeline Integration ---" -ForegroundColor Yellow
Run-Test "P16" "optimized_transformer_layer" "e2e_test" @(
    "rtl/integration/optimized_transformer_layer.v",
    "rtl/transformer/rope_encoder.v",
    "rtl/transformer/grouped_query_attention.v",
    "rtl/compute/parallel_softmax.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/recip_lut_256.v",
    "rtl/compute/gelu_activation.v",
    "rtl/compute/gelu_lut_256.v",
    "rtl/memory/kv_cache_quantizer.v",
    "rtl/compute/activation_compressor.v"
) "tb/integration/end_to_end_pipeline_tb.v"
Write-Host ""

# -- Phase 17: Continuation Integrations (Q4 + Unified Top) --
Write-Host "--- Phase 17: Continuation Integrations ---" -ForegroundColor Yellow
Run-Test "P17" "block_dequantizer" "bdq_test" @("rtl/compute/block_dequantizer.v") "tb/compute/block_dequantizer_tb.v"
Run-Test "P17" "systolic_array_q4" "sa_q4_test" @("rtl/compute/systolic_array.v") "tb/compute/systolic_array_q4_tb.v"
Run-Test "P17" "nanogpt_q4_e2e" "nanogpt_q4_test" @(
    "rtl/compute/gelu_lut_256.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/inv_sqrt_lut_256.v",
    "rtl/transformer/layer_norm.v",
    "rtl/transformer/attention_unit.v",
    "rtl/transformer/ffn_block.v",
    "rtl/gpt2/transformer_block.v",
    "rtl/gpt2/embedding_lookup.v",
    "rtl/gpt2/gpt2_engine.v"
) "tb/gpt2/nanogpt_q4_tb.v"
Run-Test "P17" "gpu_system_top_v2" "sys_v2_test" @(
    "rtl/top/reset_synchronizer.v",
    "rtl/top/gpu_config_regs.v",
    "rtl/top/command_processor.v",
    "rtl/top/perf_counters.v",
    "rtl/memory/scratchpad.v",
    "rtl/memory/dma_engine.v",
    "rtl/memory/imprinted_embedding_rom.v",
    "rtl/integration/imprinted_mini_transformer_core.v",
    "rtl/transformer/rope_encoder.v",
    "rtl/transformer/grouped_query_attention.v",
    "rtl/compute/parallel_softmax.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/recip_lut_256.v",
    "rtl/compute/gelu_lut_256.v",
    "rtl/compute/gelu_activation.v",
    "rtl/memory/kv_cache_quantizer.v",
    "rtl/compute/activation_compressor.v",
    "rtl/memory/prefetch_engine.v",
    "rtl/control/layer_pipeline_controller.v",
    "rtl/integration/optimized_transformer_layer.v",
    "rtl/top/gpu_system_top_v2.v"
) "tb/top/gpu_system_top_v2_tb.v"
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
    $exitCode = 0
} elseif ($totalFail -gt 0) {
    Write-Host "  >>> SOME TESTS FAILED - REVIEW REQUIRED <<<" -ForegroundColor Red
    $exitCode = 1
} else {
    Write-Host "  >>> NO TESTS RAN - CHECK IVERILOG INSTALLATION <<<" -ForegroundColor Red
    $exitCode = 2
}
Write-Host ""
exit $exitCode
