# ============================================================================
# run_all_tests.ps1 — Run all Verilog testbenches for the Custom GPU
# Usage: .\scripts\run_all_tests.ps1
# ============================================================================

$ErrorActionPreference = "Continue"
$iverilog = "D:\Tools\iverilog\bin\iverilog.exe"
$vvp = "D:\Tools\iverilog\bin\vvp.exe"
$root = $PSScriptRoot | Split-Path -Parent

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       CUSTOM GPU — FULL TEST SUITE                         ║" -ForegroundColor Cyan
Write-Host "║       Designed for GPT-2 Inference on FPGA                 ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
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
    $outPath = Join-Path $root "sim" $OutputBin

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

    # Count PASS/FAIL from output
    $passes = ([regex]::Matches($simOutput, '\[PASS\]')).Count
    $fails = ([regex]::Matches($simOutput, '\[FAIL\]')).Count

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

# ── Phase 1: Core Compute Primitives ──
Write-Host "─── Phase 1: Core Compute Primitives ───" -ForegroundColor Yellow
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

# ── Phase 2: Extended Compute Modules ──
Write-Host "─── Phase 2: Extended Compute Modules ───" -ForegroundColor Yellow
Run-Test "P2" "mac_unit" "mac_test" @("rtl/compute/mac_unit.v") "tb/compute/mac_unit_tb.v"
Run-Test "P2" "systolic_array" "sa_test" @("rtl/compute/systolic_array.v") "tb/compute/systolic_array_tb.v"
Run-Test "P2" "gelu_activation" "gelu_test" @("rtl/compute/gelu_activation.v") "tb/compute/gelu_activation_tb.v"
Run-Test "P2" "softmax_unit" "sm_test" @("rtl/compute/softmax_unit.v") "tb/compute/softmax_unit_tb.v"
Write-Host ""

# ── Phase 3: Transformer Building Blocks ──
Write-Host "─── Phase 3: Transformer Building Blocks ───" -ForegroundColor Yellow
Run-Test "P3" "layer_norm" "ln_test" @("rtl/transformer/layer_norm.v") "tb/transformer/layer_norm_tb.v"
Run-Test "P3" "linear_layer" "ll_test" @("rtl/transformer/linear_layer.v") "tb/transformer/linear_layer_tb.v"
Run-Test "P3" "attention_unit" "au_test" @("rtl/transformer/attention_unit.v") "tb/transformer/attention_unit_tb.v"
Run-Test "P3" "ffn_block" "ffn_test" @("rtl/transformer/ffn_block.v") "tb/transformer/ffn_block_tb.v"
Write-Host ""

# ── Phase 4: GPT-2 Full Pipeline ──
Write-Host "─── Phase 4: GPT-2 Full Pipeline ───" -ForegroundColor Yellow
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
    "rtl/compute/softmax_unit.v"
) "tb/gpt2/gpt2_engine_tb.v"
Write-Host ""

# ── Summary ──
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    TEST RESULTS SUMMARY                    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Modules tested : $totalModules" 
Write-Host "  Total PASS     : $totalPass" -ForegroundColor Green
Write-Host "  Total FAIL     : $totalFail" -ForegroundColor $(if($totalFail -gt 0){"Red"}else{"Green"})
Write-Host ""

$results | Format-Table -AutoSize

if ($totalFail -eq 0) {
    Write-Host "  ✅ ALL TESTS PASSED — GPU READY FOR DEPLOYMENT" -ForegroundColor Green
} else {
    Write-Host "  ❌ SOME TESTS FAILED — REVIEW REQUIRED" -ForegroundColor Red
}
Write-Host ""
