# ============================================================================
# run_demo.ps1 - Science Fest Live Demo
# Compiles and runs the GPT-2 pipeline demo with formatted output
# ============================================================================

$iverilog = "D:\Tools\iverilog\bin\iverilog.exe"
$vvp = "D:\Tools\iverilog\bin\vvp.exe"

# Get project root (parent of scripts folder)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir

$sources = @(
    "$root\rtl\gpt2\embedding_lookup.v",
    "$root\rtl\gpt2\transformer_block.v",
    "$root\rtl\gpt2\gpt2_engine.v",
    "$root\rtl\transformer\layer_norm.v",
    "$root\rtl\transformer\attention_unit.v",
    "$root\rtl\transformer\ffn_block.v",
    "$root\rtl\transformer\linear_layer.v",
    "$root\rtl\compute\gelu_activation.v",
    "$root\rtl\compute\softmax_unit.v",
    "$root\tb\demo\gpt2_demo_tb.v"
)

$outBin = "$root\sim\gpt2_demo"

Write-Host ""
Write-Host "Compiling BitbyBit GPU..." -ForegroundColor Cyan

& $iverilog -o $outBin @sources 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation FAILED!" -ForegroundColor Red
    exit 1
}

Write-Host "Running GPT-2 inference demo..." -ForegroundColor Green
Write-Host ""

& $vvp $outBin 2>&1
