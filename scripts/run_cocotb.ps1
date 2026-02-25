# ============================================================================
# run_cocotb.ps1 — Run Cocotb cosimulation on Windows (no make required)
# Usage: .\scripts\run_cocotb.ps1
# ============================================================================

$iverilog = "D:\Tools\iverilog\bin\iverilog.exe"
$vvp = "D:\Tools\iverilog\bin\vvp.exe"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir

# Step 1: Extract GPT-2 weights if not already done
$weightFile = Join-Path $root "weights\gpt2_real\gpt2_q88_weights.npz"
if (-not (Test-Path $weightFile)) {
    Write-Host "Extracting GPT-2 weights..." -ForegroundColor Cyan
    python "$root\scripts\extract_gpt2_weights.py"
    Write-Host ""
}

# Step 2: Set cocotb environment variables
$env:MODULE = "test_gpt2_cosim"
$env:TOPLEVEL = "gpt2_engine"
$env:TOPLEVEL_LANG = "verilog"
$env:COCOTB_REDUCED_LOG_FMT = "1"

# Step 3: Find cocotb-config
$cocotbLibDir = & python -c "import cocotb; import os; print(os.path.dirname(cocotb.__file__))" 2>$null
$cocotbVpiLib = "$cocotbLibDir\libs\icarus\cocotb_vpi.vpi"

if (-not $cocotbVpiLib -or -not (Test-Path $cocotbVpiLib -ErrorAction SilentlyContinue)) {
    # Alternative: find via cocotb-config
    $cocotbVpiLib = & cocotb-config --lib-name-path vpi icarus 2>$null
}

Write-Host "Cocotb VPI: $cocotbVpiLib" -ForegroundColor DarkGray

# Step 4: Compile Verilog
$sources = @(
    "$root\rtl\gpt2\embedding_lookup.v",
    "$root\rtl\gpt2\transformer_block.v",
    "$root\rtl\gpt2\gpt2_engine.v",
    "$root\rtl\transformer\layer_norm.v",
    "$root\rtl\transformer\attention_unit.v",
    "$root\rtl\transformer\ffn_block.v",
    "$root\rtl\transformer\linear_layer.v",
    "$root\rtl\compute\gelu_activation.v",
    "$root\rtl\compute\softmax_unit.v"
)

$buildDir = "$root\tb\cocotb\sim_build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$outBin = "$buildDir\gpt2_cosim"

Write-Host ""
Write-Host "Compiling BitbyBit GPU for cosimulation..." -ForegroundColor Cyan
& $iverilog -o $outBin -s gpt2_engine -g2005-sv @sources 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation FAILED!" -ForegroundColor Red
    exit 1
}
Write-Host "Compilation successful!" -ForegroundColor Green

# Step 5: Run cocotb via vvp
Write-Host ""
Write-Host "Running Cocotb cosimulation..." -ForegroundColor Yellow
Write-Host ""

$env:PYTHONPATH = "$root\tb\cocotb"

if ($cocotbVpiLib -and (Test-Path $cocotbVpiLib -ErrorAction SilentlyContinue)) {
    & $vvp -M (Split-Path $cocotbVpiLib) -m cocotb_vpi $outBin 2>&1
} else {
    Write-Host "Cocotb VPI not found. Running alternative Python-driven simulation..." -ForegroundColor Yellow
    python "$root\tb\cocotb\run_cosim_standalone.py"
}
