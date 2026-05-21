# BitbyBit Tri-Fold Prototype: Memory Primitives Verification Script
# This script compiles and runs the testbenches for skid_buffer and dual_port_lut.

$SCRIPT_DIR = $PSScriptRoot
$ROOT_DIR = (Get-Item $SCRIPT_DIR).Parent.FullName
$RTL_DIR = "$ROOT_DIR\rtl"
$TB_DIR = "$ROOT_DIR\tb\memory"
$OUTPUT_DIR = "$ROOT_DIR\sim"

# Ensure output directory exists
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

function Run-Test {
    param (
        [string]$Name,
        [string]$RtlFile,
        [string]$TbFile
    )

    Write-Host "--------------------------------------------------" -ForegroundColor Gray
    Write-Host "Verifying $Name..." -ForegroundColor Cyan
    
    $OutputFile = "$OUTPUT_DIR\$Name.vvp"
    
    # Compile using iverilog
    Write-Host "Compiling..." -NoNewline
    & iverilog -g2012 -o $OutputFile $RtlFile $TbFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAILED (Compilation Error)" -ForegroundColor Red
        return $false
    }
    Write-Host " OK" -ForegroundColor Green

    # Run using vvp
    Write-Host "Executing simulation..."
    # We change to OUTPUT_DIR so VCD files are generated there
    Push-Location $OUTPUT_DIR
    try {
        $results = & vvp "$Name.vvp"
    } finally {
        Pop-Location
    }

    $results | ForEach-Object { Write-Host "  $_" }

    # Check for PASS/FAIL in output
    $has_pass = ($results -match "PASS:")
    $has_fail = ($results -match "FAIL:") -or ($results -match "ERROR:")

    if ($has_pass -and -not $has_fail) {
        Write-Host "RESULT: $Name passed." -ForegroundColor Green
        return $true
    } else {
        Write-Host "RESULT: $Name FAILED." -ForegroundColor Red
        return $false
    }
}

$all_passed = $true

# Test 1: Skid Buffer
if (-not (Run-Test -Name "skid_buffer" -RtlFile "$RTL_DIR\integration\skid_buffer.v" -TbFile "$TB_DIR\skid_buffer_tb.v")) {
    $all_passed = $false
}

# Test 2: Dual Port LUT
if (-not (Run-Test -Name "dual_port_lut" -RtlFile "$RTL_DIR\memory\dual_port_lut.v" -TbFile "$TB_DIR\dual_port_lut_tb.v")) {
    $all_passed = $false
}

# Test 3: Ternary Unpacker
if (-not (Run-Test -Name "ternary_unpacker" -RtlFile "$RTL_DIR\memory\ternary_unpacker.v" -TbFile "$TB_DIR\ternary_unpacker_tb.v")) {
    $all_passed = $false
}

# Test 4: Mixed Memory Controller
if (-not (Run-Test -Name "mem_controller_mixed" -RtlFile "$RTL_DIR\memory\mem_controller_mixed.v" -TbFile "$TB_DIR\mem_controller_mixed_tb.v")) {
    $all_passed = $false
}

Write-Host "--------------------------------------------------" -ForegroundColor Gray
Write-Host "Summary:" -ForegroundColor White
if ($all_passed) {
    Write-Host "OVERALL VERIFICATION: PASS" -ForegroundColor Green
    exit 0
} else {
    Write-Host "OVERALL VERIFICATION: FAIL" -ForegroundColor Red
    exit 1
}
