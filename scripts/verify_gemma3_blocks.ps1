$ErrorActionPreference = "Stop"

Set-Location -Path "D:\projects\bitbybit\custom_gpu_project"

function Run-Test {
    param(
        [string]$TestName,
        [string[]]$Files,
        [string]$SuccessRegex
    )
    
    Write-Host "Running $TestName..."
    
    # Compile
    $iverilogArgs = @("-o", "$TestName.vvp") + $Files
    & iverilog $iverilogArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$TestName] FAIL - Compilation Error" -ForegroundColor Red
        return $false
    }
    
    # Run
    $output = & vvp "$TestName.vvp"
    
    # Check
    $passed = $false
    foreach ($line in $output) {
        if ($line -match $SuccessRegex) {
            $passed = $true
            break
        }
    }
    
    if ($passed) {
        Write-Host "[$TestName] PASS" -ForegroundColor Green
        return $true
    } else {
        Write-Host "[$TestName] FAIL - Test output did not match expected success criteria" -ForegroundColor Red
        Write-Host "Output:"
        $output | Out-String | Write-Host
        return $false
    }
}

$all_passed = $true

# Test 1: rmsnorm_vp
$res1 = Run-Test -TestName "rmsnorm_vp" -Files @("rtl/compute/rmsnorm_vp.v", "rtl/compute/inv_sqrt_lut_256.v", "tb/compute/rmsnorm_vp_tb.v") -SuccessRegex "TEST PASSED!"
if (-not $res1) { $all_passed = $false }

# Test 2: rope_unit_v2
$res2 = Run-Test -TestName "rope_unit_v2" -Files @("rtl/transformer/rope_unit_v2.v", "rtl/memory/dual_port_lut.v", "tb/transformer/rope_unit_v2_tb.v") -SuccessRegex "SUCCESS: All tests passed!"
if (-not $res2) { $all_passed = $false }

# Test 3: gated_mlp_da
$res3 = Run-Test -TestName "gated_mlp_da" -Files @("rtl/transformer/gated_mlp_da.v", "rtl/compute/gelu_lut_256.v", "tb/transformer/gated_mlp_da_tb.v") -SuccessRegex "PASS: Gated MLP Output matches golden model perfectly!"
if (-not $res3) { $all_passed = $false }

if ($all_passed) {
    Write-Host "`nAll blocks passed verification!" -ForegroundColor Green
} else {
    Write-Host "`nSome blocks failed verification!" -ForegroundColor Red
}
