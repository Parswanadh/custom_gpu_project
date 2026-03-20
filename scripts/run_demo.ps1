# ============================================================================
# run_demo.ps1 - Compatibility wrapper for live simulation demos.
# Delegates to demo_day.ps1 so demo paths stay aligned with maintained benches.
# ============================================================================

[CmdletBinding()]
param(
    [ValidateSet("base", "imprint", "top", "compare", "all")]
    [string]$Mode = "compare",
    [int]$TokenId = 5,
    [int]$Position = 2,
    [ValidateSet("single", "matrix")]
    [string]$WorkloadMode = "matrix",
    [int]$WarmupRuns = 3,
    [int]$MeasuredRuns = 10,
    [int]$TokenSpace = 16,
    [int]$PositionSpace = 8,
    [int]$WorkloadCount = 3,
    [int]$WorkloadSeed = 1337
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$demoScript = Join-Path $scriptDir "demo_day.ps1"
if (!(Test-Path $demoScript)) {
    throw "Missing required script: $demoScript"
}

Write-Host "run_demo.ps1 -> demo_day.ps1 (Mode=$Mode, TokenId=$TokenId, Position=$Position, Workload=$WorkloadMode, Warmup=$WarmupRuns, Measured=$MeasuredRuns, TokenSpace=$TokenSpace, PositionSpace=$PositionSpace, WorkloadCount=$WorkloadCount, Seed=$WorkloadSeed)" -ForegroundColor Cyan

& powershell -ExecutionPolicy Bypass -File $demoScript `
    -Mode $Mode `
    -TokenId $TokenId `
    -Position $Position `
    -WorkloadMode $WorkloadMode `
    -WarmupRuns $WarmupRuns `
    -MeasuredRuns $MeasuredRuns `
    -TokenSpace $TokenSpace `
    -PositionSpace $PositionSpace `
    -WorkloadCount $WorkloadCount `
    -WorkloadSeed $WorkloadSeed
exit $LASTEXITCODE
