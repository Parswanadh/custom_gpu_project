# ============================================================================
# run_production_demo.ps1
# Canonical reviewer/judge demo flow:
#   1) Run top-level + full-model compare benchmarks
#   2) Regenerate machine-readable benchmark proof pack
# ============================================================================

[CmdletBinding()]
param(
    [ValidateSet("single", "matrix")]
    [string]$WorkloadMode = "matrix",
    [int]$TokenId = 5,
    [int]$Position = 2,
    [int]$WarmupRuns = 3,
    [int]$MeasuredRuns = 10,
    [int]$TokenSpace = 16,
    [int]$PositionSpace = 8,
    [int]$WorkloadCount = 6,
    [int]$WorkloadSeed = 1337
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDemo = Join-Path $scriptDir "run_demo.ps1"
$proofPack = Join-Path $scriptDir "build_phase3_benchmark_proof_pack.py"

if (!(Test-Path $runDemo)) {
    throw "Missing required script: $runDemo"
}
if (!(Test-Path $proofPack)) {
    throw "Missing required script: $proofPack"
}

Write-Host "=== BitbyBit Production Demo Flow ===" -ForegroundColor Cyan
Write-Host ("Running benchmark demo: mode=all workload={0} warmup={1} measured={2} workloadCount={3} seed={4}" -f $WorkloadMode, $WarmupRuns, $MeasuredRuns, $WorkloadCount, $WorkloadSeed) -ForegroundColor Yellow

& powershell -ExecutionPolicy Bypass -File $runDemo `
    -Mode all `
    -WorkloadMode $WorkloadMode `
    -TokenId $TokenId `
    -Position $Position `
    -WarmupRuns $WarmupRuns `
    -MeasuredRuns $MeasuredRuns `
    -TokenSpace $TokenSpace `
    -PositionSpace $PositionSpace `
    -WorkloadCount $WorkloadCount `
    -WorkloadSeed $WorkloadSeed

if ($LASTEXITCODE -ne 0) {
    throw "run_demo.ps1 failed with exit code $LASTEXITCODE"
}

Write-Host "Regenerating benchmark proof pack..." -ForegroundColor Yellow
python $proofPack
if ($LASTEXITCODE -ne 0) {
    throw "build_phase3_benchmark_proof_pack.py failed with exit code $LASTEXITCODE"
}

Write-Host "Production demo complete." -ForegroundColor Green
Write-Host "Artifacts:" -ForegroundColor Green
Write-Host "  sim\\compare_summary_latest.json"
Write-Host "  sim\\phase3_benchmark_proof_pack.json"
Write-Host "  sim\\phase3_benchmark_proof_pack.csv"
