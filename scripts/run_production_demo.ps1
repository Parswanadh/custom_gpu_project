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
$root = Split-Path -Parent $scriptDir
$runDemo = Join-Path $scriptDir "run_demo.ps1"
$proofPack = Join-Path $scriptDir "build_phase3_benchmark_proof_pack.py"
$validator = Join-Path $scriptDir "validate_benchmark_payload.py"
$compareSummary = Join-Path $root "sim\compare_summary_latest.json"
$proofPackJson = Join-Path $root "sim\phase3_benchmark_proof_pack.json"
$proofPackCsv = Join-Path $root "sim\phase3_benchmark_proof_pack.csv"
$schemaPath = Join-Path $root "sim\benchmark_schema.json"

if (!(Test-Path $runDemo)) {
    throw "Missing required script: $runDemo"
}
if (!(Test-Path $proofPack)) {
    throw "Missing required script: $proofPack"
}
if (!(Test-Path $validator)) {
    throw "Missing required script: $validator"
}
if (!(Test-Path $schemaPath)) {
    throw "Missing required schema: $schemaPath"
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

Write-Host "Validating benchmark payload + proof-pack contract..." -ForegroundColor Yellow
python $validator --input $compareSummary --schema $schemaPath --proof-pack $proofPackJson
if ($LASTEXITCODE -ne 0) {
    throw "validate_benchmark_payload.py failed with exit code $LASTEXITCODE"
}

if (!(Test-Path $proofPackJson)) {
    throw "Missing proof-pack JSON after build: $proofPackJson"
}
if (!(Test-Path $proofPackCsv)) {
    throw "Missing proof-pack CSV after build: $proofPackCsv"
}

Write-Host "Production demo complete." -ForegroundColor Green
Write-Host "Artifacts:" -ForegroundColor Green
Write-Host "  sim\\compare_summary_latest.json"
Write-Host "  sim\\phase3_benchmark_proof_pack.json"
Write-Host "  sim\\phase3_benchmark_proof_pack.csv"
Write-Host "  sim\\benchmark_schema.json"
