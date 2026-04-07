# ============================================================================
# validate_full_chain.ps1
# Canonical fail-closed orchestrator:
#   1) Full regression
#   2) Production benchmark + proof-pack + contract validator
#   3) WS1 parity enforce gate
#   4) Optional website integrity checks
# ============================================================================

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$SkipWebsite,
    [switch]$SkipWS1
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir
$workspaceRoot = Split-Path -Parent $root
$autoWebsite = Join-Path $workspaceRoot "auto-git-website"
$simDir = Join-Path $root "sim"
$manifestPath = Join-Path $simDir "validation_manifest_latest.json"

$runAllTests = Join-Path $scriptDir "run_all_tests.ps1"
$runProduction = Join-Path $scriptDir "run_production_demo.ps1"
$runWs1 = Join-Path $scriptDir "run_ws1_scale_proof.py"
$compareSummary = Join-Path $simDir "compare_summary_latest.json"
$proofPackJson = Join-Path $simDir "phase3_benchmark_proof_pack.json"

$assertPath = {
    param(
        [string]$Path,
        [string]$Label
    )

    if (!(Test-Path $Path)) {
        throw ("Missing required path: " + $Label + " -> " + $Path)
    }
}

$runStage = {
    param(
        [string]$Name,
        [scriptblock]$Command,
        [hashtable]$Stages
    )

    Write-Host "" -ForegroundColor Gray
    Write-Host "=== [$Name] START ===" -ForegroundColor Cyan

    $start = Get-Date
    try {
        & $Command
        $duration = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
        $Stages[$Name] = @{ status = "PASS"; duration_seconds = $duration }
        Write-Host "=== [$Name] PASS (${duration}s) ===" -ForegroundColor Green
    } catch {
        $duration = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
        $Stages[$Name] = @{ status = "FAIL"; duration_seconds = $duration; error = $_.Exception.Message }
        Write-Host "=== [$Name] FAIL (${duration}s) ===" -ForegroundColor Red
        throw
    }
}

& $assertPath -Path $runAllTests -Label "full regression runner"
& $assertPath -Path $runProduction -Label "production demo runner"
& $assertPath -Path $runWs1 -Label "WS1 runner"

if (!(Test-Path $simDir)) {
    New-Item -ItemType Directory -Path $simDir -Force | Out-Null
}

if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python is not available in PATH"
}

$stages = @{}
$commands = @()

Push-Location $root
try {
    $regressionCmd = "powershell -ExecutionPolicy Bypass -File scripts/run_all_tests.ps1"
    $commands += $regressionCmd
    & $runStage -Name "full_regression" -Stages $stages -Command {
        & powershell -ExecutionPolicy Bypass -File $runAllTests
        if ($LASTEXITCODE -ne 0) {
            throw "run_all_tests.ps1 failed with exit code $LASTEXITCODE"
        }
    }

    $productionArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $runProduction,
        "-WorkloadMode", "matrix",
        "-WarmupRuns", "3",
        "-MeasuredRuns", "10",
        "-WorkloadCount", "8",
        "-WorkloadSeed", "20260331"
    )

    if (!$Quick) {
        $productionArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", $runProduction,
            "-WorkloadMode", "matrix",
            "-WarmupRuns", "5",
            "-MeasuredRuns", "20",
            "-WorkloadCount", "20",
            "-WorkloadSeed", "20260331"
        )
    }

    $commands += ("powershell " + ($productionArgs -join " "))
    & $runStage -Name "production_benchmark" -Stages $stages -Command {
        & powershell @productionArgs
        if ($LASTEXITCODE -ne 0) {
            throw "run_production_demo.ps1 failed with exit code $LASTEXITCODE"
        }
        & $assertPath -Path $compareSummary -Label "compare summary"
        & $assertPath -Path $proofPackJson -Label "proof-pack JSON"
    }

    if (!$SkipWS1) {
        $ws1Args = @(
            $runWs1,
            "--dims", "16,32,64",
            "--workload-count", "24",
            "--workload-seed", "20260331",
            "--token-space", "16",
            "--position-space", "8",
            "--seq-len", "32",
            "--enforce-gate"
        )

        $commands += ("python " + ($ws1Args -join " "))
        & $runStage -Name "ws1_parity_gate" -Stages $stages -Command {
            & python @ws1Args
            if ($LASTEXITCODE -ne 0) {
                throw "run_ws1_scale_proof.py --enforce-gate failed with exit code $LASTEXITCODE"
            }
        }
    } else {
        $stages["ws1_parity_gate"] = @{ status = "SKIPPED"; reason = "SkipWS1 flag" }
    }

    if (!$SkipWebsite -and (Test-Path $autoWebsite)) {
        Push-Location $autoWebsite
        try {
            $commands += "npx vitest run"
            & $runStage -Name "website_vitest" -Stages $stages -Command {
                & npx vitest run
                if ($LASTEXITCODE -ne 0) {
                    throw "vitest failed with exit code $LASTEXITCODE"
                }
            }

            $playwrightArgs = @(
                "test",
                "tests/e2e/homepage.spec.ts",
                "--reporter=line"
            )

            if ($Quick) {
                $playwrightArgs += "--project=chromium"
            } else {
                $playwrightArgs += @("--project=chromium", "--project=Mobile Chrome")
            }

            $commands += ("npx playwright " + ($playwrightArgs -join " "))
            & $runStage -Name "website_playwright_suite" -Stages $stages -Command {
                & npx playwright @playwrightArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "playwright homepage suite failed with exit code $LASTEXITCODE"
                }
            }
        } finally {
            Pop-Location
        }
    } else {
        $stages["website_vitest"] = @{ status = "SKIPPED"; reason = "SkipWebsite flag or auto-git-website missing" }
        $stages["website_playwright_suite"] = @{ status = "SKIPPED"; reason = "SkipWebsite flag or auto-git-website missing" }
    }
}
finally {
    Pop-Location
}

$runId = $null
$benchmarkMeta = @{}
if (Test-Path $compareSummary) {
    try {
        $summary = Get-Content $compareSummary -Raw | ConvertFrom-Json
        $runId = $summary.run_id
        $benchmarkMeta = @{
            workload_count_effective = $summary.workload_count_effective
            measured_runs = $summary.measured_runs
        }
    } catch {
        $benchmarkMeta = @{ parse_error = $_.Exception.Message }
    }
}

$manifest = [ordered]@{
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    root = $root
    quick_mode = [bool]$Quick
    skip_website = [bool]$SkipWebsite
    skip_ws1 = [bool]$SkipWS1
    run_id = $runId
    benchmark_meta = $benchmarkMeta
    commands = $commands
    stages = $stages
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

$failedStages = @($stages.GetEnumerator() | Where-Object { $_.Value.status -eq "FAIL" })
if ($failedStages.Count -gt 0) {
    Write-Host "Validation chain finished with failures. See: $manifestPath" -ForegroundColor Red
    exit 1
}

Write-Host "Validation chain completed successfully. Manifest: $manifestPath" -ForegroundColor Green
