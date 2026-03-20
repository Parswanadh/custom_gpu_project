# ============================================================================
# demo_day.ps1
# One-command simulation demo launcher for base vs imprint model runs.
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

function Resolve-ToolPath {
    param(
        [string]$EnvVar,
        [string]$DefaultPath,
        [string]$FallbackCommand
    )

    $explicit = [Environment]::GetEnvironmentVariable($EnvVar)
    if ($explicit) {
        if (Test-Path $explicit) { return $explicit }
        throw "$EnvVar is set but not found: $explicit"
    }

    if (Test-Path $DefaultPath) { return $DefaultPath }

    $cmd = Get-Command $FallbackCommand -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Path }

    throw "Unable to locate $FallbackCommand. Set $EnvVar or install the tool."
}

function Invoke-SimBench {
    param(
        [string]$Name,
        [string[]]$Sources,
        [string]$Testbench,
        [string[]]$PlusArgs = @()
    )

    $simDir = Join-Path $root "sim"
    if (!(Test-Path $simDir)) {
        New-Item -ItemType Directory -Path $simDir | Out-Null
    }

    $outBin = Join-Path $simDir "$Name.vvp"
    $logFile = Join-Path $simDir "$Name.log"
    $allRelFiles = $Sources + @($Testbench)
    $allFiles = $allRelFiles | ForEach-Object { Join-Path $root $_ }

    Write-Host ""
    Write-Host "Compiling $Name ..." -ForegroundColor Cyan
    $compileOutput = & $iverilog -g2012 -o $outBin @allFiles 2>&1
    $compileExit = $LASTEXITCODE
    if ($compileOutput) { $compileOutput | Write-Host }
    if ($compileExit -ne 0) {
        throw "Compile failed for $Name"
    }

    Write-Host "Running $Name ..." -ForegroundColor Green
    $runOutput = & $vvp $outBin @PlusArgs 2>&1
    $runExit = $LASTEXITCODE
    $runOutput | Tee-Object -FilePath $logFile | Out-Host
    if ($runExit -ne 0) {
        throw "Simulation failed for $Name"
    }

    return $logFile
}

function Get-RegexInt {
    param(
        [string]$Text,
        [string]$Pattern
    )
    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($m.Success) { return [int]$m.Groups[1].Value }
    return $null
}

function Get-RegexDouble {
    param(
        [string]$Text,
        [string]$Pattern
    )
    $m = [regex]::Match($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

function Get-NumericStats {
    param(
        [double[]]$Values
    )
    if (-not $Values -or $Values.Count -eq 0) { return $null }
    $avg = ($Values | Measure-Object -Average).Average
    $min = ($Values | Measure-Object -Minimum).Minimum
    $max = ($Values | Measure-Object -Maximum).Maximum
    $median = ($Values | Sort-Object)[[int][math]::Floor(($Values.Count - 1) / 2)]
    $stdev = $null
    $cv = $null
    $stderr = $null
    $ci95Lower = $null
    $ci95Upper = $null
    if ($Values.Count -gt 1) {
        $variance = ($Values | ForEach-Object { [math]::Pow(($_ - $avg), 2) } | Measure-Object -Average).Average
        $stdev = [math]::Round([math]::Sqrt($variance), 4)
        if ([double]$avg -ne 0.0) {
            $cv = [math]::Round(([double]$stdev / [double]$avg) * 100.0, 4)
        }
        $stderr = [math]::Round(([double]$stdev / [math]::Sqrt([double]$Values.Count)), 4)
        $ci95Lower = [math]::Round(([double]$avg - (1.96 * [double]$stderr)), 4)
        $ci95Upper = [math]::Round(([double]$avg + (1.96 * [double]$stderr)), 4)
    }
    return [pscustomobject]@{
        Count        = $Values.Count
        Mean         = [math]::Round([double]$avg, 4)
        Median       = [math]::Round([double]$median, 4)
        StDev        = $stdev
        CoeffVarPct  = $cv
        Min          = [double]$min
        Max          = [double]$max
        StdErr       = $stderr
        CI95Lower    = $ci95Lower
        CI95Upper    = $ci95Upper
    }
}

function Get-StabilityDiagnostic {
    param(
        [object[]]$Samples
    )
    if (-not $Samples -or $Samples.Count -lt 4) { return $null }
    $speedups = $Samples | ForEach-Object { [double]$_.speedup_x }
    $mid = [int][math]::Floor($speedups.Count / 2)
    $firstHalf = $speedups[0..($mid - 1)]
    $secondHalf = $speedups[$mid..($speedups.Count - 1)]
    $firstMean = ($firstHalf | Measure-Object -Average).Average
    $secondMean = ($secondHalf | Measure-Object -Average).Average
    $driftPct = $null
    if ([double]$firstMean -ne 0.0) {
        $driftPct = [math]::Round(([math]::Abs($secondMean - $firstMean) / [double]$firstMean) * 100.0, 4)
    }
    return [ordered]@{
        first_half_mean = [math]::Round([double]$firstMean, 4)
        second_half_mean = [math]::Round([double]$secondMean, 4)
        drift_percent = $driftPct
        stable = ($driftPct -ne $null -and $driftPct -lt 5.0)
    }
}

function Get-FileSHA256 {
    param(
        [string]$Path
    )
    if (Test-Path $Path) {
        return (Get-FileHash -Path $Path -Algorithm SHA256).Hash
    }
    return $null
}

# Root directory = parent of scripts/
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $scriptDir

$iverilog = Resolve-ToolPath -EnvVar "BITBYBIT_IVERILOG" -DefaultPath "D:\Tools\iverilog\bin\iverilog.exe" -FallbackCommand "iverilog"
$vvp = Resolve-ToolPath -EnvVar "BITBYBIT_VVP" -DefaultPath "D:\Tools\iverilog\bin\vvp.exe" -FallbackCommand "vvp"

Write-Host "================================================================" -ForegroundColor Yellow
Write-Host "  BitbyBit GPU Demo Launcher (Simulation)" -ForegroundColor Yellow
$workloadExtra = if ($WorkloadMode -eq "matrix") { " | WorkloadCount: $WorkloadCount | Seed: $WorkloadSeed" } else { "" }
Write-Host "  Mode: $Mode | Workload: $WorkloadMode | TokenId: $TokenId | Position: $Position | Warmup: $WarmupRuns | Measured: $MeasuredRuns$workloadExtra" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Yellow

$topSources = @(
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
)

$baseSources = @(
    "rtl/gpt2/embedding_lookup.v",
    "rtl/integration/optimized_transformer_layer.v",
    "rtl/transformer/rope_encoder.v",
    "rtl/transformer/grouped_query_attention.v",
    "rtl/compute/parallel_softmax.v",
    "rtl/compute/exp_lut_256.v",
    "rtl/compute/recip_lut_256.v",
    "rtl/compute/gelu_activation.v",
    "rtl/compute/gelu_lut_256.v",
    "rtl/memory/kv_cache_quantizer.v",
    "rtl/compute/activation_compressor.v",
    "rtl/compute/medusa_head_predictor.v"
)

$imprintSources = @(
    "rtl/memory/imprinted_embedding_rom.v",
    "rtl/integration/imprinted_mini_transformer_core.v",
    "rtl/compute/medusa_head_predictor.v"
)

$topLog = $null
$baseLog = $null
$imprintLog = $null
$rows = @()
$pairedMeasuredSamples = @()
$runBundleManifestPath = $null

if ($MeasuredRuns -lt 1) {
    throw "MeasuredRuns must be >= 1"
}
if ($MeasuredRuns -lt 10) {
    Write-Warning "MeasuredRuns=$MeasuredRuns is below recommended statistical minimum of 10."
}
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be >= 0"
}
if ($WarmupRuns -lt 3) {
    Write-Warning "WarmupRuns=$WarmupRuns is below recommended stabilization minimum of 3."
}
if ($TokenSpace -lt 1) {
    throw "TokenSpace must be >= 1"
}
if ($PositionSpace -lt 1) {
    throw "PositionSpace must be >= 1"
}
if ($WorkloadCount -lt 1) {
    throw "WorkloadCount must be >= 1"
}

if ($Mode -in @("top", "compare", "all")) {
    $topLog = Invoke-SimBench `
        -Name "demo_gpu_system_top_v2" `
        -Sources $topSources `
        -Testbench "tb/top/gpu_system_top_v2_tb.v"
}

if (($Mode -in @("compare", "all")) -and (($WarmupRuns -gt 0) -or ($MeasuredRuns -gt 1) -or ($WorkloadMode -eq "matrix"))) {
    $totalRuns = $WarmupRuns + $MeasuredRuns
    $runId = Get-Date -Format "yyyyMMdd-HHmmss"
    $runDir = Join-Path $root "sim\bench_runs\$runId"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $workloads = @()
    if ($WorkloadMode -eq "matrix") {
        $maxUnique = $TokenSpace * $PositionSpace
        $effectiveWorkloadCount = [Math]::Min($WorkloadCount, $maxUnique)
        if ($WorkloadCount -gt $maxUnique) {
            Write-Warning ("WorkloadCount={0} exceeds unique token/position space ({1}); clamping to {1}." -f $WorkloadCount, $maxUnique)
        }

        $seenWorkloads = @{}
        $primaryWorkload = [pscustomobject]@{
            TokenId = ($TokenId % $TokenSpace)
            Position = ($Position % $PositionSpace)
        }
        $primaryKey = "{0}:{1}" -f $primaryWorkload.TokenId, $primaryWorkload.Position
        $seenWorkloads[$primaryKey] = $true
        $workloads += $primaryWorkload

        if ($effectiveWorkloadCount -gt 1) {
            $rng = New-Object System.Random($WorkloadSeed)
            $attempt = 0
            $maxAttempts = [Math]::Max(64, $maxUnique * 8)
            while ($workloads.Count -lt $effectiveWorkloadCount -and $attempt -lt $maxAttempts) {
                $candToken = $rng.Next(0, $TokenSpace)
                $candPos = $rng.Next(0, $PositionSpace)
                $wlKey = "{0}:{1}" -f $candToken, $candPos
                if (-not $seenWorkloads.ContainsKey($wlKey)) {
                    $seenWorkloads[$wlKey] = $true
                    $workloads += [pscustomobject]@{ TokenId = $candToken; Position = $candPos }
                }
                $attempt = $attempt + 1
            }

            # Deterministic fallback scan guarantees completion when random sampling collides.
            if ($workloads.Count -lt $effectiveWorkloadCount) {
                for ($t = 0; $t -lt $TokenSpace -and $workloads.Count -lt $effectiveWorkloadCount; $t++) {
                    for ($p = 0; $p -lt $PositionSpace -and $workloads.Count -lt $effectiveWorkloadCount; $p++) {
                        $wlKey = "{0}:{1}" -f $t, $p
                        if (-not $seenWorkloads.ContainsKey($wlKey)) {
                            $seenWorkloads[$wlKey] = $true
                            $workloads += [pscustomobject]@{ TokenId = $t; Position = $p }
                        }
                    }
                }
            }
        }

        if ($workloads.Count -lt $effectiveWorkloadCount) {
            throw ("Insufficient unique workloads for matrix mode: have {0}, requested {1}." -f $workloads.Count, $effectiveWorkloadCount)
        }
        $coveragePct = [math]::Round((100.0 * $workloads.Count) / $maxUnique, 4)
        Write-Host ("Matrix workload set: {0} unique points (coverage={1}% seed={2})" -f $workloads.Count, $coveragePct, $WorkloadSeed) -ForegroundColor DarkYellow
    } else {
        $workloads += [pscustomobject]@{ TokenId = ($TokenId % $TokenSpace); Position = ($Position % $PositionSpace) }
    }

    $workloadIndex = 0
    foreach ($wl in $workloads) {
        $workloadIndex = $workloadIndex + 1
        $wlToken = [int]$wl.TokenId
        $wlPos = [int]$wl.Position

        for ($runIdx = 1; $runIdx -le $totalRuns; $runIdx++) {
            $isMeasured = ($runIdx -gt $WarmupRuns)
            $phaseLabel = if ($isMeasured) { "measured" } else { "warmup" }
            Write-Host ("Running paired compare workload {0}/{1}, pass {2}/{3} ({4}) [token={5}, pos={6}]..." -f `
                $workloadIndex, $workloads.Count, $runIdx, $totalRuns, $phaseLabel, $wlToken, $wlPos) -ForegroundColor Cyan

            $baseRunLog = Invoke-SimBench `
                -Name ("demo_full_model_base_w{0}_r{1}" -f $workloadIndex, $runIdx) `
                -Sources $baseSources `
                -Testbench "tb/integration/full_model_inference_tb.v" `
                -PlusArgs @("+TOKEN_ID=$wlToken", "+POSITION=$wlPos")

            $imprintRunLog = Invoke-SimBench `
                -Name ("demo_full_model_imprint_w{0}_r{1}" -f $workloadIndex, $runIdx) `
                -Sources $imprintSources `
                -Testbench "tb/integration/full_model_inference_imprint_tb.v" `
                -PlusArgs @("+TOKEN_ID=$wlToken", "+POSITION=$wlPos")

            $baseTextRun = Get-Content $baseRunLog -Raw
            $imprintTextRun = Get-Content $imprintRunLog -Raw

            $baseCyclesRun = Get-RegexInt -Text $baseTextRun -Pattern "TOTAL INFERENCE\s+(\d+)\s+cy"
            $imprintCyclesRun = Get-RegexInt -Text $imprintTextRun -Pattern "TOTAL INFERENCE\s+(\d+)\s+cy"
            $baseTpsRun = Get-RegexInt -Text $baseTextRun -Pattern "Tokens/second:\s+~(\d+)"
            $imprintTpsRun = Get-RegexInt -Text $imprintTextRun -Pattern "Tokens/second:\s+~(\d+)"
            $baseMedusaTpsRun = Get-RegexInt -Text $baseTextRun -Pattern "effective\s+(\d+)\s+tok/s"
            $imprintMedusaTpsRun = Get-RegexInt -Text $imprintTextRun -Pattern "effective\s+(\d+)\s+tok/s"

            if ($baseCyclesRun -le 0 -or $imprintCyclesRun -le 0) {
                throw ("Invalid compare sample at workload {0}, run {1}: base={2} imprint={3}" -f `
                    $workloadIndex, $runIdx, $baseCyclesRun, $imprintCyclesRun)
            }

            if ($isMeasured) {
                if (-not $baseLog) { $baseLog = $baseRunLog }
                if (-not $imprintLog) { $imprintLog = $imprintRunLog }
                $pairedMeasuredSamples += [pscustomobject]@{
                    workload_index = $workloadIndex
                    run_index = $runIdx
                    phase = $phaseLabel
                    token_id = $wlToken
                    position = $wlPos
                    base_cycles = $baseCyclesRun
                    imprint_cycles = $imprintCyclesRun
                    base_tokens_per_second = $baseTpsRun
                    imprint_tokens_per_second = $imprintTpsRun
                    base_medusa_toks = $baseMedusaTpsRun
                    imprint_medusa_toks = $imprintMedusaTpsRun
                    speedup_x = [math]::Round(($baseCyclesRun / [double]$imprintCyclesRun), 4)
                }
            }
        }
    }

    if ($pairedMeasuredSamples.Count -eq 0) {
        throw "No measured samples were collected."
    }

    $baseCycleStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.base_cycles })
    $imprintCycleStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.imprint_cycles })
    $baseTpsStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.base_tokens_per_second })
    $imprintTpsStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.imprint_tokens_per_second })
    $baseMedusaStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.base_medusa_toks })
    $imprintMedusaStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.imprint_medusa_toks })
    $speedupStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.speedup_x })

    $rows += [pscustomobject]@{
        Variant = "Base full-model (mean)"
        CyclesPerToken = [int][math]::Round($baseCycleStats.Mean)
        TokensPerSecond = [int][math]::Round($baseTpsStats.Mean)
        MedusaEffectiveTokS = [int][math]::Round($baseMedusaStats.Mean)
    }
    $rows += [pscustomobject]@{
        Variant = "MINI imprint full-model (mean)"
        CyclesPerToken = [int][math]::Round($imprintCycleStats.Mean)
        TokensPerSecond = [int][math]::Round($imprintTpsStats.Mean)
        MedusaEffectiveTokS = [int][math]::Round($imprintMedusaStats.Mean)
    }

    $compareSummary = [ordered]@{
        benchmark_version = "phase3-v2"
        benchmark_frequency_hz = 100000000
        script = "scripts/demo_day.ps1"
        generated_utc = (Get-Date).ToUniversalTime().ToString("o")
        tools = [ordered]@{
            iverilog = $iverilog
            vvp = $vvp
        }
        run_id = $runId
        workload_mode = $WorkloadMode
        workload_generation = if ($WorkloadMode -eq "matrix") { "seeded_random_unique" } else { "single_fixed" }
        workload_count_requested = if ($WorkloadMode -eq "matrix") { $WorkloadCount } else { 1 }
        workload_count_effective = $workloads.Count
        workload_seed = if ($WorkloadMode -eq "matrix") { $WorkloadSeed } else { $null }
        token_space = $TokenSpace
        position_space = $PositionSpace
        token_id = if ($workloads.Count -eq 1) { [int]$workloads[0].TokenId } else { $null }
        position = if ($workloads.Count -eq 1) { [int]$workloads[0].Position } else { $null }
        workloads = $workloads
        warmup_runs = $WarmupRuns
        measured_runs = $MeasuredRuns
        samples = $pairedMeasuredSamples
        stats = [ordered]@{
            base_cycles = $baseCycleStats
            imprint_cycles = $imprintCycleStats
            base_tokens_per_second = $baseTpsStats
            imprint_tokens_per_second = $imprintTpsStats
            base_medusa_tokens_per_second = $baseMedusaStats
            imprint_medusa_tokens_per_second = $imprintMedusaStats
            speedup_x = $speedupStats
        }
    }
    $stabilityDiag = Get-StabilityDiagnostic -Samples $pairedMeasuredSamples
    if ($stabilityDiag) {
        $compareSummary["stability_diagnostic"] = $stabilityDiag
    }
    $compareSummary["minimum_recommended_runs"] = [ordered]@{
        warmup_runs = 3
        measured_runs = 10
    }
    $compareSummary["run_quality"] = [ordered]@{
        warmup_meets_recommendation = ($WarmupRuns -ge 3)
        measured_meets_recommendation = ($MeasuredRuns -ge 10)
        workload_diversity_ok = ($workloads.Count -ge [Math]::Min(3, $TokenSpace * $PositionSpace))
        workload_coverage_pct = [math]::Round((100.0 * $workloads.Count) / ($TokenSpace * $PositionSpace), 4)
    }
    $runBundleManifestPath = Join-Path $runDir "compare_summary.json"
    $latestSummaryPath = Join-Path $root "sim\compare_summary_latest.json"
    $compareSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $runBundleManifestPath -Encoding UTF8
    $compareSummary | ConvertTo-Json -Depth 8 | Set-Content -Path $latestSummaryPath -Encoding UTF8
} else {
    if ($Mode -in @("base", "compare", "all")) {
        $baseLog = Invoke-SimBench `
            -Name "demo_full_model_base" `
            -Sources $baseSources `
            -Testbench "tb/integration/full_model_inference_tb.v" `
            -PlusArgs @("+TOKEN_ID=$TokenId", "+POSITION=$Position")
    }

    if ($Mode -in @("imprint", "compare", "all")) {
        $imprintLog = Invoke-SimBench `
            -Name "demo_full_model_imprint" `
            -Sources $imprintSources `
            -Testbench "tb/integration/full_model_inference_imprint_tb.v" `
            -PlusArgs @("+TOKEN_ID=$TokenId", "+POSITION=$Position")
    }

    if ($baseLog) {
        $baseText = Get-Content $baseLog -Raw
        $rows += [pscustomobject]@{
            Variant = "Base full-model"
            CyclesPerToken = Get-RegexInt -Text $baseText -Pattern "TOTAL INFERENCE\s+(\d+)\s+cy"
            TokensPerSecond = Get-RegexInt -Text $baseText -Pattern "Tokens/second:\s+~(\d+)"
            MedusaEffectiveTokS = Get-RegexInt -Text $baseText -Pattern "effective\s+(\d+)\s+tok/s"
        }
    }

    if ($imprintLog) {
        $imprintText = Get-Content $imprintLog -Raw
        $rows += [pscustomobject]@{
            Variant = "MINI imprint full-model"
            CyclesPerToken = Get-RegexInt -Text $imprintText -Pattern "TOTAL INFERENCE\s+(\d+)\s+cy"
            TokensPerSecond = Get-RegexInt -Text $imprintText -Pattern "Tokens/second:\s+~(\d+)"
            MedusaEffectiveTokS = Get-RegexInt -Text $imprintText -Pattern "effective\s+(\d+)\s+tok/s"
        }
    }
}

if ($rows.Count -gt 0) {
    Write-Host ""
    Write-Host "Measured throughput summary (@100MHz):" -ForegroundColor Yellow
    $rows | Format-Table -AutoSize
}

# Keep canonical measured logs updated for downstream proof-pack tooling.
if ($topLog) {
    $topCanonical = Join-Path $root "sim\measured_gpu_system_top_v2_bench.log"
    Get-Content -Path $topLog -Raw | Set-Content -Path $topCanonical -Encoding UTF8
}
if ($baseLog) {
    $baseCanonical = Join-Path $root "sim\measured_full_model_base.log"
    Get-Content -Path $baseLog -Raw | Set-Content -Path $baseCanonical -Encoding UTF8
}
if ($imprintLog) {
    $imprintCanonical = Join-Path $root "sim\measured_full_model_imprint.log"
    Get-Content -Path $imprintLog -Raw | Set-Content -Path $imprintCanonical -Encoding UTF8
}

if ($runBundleManifestPath -and (Test-Path $runBundleManifestPath)) {
    $iverilogVersion = "unknown"
    $vvpVersion = "unknown"
    $nativePrefVar = Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue
    $nativePrefOld = $null
    if ($nativePrefVar) {
        $nativePrefOld = $nativePrefVar.Value
        $global:PSNativeCommandUseErrorActionPreference = $false
    }
    try {
        $iverilogVersion = (& $iverilog -V 2>&1 | Select-Object -First 1)
        $vvpVersion = (& $vvp -V 2>&1 | Select-Object -First 1)
    } catch {
        Write-Warning ("Tool version probe failed: {0}" -f $_.Exception.Message)
    } finally {
        if ($nativePrefVar) {
            $global:PSNativeCommandUseErrorActionPreference = $nativePrefOld
        }
    }
    if (-not $iverilogVersion) { $iverilogVersion = "unknown" }
    if (-not $vvpVersion) { $vvpVersion = "unknown" }
    $processorName = $null
    try {
        $processorName = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    } catch {
        $processorName = $env:PROCESSOR_IDENTIFIER
    }
    $compareSummary["system_environment"] = [ordered]@{
        powershell_version = $PSVersionTable.PSVersion.ToString()
        computer_name = $env:COMPUTERNAME
        processor = $processorName
        iverilog_path = $iverilog
        iverilog_version = $iverilogVersion
        vvp_path = $vvp
        vvp_version = $vvpVersion
    }
    $compareSummary["file_integrity"] = [ordered]@{
        top_log_sha256 = if ($topLog) { Get-FileSHA256 -Path (Join-Path $root "sim\measured_gpu_system_top_v2_bench.log") } else { $null }
        base_log_sha256 = if ($baseLog) { Get-FileSHA256 -Path (Join-Path $root "sim\measured_full_model_base.log") } else { $null }
        imprint_log_sha256 = if ($imprintLog) { Get-FileSHA256 -Path (Join-Path $root "sim\measured_full_model_imprint.log") } else { $null }
    }
    $compareSummary | ConvertTo-Json -Depth 12 | Set-Content -Path $runBundleManifestPath -Encoding UTF8
    $compareSummary | ConvertTo-Json -Depth 12 | Set-Content -Path (Join-Path $root "sim\compare_summary_latest.json") -Encoding UTF8
}

if ($topLog) {
    $topText = Get-Content $topLog -Raw
    $baseCmd = Get-RegexInt -Text $topText -Pattern "MATMUL completed via optimized path in (\d+) cycles"
    $miniCmd = Get-RegexInt -Text $topText -Pattern "MINI imprint profile engaged in (\d+) cycles"
    $speedup = Get-RegexDouble -Text $topText -Pattern "speedup=([0-9]+\.[0-9]+)x"

    Write-Host ""
    Write-Host "Top-level command latency summary:" -ForegroundColor Yellow
    Write-Host ("  Baseline MATMUL : {0} cycles" -f $baseCmd)
    Write-Host ("  MINI imprint    : {0} cycles" -f $miniCmd)
    Write-Host ("  Speedup         : {0}x" -f $speedup)
}

if ($pairedMeasuredSamples.Count -gt 0) {
    $speedStats = Get-NumericStats ($pairedMeasuredSamples | ForEach-Object { [double]$_.speedup_x })
    Write-Host ""
    Write-Host ("Measured full-model speedup (paired): mean={0}x min={1}x max={2}x (n={3})" -f $speedStats.Mean, $speedStats.Min, $speedStats.Max, $speedStats.Count) -ForegroundColor Green
} elseif ($baseLog -and $imprintLog -and $rows.Count -ge 2) {
    $baseCycles = [double]$rows[0].CyclesPerToken
    $imprintCycles = [double]$rows[1].CyclesPerToken
    if ($baseCycles -gt 0 -and $imprintCycles -gt 0) {
        $fullModelSpeedup = [math]::Round($baseCycles / $imprintCycles, 4)
        Write-Host ""
        Write-Host ("Measured full-model speedup (base/imprint): {0}x" -f $fullModelSpeedup) -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Logs written to:"
if ($topLog) { Write-Host "  $topLog" }
if ($baseLog) { Write-Host "  $baseLog" }
if ($imprintLog) { Write-Host "  $imprintLog" }
if ($runBundleManifestPath) {
    Write-Host "  $runBundleManifestPath"
    Write-Host ("  {0}" -f (Join-Path $root "sim\compare_summary_latest.json"))
}

