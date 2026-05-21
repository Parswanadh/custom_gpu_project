# Sync synthesis-ready RTL into fpga_dump/rtl/ for Kintex-7 (gpu_system_top_v2)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Split-Path -Parent $root
$dump = Join-Path $proj "fpga_dump\rtl"

$files = @(
    "rtl/top/reset_synchronizer.v",
    "rtl/top/gpu_config_regs.v",
    "rtl/top/command_processor.v",
    "rtl/top/perf_counters.v",
    "rtl/memory/banked_scratchpad.v",
    "rtl/memory/dma_engine.v",
    "rtl/memory/imprinted_embedding_rom.v",
    "rtl/integration/imprinted_mini_transformer_core.v",
    "rtl/transformer/rope_encoder.v",
    "rtl/memory/dual_port_lut.v",
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
    "rtl/integration/skid_buffer.v",
    "rtl/compute/rmsnorm_vp.v",
    "rtl/compute/inv_sqrt_lut_256.v",
    "rtl/transformer/rope_unit_v2.v",
    "rtl/transformer/gated_mlp_da.v",
    "rtl/top/gpu_system_top_v2.v"
)

if (-not (Test-Path $dump)) { New-Item -ItemType Directory -Path $dump -Force | Out-Null }

foreach ($rel in $files) {
    $src = Join-Path $proj $rel
    $dst = Join-Path $dump (Split-Path -Leaf $rel)
    if (-not (Test-Path $src)) { throw "Missing RTL: $src" }
    Copy-Item -Force $src $dst
    Write-Host "Copied $(Split-Path -Leaf $rel)"
}

Write-Host ""
Write-Host "FPGA dump synced: $dump ($($files.Count) files)"
