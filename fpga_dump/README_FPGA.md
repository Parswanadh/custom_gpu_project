# FPGA Synthesis Dump — gpu_system_top_v2 (Kintex-7)

Portable RTL snapshot and Vivado flow for **Digilent Genesys 2** (`XC7K325TFFG900-2`).

## Top module

| Item | Value |
|------|--------|
| Top | `gpu_system_top_v2` |
| Clock | 100 MHz (`clk`, 10 ns period) |
| Reset | `rst_async_n` — active-low async board reset → `reset_synchronizer` |
| Host interface | AXI4-Lite slave (`s_axi_*`) + command FIFO (`cmd_valid`/`cmd_data`) |
| Memory | AXI4 master (`m_axi_*`) for DMA weight/activation transfers |

## Directory layout

```
fpga_dump/
├── rtl/                    # 27 flat Verilog sources (copied from ../rtl/)
├── constraints/
│   └── kintex7_genesys2.xdc
├── scripts/
│   ├── create_vivado_project.tcl
│   └── build_bitstream.tcl
├── filelist.f              # One RTL path per line (relative to fpga_dump/)
└── README_FPGA.md
```

## RTL dependency closure (27 files)

All modules required to elaborate `gpu_system_top_v2` are included:

`reset_synchronizer`, `gpu_config_regs`, `command_processor`, `perf_counters`, `banked_scratchpad`, `dma_engine`, `imprinted_embedding_rom`, `imprinted_mini_transformer_core`, `rope_encoder`, `dual_port_lut`, `grouped_query_attention`, `parallel_softmax`, `exp_lut_256`, `recip_lut_256`, `gelu_lut_256`, `gelu_activation`, `kv_cache_quantizer`, `activation_compressor`, `prefetch_engine`, `layer_pipeline_controller`, `optimized_transformer_layer`, `skid_buffer`, `rmsnorm_vp`, `inv_sqrt_lut_256`, `rope_unit_v2`, `gated_mlp_da`, `gpu_system_top_v2`

Sources were copied from `custom_gpu_project/rtl/{top,memory,integration,transformer,compute,control}/`.

## Bring-up steps

### 1. Prerequisites

- Xilinx Vivado (2020.2 or newer recommended)
- Genesys 2 board, USB-JTAG cable
- Optional: MicroBlaze/Zynq PS for AXI host (not included in this dump)

### 2. Create project

```powershell
cd custom_gpu_project\fpga_dump
vivado -mode batch -source scripts\create_vivado_project.tcl
```

This creates `vivado_proj/bitbybit_gpu_genesys2.xpr` with part `XC7K325TFFG900-2` and top `gpu_system_top_v2`.

### 3. Synthesize and build bitstream

```powershell
vivado -mode batch -source scripts\build_bitstream.tcl
```

Or open the `.xpr` in GUI and run **Generate Bitstream**.

### 4. Board constraints

`constraints/kintex7_genesys2.xdc` defines:

- 100 MHz `create_clock` on `clk`
- Async reset timing exception on `rst_async_n` (LOC: Genesys 2 `cpu_resetn`, pin R19)
- Placeholder input/output delays on AXI and command ports

**Clock note:** Genesys 2 provides a **differential** 100 MHz LVDS clock (`sysclk_p` / `sysclk_n`, pins AD12/AD11). `gpu_system_top_v2` expects a single-ended `clk`. Instantiate `IBUFDS` in a board wrapper (e.g. `genesys2_gpu_wrapper`) and connect `clk` to the buffer output before assigning `PACKAGE_PIN` to `sysclk_p/n`.

### 5. AXI hookup (placeholder)

This dump does **not** include a block design. For lab bring-up:

1. Tie or connect `s_axi_*` to a MicroBlaze AXI-Lite master or loopback testbench.
2. Route `m_axi_*` through an AXI interconnect to MIG/DDR3 (Genesys 2 memory controller IP).
3. Assign FMC/GPIO LOC constraints where AXI signals leave the FPGA (see commented placeholders in `.xdc`).

### 6. Program device

```tcl
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] 0]
set_property PROGRAM.FILE {vivado_proj/bitbybit_gpu_genesys2.runs/impl_1/gpu_system_top_v2.bit} [current_hw_device]
program_hw_devices [current_hw_device]
```

## Regenerating the dump

From `custom_gpu_project/fpga_dump`:

```powershell
$src = "..\rtl"
$dst = "rtl"
# Re-copy each file listed in filelist.f from its original subtree under $src
```

Or re-run the parent agent copy script against `filelist.f`.

## Known synthesis notes

- Large imprinted ROMs (`imprinted_embedding_rom`, transformer weights) may dominate BRAM utilization; review `report_utilization` after synth.
- Full GPU at 100 MHz may not close timing without pipeline tuning; use `report_timing_summary` and consider lowering `clk` period for first bring-up.
- Simulation file `../sim/sys_v2_test_output.log` may list missing modules if the full repo testbench omits files; this dump includes the closure for `gpu_system_top_v2` only.

## Part number

| Board | FPGA |
|-------|------|
| Digilent Genesys 2 | XC7K325TFFG900-2 |
