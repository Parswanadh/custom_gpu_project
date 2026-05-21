################################################################################
# Kintex-7 Genesys 2 constraints for gpu_system_top_v2
# Part: XC7K325TFFG900-2
# Board clock: 100 MHz differential (sysclk_p/n)
#
# Top ports: clk, rst_async_n (active-low async reset)
# Map board pins via a thin wrapper or uncomment LOC lines below.
################################################################################

## ---- Clock (100 MHz) ----
create_clock -period 10.000 -name sys_clk [get_ports clk]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk]

# Genesys 2 differential system clock (requires IBUFDS in board wrapper)
# set_property -dict { PACKAGE_PIN AD12 IOSTANDARD LVDS } [get_ports sysclk_p]
# set_property -dict { PACKAGE_PIN AD11 IOSTANDARD LVDS } [get_ports sysclk_n]
# create_clock -period 10.000 -name sys_clk_board [get_ports sysclk_p]

## ---- Async reset (active-low) ----
set_property -dict { PACKAGE_PIN R19 IOSTANDARD LVCMOS33 } [get_ports rst_async_n]
set_false_path -from [get_ports rst_async_n]
set_false_path -to [get_ports rst_async_n]

## ---- Interrupt (optional status LED hookup) ----
# set_property -dict { PACKAGE_PIN T28 IOSTANDARD LVCMOS33 } [get_ports irq_out]

################################################################################
# Placeholder AXI4-Lite slave (host config) — timing relaxed until PS/DMA hookup
################################################################################
set_input_delay  -clock sys_clk -max 4.0 [get_ports {s_axi_* cmd_*}]
set_input_delay  -clock sys_clk -min 0.5 [get_ports {s_axi_* cmd_*}]
set_output_delay -clock sys_clk -max 4.0 [get_ports {s_axi_* cmd_*}]
set_output_delay -clock sys_clk -min 0.5 [get_ports {s_axi_* cmd_*}]

# Tie off unused AXI response inputs in simulation; on FPGA connect to PS or tie:
# set_property KEEP_HIERARCHY TRUE [get_cells u_gpu_top]

################################################################################
# Placeholder AXI4 master (DMA) — connect to DDR via AXI interconnect in BD
################################################################################
set_output_delay -clock sys_clk -max 4.0 [get_ports {m_axi_ar* m_axi_aw* m_axi_w*}]
set_output_delay -clock sys_clk -min 0.5 [get_ports {m_axi_ar* m_axi_aw* m_axi_w*}]
set_input_delay  -clock sys_clk -max 4.0 [get_ports {m_axi_r* m_axi_bvalid}]
set_input_delay  -clock sys_clk -min 0.5 [get_ports {m_axi_r* m_axi_bvalid}]

# FMC / expansion connector placeholders (uncomment and assign when routing AXI to FMC)
# set_property PACKAGE_PIN <pin> [get_ports {m_axi_araddr[0]}]
# set_property PACKAGE_PIN <pin> [get_ports {s_axi_awaddr[0]}]

################################################################################
# Performance / debug outputs (optional)
################################################################################
# set_output_delay -clock sys_clk -max 4.0 [get_ports {cycle_count[*] zero_skip_total[*] mac_total[*]}]

################################################################################
# Bitstream generation
################################################################################
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
