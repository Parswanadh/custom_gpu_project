# build_bitstream.tcl
# Run synthesis, implementation, and bitstream generation.
#
# Prerequisite: scripts/create_vivado_project.tcl
#
# Usage (from fpga_dump/):
#   vivado -mode batch -source scripts/build_bitstream.tcl

set script_dir [file normalize [file dirname [info script]]]
set dump_root  [file normalize [file join $script_dir ..]]
set proj_dir   [file join $dump_root vivado_proj]
set proj_name  bitbybit_gpu_genesys2
set xpr        [file join $proj_dir ${proj_name}.xpr]

if {![file exists $xpr]} {
    puts "Project not found: $xpr"
    puts "Run create_vivado_project.tcl first."
    exit 1
}

open_project $xpr

set_property top gpu_system_top_v2 [current_fileset]
update_compile_order -fileset sources_1

reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed."
    exit 1
}

reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation / bitstream failed."
    exit 1
}

set bit_dir [file join $proj_dir bitbybit_gpu_genesys2.runs impl_1]
set bit_glob [glob -nocomplain -directory $bit_dir *.bit]
if {[llength $bit_glob] > 0} {
    puts "Bitstream: [lindex $bit_glob 0]"
} else {
    puts "WARNING: No .bit file found under $bit_dir"
}

close_project
puts "Build complete."
