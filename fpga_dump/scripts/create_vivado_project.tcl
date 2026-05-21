# create_vivado_project.tcl
# Create Vivado project for BitbyBit gpu_system_top_v2 on Kintex-7 Genesys 2
#
# Usage (from fpga_dump/):
#   vivado -mode batch -source scripts/create_vivado_project.tcl

set script_dir [file normalize [file dirname [info script]]]
set dump_root  [file normalize [file join $script_dir ..]]
set proj_dir   [file join $dump_root vivado_proj]
set proj_name  bitbybit_gpu_genesys2

file mkdir $proj_dir

create_project -force $proj_name $proj_dir -part XC7K325TFFG900-2

set_property target_language Verilog [current_project]
set_property default_lib work [current_project]

# RTL from filelist.f
set flist [file join $dump_root filelist.f]
set fh [open $flist r]
while {[gets $fh line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string index $line 0] eq "#"} { continue }
    set vpath [file join $dump_root $line]
    if {![file exists $vpath]} {
        puts "ERROR: missing RTL file $vpath"
        close $fh
        exit 1
    }
    add_files -norecurse $vpath
}
close $fh

update_compile_order -fileset sources_1
set_property top gpu_system_top_v2 [current_fileset]

# Constraints
add_files -fileset constrs_1 -norecurse [file join $dump_root constraints kintex7_genesys2.xdc]
set_property used_in_synthesis true [get_files kintex7_genesys2.xdc]
set_property used_in_implementation true [get_files kintex7_genesys2.xdc]

# Synthesis strategy (moderate runtime)
set_property strategy Vivado_Synthesis_Defaults [get_runs synth_1]

puts "Project created: $proj_dir/$proj_name.xpr"
puts "Top module: gpu_system_top_v2"
puts "Part: XC7K325TFFG900-2"
