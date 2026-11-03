# Export the synthesized netlist exactly as 1_synth.odb holds it, for gate-level sim.
source $::env(SCRIPTS_DIR)/load.tcl
load_design 1_synth.odb 1_synth.sdc
write_verilog /work/sim/1_synth_sim.v
