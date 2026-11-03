# Shared ASAP7 constraints for XLS-generated blocks (ports: clk, rst, data/vld/rdy).
# Units: ps. CLK_PERIOD_PS comes from the environment; default 500 ps (2 GHz).

set clk_period [expr {[info exists ::env(CLK_PERIOD_PS)] ? $::env(CLK_PERIOD_PS) : 500}]
set clk_io_pct 0.2

create_clock -name core_clock -period $clk_period [get_ports clk]
create_clock -name vclk -period $clk_period

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay [expr {$clk_period * $clk_io_pct}] -clock vclk $non_clock_inputs
set_output_delay [expr {$clk_period * $clk_io_pct}] -clock vclk [all_outputs]
