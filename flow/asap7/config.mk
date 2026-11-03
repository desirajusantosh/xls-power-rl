# ORFS config for one XLS-generated block on ASAP7.
# DESIGN_NAME and VERILOG_FILES are passed on the make command line by scripts/orfs_run.sh.
# Floorplan settings follow the stock asap7/gcd smoke test.

export PLATFORM          = asap7
export SDC_FILE          = /xpr/flow/asap7/constraint.sdc

export CORE_UTILIZATION  = 50
export CORE_ASPECT_RATIO = 1
export CORE_MARGIN       = 0.5
export PLACE_DENSITY     = 0.40
export GPL_PLACE_IOS     = 1
