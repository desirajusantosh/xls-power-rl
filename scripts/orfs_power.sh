#!/usr/bin/env bash
# Vectorless report_power on a finished ORFS run (see orfs_run.sh).
#   scripts/orfs_power.sh <top_module> <synth|route> ["0.0 0.1 0.2"]
# For VCD-based power, use xpr.gatesim (it sets XPR_VCDS and calls orfs_tcl.sh).
set -euo pipefail
TOP=$1
export XPR_POWER_STAGE=$2 XPR_POWER_ACTIVITIES=${3:-0.1}
bash "$(dirname "$0")/orfs_tcl.sh" "$TOP" flow/power.tcl "power_$2"
grep -E '^XPR_(POWER|WARN)' "${ORFS_RUNS:-$HOME/runs/orfs}/$TOP/power_$2.log"
