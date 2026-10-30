#!/usr/bin/env bash
# Flatten a hierarchical XLS Verilog module (proc networks: node + FIFO submodules) into one
# module, so RTL-level activity covers every net, not just the top level's.
#   scripts/flatten_rtl.sh <top_module>      # reads/writes ~/runs/orfs/<top>/{<top>.v,<top>_flat.v}
# Uses yosys from the ORFS image: proc + flatten + opt_clean only (no synthesis/mapping),
# so the result is still word-level RTL. xpr.gatesim / xpr.proxy use <top>_flat.v at the
# RTL level whenever it exists.
set -euo pipefail
TOP=$1
OUT=${ORFS_RUNS:-$HOME/runs/orfs}/$TOP
IMAGE=${ORFS_IMAGE:-openroad/orfs:latest}
docker run --rm --user "$(id -u):$(id -g)" -v "$OUT":/work "$IMAGE" bash -c "
  source ./env.sh >/dev/null
  yosys -q -p 'read_verilog /work/$TOP.v; hierarchy -top $TOP; proc; flatten; opt_clean; write_verilog -noattr /work/${TOP}_flat.v'
"
echo "wrote $OUT/${TOP}_flat.v ($(grep -c . "$OUT/${TOP}_flat.v") lines)"
