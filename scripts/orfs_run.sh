#!/usr/bin/env bash
# Run one Verilog block through OpenROAD-flow-scripts (Docker) on ASAP7.
#   scripts/orfs_run.sh <verilog> <top_module> [make_target]
# Outputs: ~/runs/orfs/<top>/{results,logs,reports,objects}. Default target: finish.
set -euo pipefail
VERILOG=$(realpath "$1"); TOP=$2; TARGET=${3:-finish}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${ORFS_RUNS:-$HOME/runs/orfs}/$TOP
IMAGE=${ORFS_IMAGE:-openroad/orfs:latest}
mkdir -p "$OUT"
cp "$VERILOG" "$OUT/$TOP.v"
# The clock period is a property of the run: later power and sim steps read it back.
echo "${CLK_PERIOD_PS:-500}" > "$OUT/clk_period_ps.txt"

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$REPO":/xpr:ro -v "$OUT":/work \
  -e CLK_PERIOD_PS="${CLK_PERIOD_PS:-500}" \
  "$IMAGE" bash -c "
    source ./env.sh >/dev/null
    cd flow
    make DESIGN_CONFIG=/xpr/flow/asap7/config.mk \
         DESIGN_NAME=$TOP VERILOG_FILES=/work/$TOP.v \
         WORK_HOME=/work FLOW_VARIANT=base $TARGET
  " 2>&1 | tee "$OUT/orfs.log" | grep -E '^(\[INFO\]|Log|Elapsed|\[ERROR|Error|make)' || true

docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' > "$OUT/orfs_image.txt"
echo "done: $OUT"
