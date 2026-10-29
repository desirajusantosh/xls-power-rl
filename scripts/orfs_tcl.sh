#!/usr/bin/env bash
# Run an OpenROAD Tcl script inside ORFS's environment on a finished run.
#   scripts/orfs_tcl.sh <top_module> <script.tcl (repo-relative)> <log_stem>
# Env vars named XPR_* are passed into the container. /work is the run dir.
set -euo pipefail
TOP=$1; TCL=$2; STEM=$3
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${ORFS_RUNS:-$HOME/runs/orfs}/$TOP
IMAGE=${ORFS_IMAGE:-openroad/orfs:latest}

PERIOD=$(cat "$OUT/clk_period_ps.txt" 2>/dev/null || echo 500)
envs=(-e CLK_PERIOD_PS="$PERIOD")
while IFS='=' read -r k _; do envs+=(-e "$k"); done < <(env | grep -E '^XPR_' || true)

docker run --rm --user "$(id -u):$(id -g)" \
  -v "$REPO":/xpr:ro -v "$OUT":/work "${envs[@]}" \
  "$IMAGE" bash -c "
    source ./env.sh >/dev/null
    cd flow
    make DESIGN_CONFIG=/xpr/flow/asap7/config.mk \
         DESIGN_NAME=$TOP VERILOG_FILES=/work/$TOP.v \
         WORK_HOME=/work FLOW_VARIANT=base \
         RUN_SCRIPT=/xpr/$TCL RUN_LOG_NAME_STEM=$STEM run
  " > "$OUT/$STEM.log" 2>&1 || { tail -30 "$OUT/$STEM.log"; exit 1; }
