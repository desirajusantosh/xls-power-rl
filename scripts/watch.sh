#!/usr/bin/env bash
# Live view of what the flow is doing: EDA processes, Docker, recent runs.
# Usage (from Windows): wsl -d Ubuntu-24.04 -- bash scripts/watch.sh
RUNS=${RUNS:-$HOME/runs}
while true; do
  clear
  echo "== $(date '+%H:%M:%S')  xls-power-rl activity  (Ctrl-C to stop)"
  echo
  echo "-- EDA / build processes"
  ps -eo pid,etime,pcpu,rss,args --sort=-pcpu \
    | grep -E 'codegen_main|opt_main|ir_converter|interpreter_main|verilator|iverilog|vvp|yosys|openroad|opensta|make|docker (pull|run)|pytest' \
    | grep -v grep | cut -c1-150 | head -12 || true
  echo
  echo "-- Docker images and containers"
  docker images --format '{{.Repository}}:{{.Tag}}  {{.Size}}' 2>/dev/null || sudo -n docker images 2>/dev/null || echo "(no docker access yet)"
  docker ps --format '{{.Names}}  {{.Image}}  {{.Status}}' 2>/dev/null
  echo
  echo "-- Most recent run outputs in $RUNS"
  find "$RUNS" -type f -printf '%TT  %8s  %P\n' 2>/dev/null | sort -r | head -12 | cut -c1-8,20-
  echo
  echo "-- Memory"
  free -h | head -2
  sleep 3
done
