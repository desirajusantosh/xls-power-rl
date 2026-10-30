#!/usr/bin/env bash
# Status of the latest pipeline run, by its recorded PID (not by name matching).
#   wsl -d Ubuntu-24.04 -- bash scripts/pipeline_status.sh
D=$(ls -d "$HOME"/runs/pipeline/*/ 2>/dev/null | tail -1)
[ -z "$D" ] && { echo "no pipeline runs"; exit 0; }
PID=$(cat "$D/pid" 2>/dev/null)
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then echo "RUNNING (pid $PID) $D"; else echo "NOT RUNNING $D"; fi
tail -n "${1:-15}" "$D/SUMMARY.txt"
