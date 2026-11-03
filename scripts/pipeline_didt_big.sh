#!/usr/bin/env bash
# Unattended di/dt (power-swing) study for matmul4 and both systolic variants.
# Waits for the running systolic pipeline (by its pid file) so they don't compete for CPU.
#   wsl -d Ubuntu-24.04 -- bash scripts/pipeline_didt_big.sh     # from a terminal that stays open
# Proxy level per design: gate (synth) when its measured throughput is <= MAX_GATE_MS per
# evaluation, else RTL; validation is per-cycle post-route either way. One SUMMARY.txt.
# DESIGNS=matmul4:matmul4_p3,... picks a subset (block:top pairs); WAIT_FOR=none starts at once.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=src
PY=${PY:-$HOME/src/xls-power-rl/.venv/bin/python}
WAIT_FOR=${WAIT_FOR:-$(ls -d "$HOME"/runs/pipeline/*/ | tail -1)pid}
P=$HOME/runs/pipeline/didt-$(date +%Y%m%d-%H%M%S); mkdir -p "$P"
S=$P/SUMMARY.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$S"; }
SEEDS="--seeds 0 1 2 3 4 --budget 20000"
MAX_GATE_MS=${MAX_GATE_MS:-40}

echo $$ > "$P/pid"
say "di/dt pipeline start (log dir $P, pid $$), waiting for $WAIT_FOR"
W=$(cat "$WAIT_FOR" 2>/dev/null)
while [ -n "$W" ] && kill -0 "$W" 2>/dev/null; do sleep 120; done
say "previous pipeline finished; starting"

run_didt() {  # $1 block, $2 top
  local block=$1 top=$2 lv cfg ms=skipped
  # PROXY_LEVEL=rtl skips the gate-proxy build: Verilator needs hours and ~9 GB for 84k cells.
  if [ "${PROXY_LEVEL:-}" != rtl ]; then
    $PY scripts/proxy_calibrate.py --block "$block" --top "$top" --level synth > "$P/calib_${top}_synth.log" 2>&1
    ms=$(grep -oP 'throughput: \K[0-9.]+' "$P/calib_${top}_synth.log" || echo 1e9)
  fi
  if [ "$ms" != skipped ] && python3 -I -c "import sys; sys.exit(0 if float('$ms') <= $MAX_GATE_MS else 1)"; then
    lv=synth; cfg=${top}_gate_didt8
  else
    lv=rtl; cfg=${top}_didt8
  fi
  say "$top: gate proxy ${ms} ms/eval -> searching with the $lv proxy ($cfg)"
  $PY scripts/search.py --block "$block" --top "$top" --methods random_legal ga cem rl $SEEDS \
      --proxy-level $lv --objective swing --swing-period 8 > "$P/search_$top.log" 2>&1
  say "$top search done"
  $PY scripts/validate_didt.py --block "$block" --top "$top" --config "$cfg" > "$P/validate_$top.log" 2>&1
  { echo "== di/dt $cfg"; grep -E '^(random_legal|ga|cem|rl|hand) |Spearman|REJECTED|target period' "$P/validate_$top.log"; } >> "$S"
  # Illustrative AVP/loadline view; EDC is set by the average-power search winner.
  $PY scripts/avp_loadline.py --top "$top" --didt-config "$cfg" --avg-config "${top}_rerank20" > "$P/avp_$top.log" 2>&1 \
    || $PY scripts/avp_loadline.py --top "$top" --didt-config "$cfg" > "$P/avp_$top.log" 2>&1
  { echo "== AVP $top"; cat "$P/avp_$top.log"; } >> "$S"
}
designs=${DESIGNS:-matmul4:matmul4_p3,systolic4:systolic4_p2,systolic4_iso:systolic4iso_p2}
for d in ${designs//,/ }; do
  run_didt "${d%%:*}" "${d#*:}" &
done
wait
say "di/dt pipeline done"
