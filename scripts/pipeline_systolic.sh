#!/usr/bin/env bash
# Unattended pipeline: finish the di/dt study, then run the systolic array end to end.
# Writes everything under ~/runs/pipeline/<timestamp>/ and one compact SUMMARY.txt.
#   wsl -d Ubuntu-24.04 -- bash scripts/pipeline_systolic.sh        # from a terminal that stays open
# Keep a wsl.exe session attached for the whole run: WSL shuts its VM down once no
# Windows-side session is attached, killing detached (setsid/nohup) processes with it.
# SKIP_DIDT=1 skips the di/dt stage. CLKS=1200,1400 overrides the clock retry list (ps).
# The pipeline's PID is written to <logdir>/pid.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH=src
PY=${PY:-$HOME/src/xls-power-rl/.venv/bin/python}
P=$HOME/runs/pipeline/$(date +%Y%m%d-%H%M%S); mkdir -p "$P"
S=$P/SUMMARY.txt
say() { echo "[$(date +%H:%M)] $*" | tee -a "$S"; }
SEEDS="--seeds 0 1 2 3 4 --budget 20000"

echo $$ > "$P/pid"
say "pipeline start (log dir $P, pid $$)"

# --- 1. di/dt: wait for the running swing searches, then validate per-cycle post-route.
if [ -z "${SKIP_DIDT:-}" ]; then
  while pgrep -f "scripts/search.py .*--objective swing" > /dev/null; do sleep 60; done
  say "di/dt searches finished"
  for spec in "mac mac_p2 mac_p2_gate_didt8" "bf16_mul bf16mul_p4 bf16mul_p4_gate_didt8" \
              "bf16_mul bf16mul_p4 bf16mul_p4_finite_gate_didt8"; do
    set -- $spec
    $PY scripts/validate_didt.py --block $1 --top $2 --config $3 > "$P/didt_$3.log" 2>&1
    { echo "== di/dt $3"; grep -E '^(random|random_legal|ga|cem|rl|llm|hand) |Spearman|REJECTED' "$P/didt_$3.log"; } >> "$S"
  done
  say "di/dt validation done"
fi

# --- 2. Systolic array: ORFS (retry at slower clocks if timing fails), power, proxies, search.
run_block() {  # $1 block, $2 top
  local block=$1 top=$2 v=$HOME/runs/$1/p2/$2.v
  # Seconds of RTL simulation before hours of ORFS: the suite must pass every gate.
  if ! $PY scripts/rtl_check.py --block "$block" --rtl "$v" > "$P/rtl_check_$top.log" 2>&1; then
    say "$top rtl_check FAILED: $(tail -n 1 "$P/rtl_check_$top.log"); skipping this block"
    return 1
  fi
  say "$top rtl_check passed"
  local clks=${CLKS:-500,600,700}
  for clk in ${clks//,/ }; do
    CLK_PERIOD_PS=$clk bash scripts/orfs_run.sh "$v" "$top" > "$P/orfs_${top}_$clk.log" 2>&1
    local ws
    ws=$(python3 -I -c "import json;print(json.load(open('$HOME/runs/orfs/$top/logs/asap7/$top/base/6_report.json'))['finish__timing__setup__ws'])" 2>/dev/null || echo nan)
    say "$top ORFS at $clk ps: setup WNS $ws ps"
    python3 -I -c "import sys; sys.exit(0 if float('$ws') >= 0 else 1)" 2>/dev/null && break
  done
  python3 -I -c "
import json; j=json.load(open('$HOME/runs/orfs/$top/logs/asap7/$top/base/6_report.json'))
print('$top post-route: area %.1f um2, cells %d, power(ORFS default activity) %.2f mW' % (j['finish__design__instance__area'], j['finish__design__instance__count'], j['finish__power__total']*1e3))" >> "$S"
  # Stale artifacts from any earlier attempt at another clock must not be reused.
  rm -rf "$HOME/runs/orfs/$top/sim" "$HOME/runs/orfs/$top/proxy_build_rtl" "$HOME/runs/orfs/$top/proxy_build_synth"
  bash scripts/flatten_rtl.sh "$top" >> "$S" 2> "$P/flatten_$top.log" || say "$top flatten FAILED (see flatten_$top.log)"
  $PY scripts/vcd_power.py --block "$block" --top "$top" > "$P/power_$top.log" 2>&1
  grep -E '^(synth|route) +P_leak' "$P/power_$top.log" | sed "s/^/$top /" >> "$S"
  for lv in rtl synth; do
    $PY scripts/proxy_calibrate.py --block "$block" --top "$top" --level $lv > "$P/calib_${top}_$lv.log" 2>&1
    grep -E 'bits counted|max \|diff\||throughput' "$P/calib_${top}_$lv.log" | sed "s/^/$top $lv: /" >> "$S"
  done
  # No --rerank-k: it builds the gate proxy, which does not compile at this size (decisions.md, 2026-11-01).
  $PY scripts/search.py --block "$block" --top "$top" --methods random_legal ga cem rl $SEEDS \
      > "$P/search_$top.log" 2>&1
  say "$top search done"
  $PY scripts/transfer_patterns.py --src matmul4_p3_rerank20 --method llm --dst "$top" \
      --as-method llm_transfer >> "$P/search_$top.log" 2>&1
  $PY scripts/validate_search.py --block "$block" --top "$top" --config "$top" > "$P/validate_$top.log" 2>&1
  { echo "== validate $top"; grep -E '^(random_legal|ga|cem|rl|llm_transfer|suite best) |fidelity|pick|REJECTED' "$P/validate_$top.log"; } >> "$S"
}
run_block systolic4 systolic4_p2 &
run_block systolic4_iso systolic4iso_p2 &
wait
say "pipeline done"
