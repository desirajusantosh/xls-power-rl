# xls-power-rl

Tooling for power models and worst-case (power-virus) activity search on XLS-generated ML-accelerator blocks, from RTL to post-route on ASAP7, using only open-source tools.

**Status:** research code. Results and the write-up are not published yet.

## What's here

- **Flow:** DSLX → pipelined Verilog (XLS) → place and route (OpenROAD-flow-scripts, ASAP7) → gate-level VCD → power with OpenSTA.
- **Power model:** fits `P(α) = P_leak + P_intercept + slope · α` at synthesis and post-route from measured runs.
- **Fast proxy:** a Verilator toggle-rate model, calibrated against Icarus, at RTL or gate level.
- **Search:** random, GA, CEM and RL (GRU + REINFORCE) over input patterns, with a fail-closed scorer. Validity gates reject reset driving, stalled handshakes and reference mismatches, and every rejected candidate is logged.
- **Objectives:** average power, and periodic power swing (di/dt).
- **Validation:** re-simulation of each search's picks at gate level, with post-route `report_power`.

## Quick start

Requires Linux or WSL2 (Ubuntu 24.04), Python ≥ 3.11 and Docker. Full setup: [docs/setup.md](docs/setup.md).

```bash
# Python harness and unit tests (no EDA tools needed)
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,rl,plots]"
.venv/bin/pytest -q

# One block end to end: DSLX -> Verilog -> ORFS (ASAP7) -> VCD power model
python scripts/build_block.py mac 2
CLK_PERIOD_PS=500 scripts/orfs_run.sh ~/runs/mac/p2/mac_p2.v mac_p2
python scripts/vcd_power.py --block mac --top mac_p2

# Calibrate the proxy, search, then validate post-route
python scripts/proxy_calibrate.py --block mac --top mac_p2 --level synth
python scripts/search.py --block mac --top mac_p2 --proxy-level synth --seeds 0 1 2 3 4
python scripts/validate_search.py --block mac --top mac_p2 --config mac_p2_gate
```

Run scripts with `PYTHONPATH=src`, or after `pip install -e .`.

## Layout

```
designs/   DSLX sources: INT8 MAC, bf16 multiplier, INT8 4x4 matvec, systolic-array generator
src/xpr/   Python harness: blocks, gates, stimulus, gate-level sim, VCD, power fit, proxy, search
flow/      ORFS config (asap7/), power and netlist Tcl, testbench, Verilator proxy driver
scripts/   build, ORFS, power fit, calibration, search, validation and pipeline scripts
tests/     pytest suite (runs on any OS; no EDA tools needed)
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
