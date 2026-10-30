"""RTL and gate-level simulation, and VCD-annotated power, for one block.

For each stimulus and each level (rtl, synth, route):
  1. simulate that level's own design with Icarus (zero delay),
  2. check the trace against the validity gates using the DUT's sampled ready,
  3. compare outputs against the block's reference model,
  4. measure alpha (toggles per non-clock net bit per cycle) from the window VCD.
Then report_power runs once per physical level (synth, route) over all accepted
VCDs. Rejected traces never reach report_power. RTL alpha is the fast proxy.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

from . import stimulus as st
from .gates import check_trace
from .orfs import parse_power
from .vcd import read_toggles

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("ASAP7_MODELS", Path.home() / "runs" / "asap7_models"))
CELL_MODELS = ["dff.v", "empty.v"] + [f"asap7sc7p5t_{g}_RVT_TT_201020.v" for g in ("AO", "OA", "SIMPLE", "INVBUF")]
LEVELS = ("rtl", "synth", "route")
POWER_LEVELS = ("synth", "route")
FILL_CYCLES = 16  # >= pipeline latency incl. input/output flops


def is_clock_net(name: str) -> bool:
    return "clk" in name.lower()


@dataclass
class SimResult:
    stim: str
    regime: str
    level: str
    accepted: bool
    failed: list[str]
    alpha: float
    n_bits: int
    window_cycles: int
    vcd: str
    mismatches: list[tuple[int, int, int]] = field(default_factory=list)  # (index, got, want)


def clock_period_ps(run_dir: Path) -> int:
    return int((run_dir / "clk_period_ps.txt").read_text())


def prepare(run_dir: Path, top: str) -> None:
    """Export the netlists each level simulates: synth from 1_synth.odb, route from 6_final.v."""
    sim = run_dir / "sim"
    sim.mkdir(exist_ok=True)
    if not (sim / "1_synth_sim.v").exists():
        subprocess.run(["bash", str(REPO / "scripts/orfs_tcl.sh"), top, "flow/netlist.tcl", "netlist_synth"], check=True)
    final = sim / "6_final_sim.v"
    if not final.exists():
        final.write_text((run_dir / "results/asap7" / top / "base/6_final.v").read_text())


def rtl_source(run_dir: Path, top: str) -> Path:
    """The RTL to simulate: the flattened copy for hierarchical designs, if one was made."""
    flat = run_dir / f"{top}_flat.v"
    return flat if flat.exists() else run_dir / f"{top}.v"


def level_source(run_dir: Path, top: str, level: str) -> list[Path]:
    if level == "rtl":
        return [rtl_source(run_dir, top)]
    netlist = run_dir / "sim" / {"synth": "1_synth_sim.v", "route": "6_final_sim.v"}[level]
    return [netlist, *(MODELS / m for m in CELL_MODELS)]


def compile_level(block: ModuleType, run_dir: Path, top: str, level: str) -> Path:
    vvp = run_dir / "sim" / f"{level}.vvp"
    half = clock_period_ps(run_dir) // 2
    subprocess.run(
        ["iverilog", "-g2012", f"-DDUT={top}", f"-DIN_W={block.IN_W}", f"-DOUT_W={block.OUT_W}",
         f"-DHALF_PS={half}", "-o", str(vvp), str(REPO / "flow/sim/tb_stream.v"),
         *map(str, level_source(run_dir, top, level))],
        check=True, capture_output=True, text=True,
    )
    return vvp


def simulate(block: ModuleType, vvp: Path, stim: st.Stimulus, out: Path, level: str) -> SimResult:
    out.mkdir(parents=True, exist_ok=True)
    cycles = st.write_hex(stim, block.IN_W, out / "stim.hex")
    start, end = st.RESET_CYCLES + FILL_CYCLES, cycles
    vcd, log = out / f"{level}.vcd", out / f"{level}.log"
    args = ["vvp", "-n", str(vvp), f"+stim={out / 'stim.hex'}", f"+cycles={cycles}",
            f"+win_start={start}", f"+win_end={end}", f"+vcd={vcd}", f"+log={log}"]
    if stim.regime == "leak":
        args.append("+clk_stop")
    subprocess.run(args, check=True, capture_output=True, text=True)

    rows = [line.split() for line in log.read_text().splitlines()]
    ready = np.array([r[1] == "1" for r in rows])
    got = [int(r[3], 16) for r in rows if r[2] == "1"]

    regime = "idle" if stim.regime in ("idle", "leak") else stim.regime
    failed = check_trace(st.to_trace(stim, block.IN_W, ready, (start, end)), latency=FILL_CYCLES, regime=regime).failed

    want = block.reference(stim)
    mismatches = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if not block.same(g, w)]
    if stim.regime != "leak" and (mismatches or len(got) < len(want) - FILL_CYCLES):
        failed.append("outputs_match_reference")

    toggles = read_toggles(vcd)
    window = end - start
    alpha = toggles.alpha(window, exclude=is_clock_net)
    n_bits = sum(1 for n in toggles.counts if not is_clock_net(n))
    if stim.regime not in ("idle", "leak") and alpha == 0.0:
        failed.append("nonzero_net_activity")
    return SimResult(stim.name, stim.regime, level, not failed, failed, alpha, n_bits, window, str(vcd), mismatches[:20])


def power(run_dir: Path, top: str, level: str, results: list[SimResult]) -> tuple[dict[str, dict], str]:
    """report_power over the accepted VCDs of one level, in one OpenROAD session."""
    accepted = [r for r in results if r.accepted and r.level == level]
    # Leak runs go last: power.tcl deletes the clocks after reading them.
    accepted.sort(key=lambda r: r.regime == "leak")
    container_path = lambda p: "/work/" + str(Path(p).relative_to(run_dir))
    env = dict(os.environ,
               XPR_POWER_STAGE=level,
               XPR_VCDS=" ".join(f"{r.stim}={container_path(r.vcd)}" for r in accepted),
               XPR_NOCLOCK_VCDS=" ".join(r.stim for r in accepted if r.regime == "leak"))
    stem = f"power_vcd_{level}_{os.getpid()}"
    subprocess.run(["bash", str(REPO / "scripts/orfs_tcl.sh"), top, "flow/power.tcl", stem], check=True, env=env)
    text = (run_dir / f"{stem}.log").read_text()
    return {p.label: asdict(p) for p in parse_power(text)}, text


def phase_power(run_dir: Path, top: str, level: str, vcds: dict[str, str], first_cycle: int,
                cycles: int) -> dict[str, list[float]]:
    """Per-cycle total power over `cycles` consecutive cycles from `first_cycle`, per VCD.

    Each cycle is its own report_power over a one-cycle VCD window that includes the
    input change at the cycle's start and its rising edge. Returns {name: [W, ...]}.
    """
    period = clock_period_ps(run_dir)
    container_path = lambda p: "/work/" + str(Path(p).relative_to(run_dir))
    items = []
    for name, vcd in vcds.items():
        for k in range(cycles):
            t0 = (first_cycle + k) * period - 1
            items.append(f"{name}#{k}={container_path(vcd)}@{t0}:{t0 + period}")
    env = dict(os.environ, XPR_POWER_STAGE=level, XPR_VCDS=" ".join(items), XPR_NOCLOCK_VCDS="")
    stem = f"power_phase_{level}_{os.getpid()}"
    subprocess.run(["bash", str(REPO / "scripts/orfs_tcl.sh"), top, "flow/power.tcl", stem], check=True, env=env)
    out: dict[str, list[float]] = {n: [0.0] * cycles for n in vcds}
    for p in parse_power((run_dir / f"{stem}.log").read_text()):
        name, k = p.label.rsplit("#", 1)
        out[name][int(k)] = p.total_w
    return out


def run_suite(block: ModuleType, top: str, suite: list[st.Stimulus], run_dir: Path,
              out_name: str = "power_vcd.json") -> dict:
    prepare(run_dir, top)
    results: list[SimResult] = []
    for level in LEVELS:
        vvp = compile_level(block, run_dir, top, level)
        for s in suite:
            results.append(simulate(block, vvp, s, run_dir / "sim" / s.name, level))
    out = {"block": block.NAME, "top": top, "clk_period_ps": clock_period_ps(run_dir),
           "sims": [asdict(r) for r in results], "power": {}, "annotation": {}}
    for level in POWER_LEVELS:
        pts, text = power(run_dir, top, level, results)
        out["power"][level] = pts
        out["annotation"][level] = [l for l in text.splitlines() if l.startswith("XPR_ANNOTATION")]
    (run_dir / out_name).write_text(json.dumps(out, indent=2))
    return out
