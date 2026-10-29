"""Power model from gate-level VCD at synth and post-route, for one block run.

  python scripts/vcd_power.py --block mac --top mac_p2 [--cycles 2000] [--reuse]

Needs a finished ORFS run in ~/runs/orfs/<top> (scripts/orfs_run.sh).
Writes <run>/power_vcd.json, <run>/fit.json and docs/img/<top>_power_fit.png.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xpr import gatesim
from xpr.blocks import BLOCKS
from xpr.powerfit import Run, fit

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--top", required=True)
    ap.add_argument("--cycles", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true", help="refit from an existing power_vcd.json")
    args = ap.parse_args()
    block = BLOCKS[args.block]
    run_dir = Path.home() / "runs" / "orfs" / args.top

    if args.reuse:
        data = json.loads((run_dir / "power_vcd.json").read_text())
    else:
        data = gatesim.run_suite(block, args.top, block.suite(args.cycles), run_dir)
    period_ps = data["clk_period_ps"]

    for s in data["sims"]:
        if not s["accepted"]:
            print(f"REJECTED {s['level']} {s['stim']}: {s['failed']} {s['mismatches'][:3]}")
    rtl_alpha = {s["stim"]: s["alpha"] for s in data["sims"] if s["level"] == "rtl"}

    fits, rows = {}, []
    for level, pts in data["power"].items():
        sims = {s["stim"]: s for s in data["sims"] if s["level"] == level and s["accepted"]}
        runs = []
        for name, p in pts.items():
            s = sims[name]
            # Fail closed: a clock-stopped run must report (almost) no dynamic power,
            # or its "leakage" silently absorbs the intercept.
            if s["regime"] == "leak" and p["internal_w"] + p["switching_w"] > 0.01 * p["total_w"]:
                raise SystemExit(f"{level}/{name}: clock-stopped run reports dynamic power "
                                 f"{(p['internal_w'] + p['switching_w']) * 1e6:.2f} uW; refusing to fit")
            runs.append(Run(p["total_w"], s["alpha"], clocked=s["regime"] != "leak", label=name))
            rows.append((level, name, s["regime"], s["alpha"], rtl_alpha.get(name, float("nan")),
                         p["total_w"], p["internal_w"], p["switching_w"], p["leakage_w"]))
        m = fit(runs)
        fits[level] = {k: v for k, v in asdict(m).items() if k != "residuals"}
        fits[level]["residuals"] = dict(zip([r.label for r in runs if r.clocked], m.residuals))
        fits[level]["clk_period_ps"] = period_ps

    print(f"\n{args.top} at {period_ps} ps ({1e6 / period_ps:.0f} MHz)")
    print(f"{'level':6} {'stimulus':22} {'regime':8} {'alpha':>7} {'a_rtl':>7} {'total uW':>9} {'pJ/cyc':>7} {'int':>8} {'sw':>8} {'leak nW':>8}")
    for lv, name, rg, a, ar, tot, it, sw, lk in rows:
        print(f"{lv:6} {name:22} {rg:8} {a:7.4f} {ar:7.4f} {tot*1e6:9.2f} {tot*period_ps:7.3f} "
              f"{it*1e6:8.2f} {sw*1e6:8.2f} {lk*1e9:8.2f}")
    print()
    for lv, f in fits.items():
        print(f"{lv:6} P_leak={f['p_leak']*1e9:.2f} nW  P_intercept={f['p_intercept']*1e6:.2f} uW "
              f"(+-{f['intercept_stderr']*1e6:.2f})  slope={f['slope']*1e6:.1f} uW/alpha "
              f"(+-{f['slope_stderr']*1e6:.1f})  R2={f['r2']:.4f}  n={f['n_dyn']}")
    (run_dir / "fit.json").write_text(json.dumps(fits, indent=2))
    plot(args.top, period_ps, rows, fits, REPO / "docs" / "img" / f"{args.top}_power_fit.png")


def plot(top, period_ps, rows, fits, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"synth": "#2a6fdb", "route": "#d9480f"}
    markers = {"idle": "s", "average": "o", "worst": "^"}
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for lv, f in fits.items():
        pts = [r for r in rows if r[0] == lv and r[2] != "leak"]
        for rg, mk in markers.items():
            xs = [r[3] for r in pts if r[2] == rg]
            ys = [r[5] * 1e6 for r in pts if r[2] == rg]
            if xs:
                ax.scatter(xs, ys, marker=mk, color=colors[lv], s=36, label=f"{lv} {rg}")
        xmax = max(r[3] for r in pts) * 1.05
        y0 = (f["p_leak"] + f["p_intercept"]) * 1e6
        ax.plot([0, xmax], [y0, y0 + f["slope"] * xmax * 1e6], color=colors[lv], lw=1, alpha=0.7)
    ax.set_xlabel("α (toggles per non-clock net per cycle, gate-level VCD)")
    ax.set_ylabel("Total power (µW)")
    ax.set_title(f"{top}: P = P_leak + P_intercept + slope·α  (ASAP7 FF, {1e6 / period_ps:.0f} MHz)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
