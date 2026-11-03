"""Illustrative AVP / loadline / EDC view of the searched stimuli. A model on top of measured
per-cycle post-route power, not a validated PDN analysis.

    python scripts/avp_loadline.py --top mac_p2 --didt-config mac_p2_gate_didt8 --avg-config mac_p2_gate

Scaling: the block is tiled N times, all tiles running the same pattern in lock-step (how a
power virus behaves on a many-core or many-PE chip). N is chosen so the worst average-power
pattern draws exactly EDC: that is how EDC is characterized in practice.

Model (all parameters are knobs with illustrative defaults, not a real product's):
  VR:  ideal source V = VID - R_LL * I_sense, where I_sense is the package current low-passed
       at the regulator's control bandwidth (adaptive voltage positioning / loadline).
  PDN: series R_pkg + L_pkg from the VR to the die; die decap C_die with ESR R_c. Resonance
       f_res = 1 / (2 pi sqrt(L_pkg C_die)), the "first droop" frequency. L and C are set from
       f_res and a characteristic impedance Z0 = sqrt(L/C), the PDN's design target, so that
       sweeping f_res changes only where the resonance sits, not how strong the PDN is.
  Load: current source I(t) = N * P(t) / V_lib, piecewise constant per clock cycle.
Each transient starts in idle steady state and steps into the pattern (the idle -> active
load step), so it shows the first droop, then the steady ripple at the pattern's period.

Writes docs/img/<top>_avp.png and <run>/avp_<top>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["random", "random_legal", "ga", "cem", "rl", "rl_joint", "llm", "hand"]
COLORS = {"random": "#999999", "random_legal": "#2a6fdb", "ga": "#2b9348", "cem": "#7b2cbf", "rl": "#d9480f",
          "rl_joint": "#f4a261", "llm": "#c9184a", "hand": "black", "edc_step": "#555555"}


def simulate(i_load: np.ndarray, dt: float, p: argparse.Namespace, f_res: np.ndarray) -> np.ndarray:
    """Die voltage for load currents i_load (T, M) under each resonance in f_res (F,) -> (T, F, M)."""
    w = 2 * np.pi * f_res[:, None]  # (F, 1)
    L, C = p.z0 / w, 1.0 / (w * p.z0)
    tau = 1.0 / (2 * np.pi * p.vr_bw)
    i0 = i_load[0][None, :]
    il = np.repeat(i0, len(f_res), axis=0)  # package (inductor) current, (F, M)
    isense = il.copy()
    vc = p.vid - (p.r_ll + p.r_pkg) * il  # idle steady state
    out = np.empty((len(i_load), len(f_res), i_load.shape[1]))
    for k, i in enumerate(i_load):
        vdie = vc + p.r_c * (il - i)
        il = il + dt / L * ((p.vid - p.r_ll * isense) - p.r_pkg * il - vdie)
        vc = vc + dt / C * (il - i)
        isense = isense + dt / tau * (il - isense)
        out[k] = vc + p.r_c * (il - i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--didt-config", required=True)
    ap.add_argument("--avg-config", help="average-power validation config, for the EDC-defining pattern")
    ap.add_argument("--vid", type=float, default=0.80)
    ap.add_argument("--vmin", type=float, default=0.70)
    ap.add_argument("--v-lib", type=float, default=0.70, help="voltage the power was characterized at")
    ap.add_argument("--edc", type=float, default=100.0, help="A")
    ap.add_argument("--r-ll", type=float, default=0.5e-3)
    ap.add_argument("--r-pkg", type=float, default=0.1e-3)
    ap.add_argument("--r-c", type=float, default=0.15e-3, help="die decap ESR")
    ap.add_argument("--z0", type=float, default=0.5e-3, help="PDN characteristic impedance sqrt(L/C)")
    ap.add_argument("--f-res", type=float, default=100e6)
    ap.add_argument("--vr-bw", type=float, default=1e6)
    ap.add_argument("--idle-cycles", type=int, default=400)
    ap.add_argument("--burst-cycles", type=int, default=1600)
    ap.add_argument("--substeps", type=int, default=40)
    p = ap.parse_args()

    run = Path.home() / "runs/orfs" / p.top
    t_clk = int((run / "clk_period_ps.txt").read_text()) * 1e-12
    didt = json.loads((run / f"validate_{p.didt_config}_summary.json").read_text())
    pw = json.loads((run / "power_vcd.json").read_text())["power"]["route"]
    p_idle = pw["idle"]["total_w"]
    if p.avg_config:
        avg = json.loads((run / f"validate_{p.avg_config}_summary.json").read_text())
        p_edc, edc_from = max((b["route_best_w"], m) for m, b in avg.items()
                              if isinstance(b, dict) and b.get("route_best_w"))
    else:
        p_edc, edc_from = max((b["mean_w"], m) for m, b in didt.items())
    n_tiles = p.edc * p.v_lib / p_edc
    amps = n_tiles / p.v_lib  # W of block power -> A of chip current

    methods = [m for m in METHOD_ORDER if m in didt]
    period = len(didt[methods[0]]["wave"]) // 2
    burst = [np.resize(np.array(didt[m]["wave"]), p.burst_cycles) for m in methods]
    names = methods + ["edc_step"]
    burst.append(np.full(p.burst_cycles, p_edc))  # the average-power virus, as a constant step
    per_cycle = np.stack([np.concatenate([np.full(p.idle_cycles, p_idle), b]) for b in burst], axis=1) * amps
    i_load = np.repeat(per_cycle, p.substeps, axis=0)
    dt = t_clk / p.substeps

    f_sweep = np.geomspace(20e6, 1e9, 40)
    v = simulate(i_load, dt, p, np.concatenate([[p.f_res], f_sweep]))
    v_main, v_sweep = v[:, 0, :], v[:, 1:, :]
    # The same EDC step on a flat regulator (R_LL = 0) set to the same full-load DC voltage:
    # without AVP the droop undershoots its final value, which costs extra guardband.
    p_flat = argparse.Namespace(**{**vars(p), "r_ll": 0.0, "vid": p.vid - p.r_ll * p.edc})
    v_flat = simulate(i_load[:, -1:], dt, p_flat, np.array([p.f_res]))[:, 0, 0]
    t_ns = np.arange(len(i_load)) * dt * 1e9

    rows = {}
    steady = slice((p.idle_cycles + p.burst_cycles // 2) * p.substeps, None)
    for j, m in enumerate(names):
        i_mean = per_cycle[p.idle_cycles:, j].mean()
        v_static = p.vid - (p.r_ll + p.r_pkg) * i_mean
        rows[m] = {"i_mean_a": i_mean, "i_peak_a": per_cycle[p.idle_cycles:, j].max(), "v_static": v_static,
                   "v_min": v_main[:, j].min(), "first_droop_mv": (v_static - v_main[:, j].min()) * 1e3,
                   "ripple_mv": (v_main[steady, j].max() - v_main[steady, j].min()) * 1e3,
                   "worst_f_res_mhz": f_sweep[v_sweep[:, :, j].min(axis=0).argmin()] / 1e6}
    v_flat_static = p_flat.vid - p.r_pkg * rows["edc_step"]["i_mean_a"]
    rows["edc_step_flat_vr"] = {**rows["edc_step"], "v_static": v_flat_static, "v_min": v_flat.min(),
                                "first_droop_mv": (v_flat_static - v_flat.min()) * 1e3, "worst_f_res_mhz": None}
    f_swing = 1 / (period * t_clk)
    print(f"{p.top}: clock {t_clk * 1e12:.0f} ps, pattern period {period} cycles -> swing {f_swing / 1e6:.0f} MHz; "
          f"N = {n_tiles:.0f} tiles so that '{edc_from}' (avg-power winner, {p_edc * 1e3:.3f} mW/tile) draws EDC "
          f"{p.edc:.0f} A; f_res {p.f_res / 1e6:.0f} MHz")
    print(f"{'pattern':14} {'I mean A':>8} {'I peak A':>8} {'V static':>8} {'V min':>7} {'droop mV':>8} "
          f"{'ripple mV':>9} {'worst f_res MHz':>15}")
    for m, r in rows.items():
        f = "-" if r["worst_f_res_mhz"] is None else f"{r['worst_f_res_mhz']:.0f}"
        print(f"{m:14} {r['i_mean_a']:8.1f} {r['i_peak_a']:8.1f} {r['v_static']:8.4f} {r['v_min']:7.4f} "
              f"{r['first_droop_mv']:8.1f} {r['ripple_mv']:9.1f} {f:>15}")
    (run / f"avp_{p.top}.json").write_text(json.dumps(
        {"params": vars(p), "n_tiles": n_tiles, "edc_from": edc_from, "swing_mhz": f_swing / 1e6, "rows": rows}, indent=2))
    plot(p, rows, names, t_ns, v_main, v_flat, f_sweep, v_sweep, f_swing, n_tiles,
         REPO / "docs/img" / f"{p.top}_avp.png")


def plot(p, rows, names, t_ns, v_main, v_flat, f_sweep, v_sweep, f_swing, n_tiles, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    label = lambda m: "avg-power virus (step to EDC)" if m == "edc_step" else m

    i = np.linspace(0, p.edc * 1.15, 50)
    a1.plot(i, p.vid - p.r_ll * i, color="#2a6fdb", lw=1.2, label=f"loadline VID − R_LL·I ({p.r_ll * 1e3:.1f} mΩ)")
    a1.plot(i, p.vid - (p.r_ll + p.r_pkg) * i, color="#2a6fdb", lw=1, ls="--", label="+ package IR drop (die, DC)")
    a1.axvline(p.edc, color="#c9184a", lw=1, ls=":", label=f"EDC {p.edc:.0f} A")
    a1.axhline(p.vmin, color="#c9184a", lw=1, ls="-.", label=f"Vmin {p.vmin:.2f} V")
    for m in names:
        r = rows[m]
        a1.errorbar([r["i_mean_a"]], [r["v_static"]], yerr=[[max(0.0, r["v_static"] - r["v_min"])], [0]], fmt="o",
                    color=COLORS[m], ms=4, capsize=2, label=label(m))
    a1.set_xlabel("chip current (A)")
    a1.set_ylabel("die voltage (V)")
    a1.set_title("Static operating point; bar = transient minimum")
    a1.legend(fontsize=6.5, loc="lower left")
    a1.grid(alpha=0.3)

    lo = max(0, (p.idle_cycles - 40) * p.substeps)
    hi = (p.idle_cycles + 260) * p.substeps
    # Readability: the hand stressmark, the searched pattern with the largest ripple, and the EDC step.
    searched = [m for m in names if m not in ("hand", "edc_step")]
    shown = {"hand", "edc_step", max(searched, key=lambda m: rows[m]["ripple_mv"])}
    for j, m in enumerate(names):
        if m in shown:
            a2.plot(t_ns[lo:hi], v_main[lo:hi, j], color=COLORS[m], lw=1.3,
                    ls="--" if m == "edc_step" else "-", label=label(m))
    a2.plot(t_ns[lo:hi], v_flat[lo:hi], color=COLORS["edc_step"], lw=1.2, ls=":",
            label="same step, flat VR (no AVP), same full-load DC")
    a2.axhline(p.vmin, color="#c9184a", lw=1, ls="-.")
    a2.set_xlabel("time (ns)")
    a2.set_ylabel("die voltage (V)")
    a2.set_title(f"Idle → pattern step, f_res {p.f_res / 1e6:.0f} MHz")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=6.5, loc="lower left")

    for j, m in enumerate(names):
        a3.semilogx(f_sweep / 1e6, (rows[m]["v_static"] - v_sweep[:, :, j].min(axis=0)) * 1e3, color=COLORS[m],
                    ls="--" if m == "edc_step" else "-", lw=1.4, label=label(m))
    a3.axvline(f_swing / 1e6, color="black", lw=0.8, ls=":", label=f"pattern swing {f_swing / 1e6:.0f} MHz")
    a3.set_xlabel("PDN resonance f_res (MHz)")
    a3.set_ylabel("worst droop below static point (mV)")
    a3.set_title(f"Droop vs. PDN resonance (Z0 {p.z0 * 1e3:.1f} mΩ held fixed)")
    a3.grid(alpha=0.3, which="both")
    a3.legend(fontsize=6.5)

    fig.suptitle(f"{p.top}: illustrative AVP/EDC model, N = {n_tiles:.0f} lock-step tiles (parameters are not a real product's)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
