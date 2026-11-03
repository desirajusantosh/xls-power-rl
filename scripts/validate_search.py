"""Validate search winners at gate level (synth, post-route) and measure proxy fidelity.

  python scripts/validate_search.py --block mac --top mac_p2 [--config mac_p2] [--per-method 6]

For each method: each seed's best pattern plus the best of the rest, deduplicated.
The suite's stimuli (bit-flip sweep, hand-crafted, synthetic layer) come along as
reference points from power_vcd.json. Every candidate goes through the same
gates and reference check as any other stimulus.

Writes <run>/validate_<config>.json and docs/img/<config>_search.png.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from xpr import gatesim
from xpr.blocks import BLOCKS
from xpr.search.space import Space

REPO = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["random", "random_legal", "ga", "cem", "rl", "rl_joint", "llm", "llm_transfer"]


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--top", required=True)
    ap.add_argument("--config", help="search dir name under ~/runs/search (default: top)")
    ap.add_argument("--per-method", type=int, default=6)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    block = BLOCKS[args.block]
    config = args.config or args.top
    run_dir = Path.home() / "runs" / "orfs" / args.top
    search_dir = Path.home() / "runs" / "search" / config
    runs = [json.loads(p.read_text()) for p in sorted(search_dir.glob("*_s*.json"))]
    period = runs[0]["period"]
    space = Space(block, period=period, finite_only=runs[0]["finite_only"])
    # The proxy the search optimized: fidelity is measured for that proxy's alpha.
    proxy_level = runs[0].get("proxy_level", "rtl")

    # Candidate selection: each seed's best, then the next best across seeds.
    cands, seen = [], set()
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)
    for m, rs in by_method.items():
        picked = []
        for r in rs:
            if r["top"]:
                picked.append((r["top"][0]["alpha"], r["top"][0]["pattern"], r["seed"]))
        rest = sorted(((t["alpha"], t["pattern"], r["seed"]) for r in rs for t in r["top"][1:]), key=lambda t: -t[0])
        picked += rest
        n = 0
        for alpha, pat, seed in picked:
            key = (m, json.dumps(pat))
            if key in seen or n >= args.per_method:
                continue
            seen.add(key)
            cands.append({"name": f"{config}__{m}_s{seed}_{n}", "method": m, "seed": seed,
                          "proxy_alpha": alpha, "pattern": pat})
            n += 1

    out_path = run_dir / f"validate_{config}.json"
    if args.reuse:
        data = json.loads(out_path.read_text())
    else:
        stims = [space.to_stimulus(np.array(c["pattern"], bool), c["name"]) for c in cands]
        data = gatesim.run_suite(block, args.top, stims, run_dir, out_name=out_path.name)
        data["candidates"] = cands
        out_path.write_text(json.dumps(data, indent=2))

    period_ps = data["clk_period_ps"]
    sims = {(s["stim"], s["level"]): s for s in data["sims"]}
    rows = []
    for c in data["candidates"]:
        rtl = sims[(c["name"], "rtl")]
        route = data["power"]["route"].get(c["name"])
        synth = data["power"]["synth"].get(c["name"])
        ok = all(sims[(c["name"], lv)]["accepted"] for lv in ("rtl", "synth", "route"))
        if not ok:
            print(f"REJECTED at validation: {c['name']}: "
                  f"{[sims[(c['name'], lv)]['failed'] for lv in ('rtl', 'synth', 'route')]}")
        rows.append({**c, "accepted": ok, "rtl_alpha": rtl["alpha"],
                     "proxy_alpha_level": sims[(c["name"], proxy_level)]["alpha"],
                     "gate_alpha": sims[(c["name"], "route")]["alpha"],
                     "synth_w": synth["total_w"] if synth else None, "route_w": route["total_w"] if route else None})

    # Reference points from the block's suite (same run, same flow).
    base = json.loads((run_dir / "power_vcd.json").read_text())
    base_rtl = {s["stim"]: s["alpha"] for s in base["sims"] if s["level"] == "rtl"}
    base_proxy = {s["stim"]: s["alpha"] for s in base["sims"] if s["level"] == proxy_level}
    ref = [{"name": n, "method": "suite:" + next(s["regime"] for s in base["sims"] if s["stim"] == n),
            "rtl_alpha": base_rtl[n], "proxy_alpha_level": base_proxy[n],
            "route_w": p["total_w"], "synth_w": base["power"]["synth"][n]["total_w"]}
           for n, p in base["power"]["route"].items() if n not in ("leak", "idle")]
    if space.finite_only:
        ref = [r for r in ref if r["name"] != "alt_0_ffff"]  # NaN pattern is illegal in finite-only mode

    print(f"\n{config}: {len(rows)} search candidates validated at {period_ps} ps")
    print(f"{'method':14} {'proxy best (mean±sd over seeds)':>32} {'best route uW':>14} {'pJ/cyc':>7} {'best synth uW':>14}")
    summary = {}
    for m in METHOD_ORDER:
        rs = by_method.get(m, [])
        if not rs:
            continue
        bests = np.array([r["best"] for r in rs])
        val = [x for x in rows if x["method"] == m and x["accepted"] and x["route_w"]]
        br = max(val, key=lambda x: x["route_w"]) if val else None
        bs = max((x["synth_w"] for x in val), default=float("nan"))
        summary[m] = {"proxy_best_mean": float(bests.mean()), "proxy_best_sd": float(bests.std(ddof=1)) if len(bests) > 1 else 0.0,
                      "proxy_best_max": float(bests.max()), "route_best_w": br["route_w"] if br else None,
                      "route_best_name": br["name"] if br else None, "synth_best_w": bs}
        print(f"{m:14} {bests.mean():20.4f} ± {bests.std(ddof=1) if len(bests) > 1 else 0:.4f}      "
              f"{br['route_w'] * 1e6 if br else float('nan'):14.1f} {br['route_w'] * period_ps if br else float('nan'):7.3f} {bs * 1e6:14.1f}")
    best_ref = max(ref, key=lambda r: r["route_w"])
    print(f"{'suite best':14} {best_ref['name']:>32} {best_ref['route_w'] * 1e6:14.1f} {best_ref['route_w'] * period_ps:7.3f}")

    allpts = [r for r in rows if r["accepted"] and r["route_w"]] + ref
    rho_route = spearman([r["proxy_alpha_level"] for r in allpts], [r["route_w"] for r in allpts])
    rho_rtl_route = spearman([r["rtl_alpha"] for r in allpts], [r["route_w"] for r in allpts])
    rho_synth_route = spearman([r["synth_w"] for r in allpts], [r["route_w"] for r in allpts])
    proxy_pick = max(allpts, key=lambda r: r["proxy_alpha_level"])
    power_pick = max(allpts, key=lambda r: r["route_w"])
    print(f"\nproxy fidelity over {len(allpts)} points (search proxy = {proxy_level} alpha): "
          f"Spearman(proxy alpha, route power) = {rho_route:.3f}; "
          f"Spearman(RTL alpha, route) = {rho_rtl_route:.3f}; Spearman(synth power, route) = {rho_synth_route:.3f}")
    print(f"proxy's pick: {proxy_pick['name']} ({proxy_pick['route_w'] * 1e6:.1f} uW); "
          f"true best: {power_pick['name']} ({power_pick['route_w'] * 1e6:.1f} uW); "
          f"regret {100 * (1 - proxy_pick['route_w'] / power_pick['route_w']):.1f}%")
    summary["_fidelity"] = {"n": len(allpts), "proxy_level": proxy_level, "spearman_proxy_route": rho_route,
                            "spearman_rtl_route": rho_rtl_route, "spearman_synth_route": rho_synth_route,
                            "proxy_pick": proxy_pick["name"], "proxy_pick_route_w": proxy_pick["route_w"],
                            "true_best": power_pick["name"], "true_best_route_w": power_pick["route_w"],
                            "suite_best": best_ref["name"], "suite_best_route_w": best_ref["route_w"]}
    (run_dir / f"validate_{config}_summary.json").write_text(json.dumps(summary, indent=2))
    plot(config, runs, rows, ref, REPO / "docs" / "img" / f"{config}_search.png")


def plot(config, runs, rows, ref, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#999999", "random_legal": "#2a6fdb", "ga": "#2b9348", "cem": "#7b2cbf", "rl": "#d9480f",
              "rl_joint": "#f4a261", "llm": "#c9184a", "llm_transfer": "#ff8fab"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for m in METHOD_ORDER:
        rs = [r for r in runs if r["method"] == m]
        if not rs:
            continue
        grid = np.linspace(1, rs[0]["budget"], 200)
        ys = np.array([np.interp(grid, [c[0] for c in r["curve"]], [c[1] for c in r["curve"]]) for r in rs])
        a1.plot(grid, ys.mean(0), color=colors[m], label=f"{m} (n={len(rs)})")
        a1.fill_between(grid, ys.min(0), ys.max(0), color=colors[m], alpha=0.15)
    a1.set_xlabel("proxy evaluations")
    a1.set_ylabel("best legal RTL α so far")
    a1.set_title(f"{config}: search progress (mean, min–max over seeds)")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)

    for m in METHOD_ORDER:
        pts = [r for r in rows if r["method"] == m and r["accepted"] and r["route_w"]]
        a2.scatter([p["rtl_alpha"] for p in pts], [p["route_w"] * 1e6 for p in pts], color=colors[m], s=28, label=m)
    a2.scatter([r["rtl_alpha"] for r in ref], [r["route_w"] * 1e6 for r in ref], marker="x", color="black", s=30,
               label="suite (bit-flip, hand-crafted, layer)")
    a2.set_xlabel("RTL α (proxy)")
    a2.set_ylabel("post-route power (µW)")
    a2.set_title("proxy vs. post-route power")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
