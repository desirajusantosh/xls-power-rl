"""Validate di/dt (power-swing) search winners with per-cycle post-route power.

  python scripts/validate_didt.py --block mac --top mac_p2 --config mac_p2_gate_didt8

Candidates: each method's best patterns, plus hand-written square-wave stressmarks
(4 cycles of maximum-activity alternation, 4 cycles with valid low and inputs held).
All go through gatesim's gates and reference check at rtl/synth/route. For each
accepted candidate, post-route power is measured cycle by cycle over two steady-
state periods; the swing is the amplitude of that waveform at the target period.

Writes <run>/validate_<config>.json and docs/img/<config>_waveforms.png.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from xpr import gatesim, stimulus as st
from xpr.blocks import BLOCKS
from xpr.search.objectives import harmonic_amplitude
from xpr.search.space import Space

REPO = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["random", "random_legal", "ga", "rl", "hand"]


def pattern_from_words(space: Space, vld_data: list[tuple[int, int]]) -> np.ndarray:
    """[(valid, data), ...] -> (P, W) bits, MSB first {rst=0, vld, data}."""
    W = space.width
    words = [(v << space.block.IN_W) | d for v, d in vld_data]
    return np.array([[(w >> (W - 1 - i)) & 1 for i in range(W)] for w in words], bool)


def hand_written(space: Space, block) -> dict[str, np.ndarray]:
    """Square waves at the target period: 4 cycles alternating, 4 cycles idle-held."""
    if block.NAME == "mac":
        # Both operands alternate together, so the product changes too (0x55^2 vs 86^2).
        x, y = block.pack(0x55, 0x55, 0), block.pack(-86, -86, 0)
        pats = {"hand_square_55aa": [(1, int(x)), (1, int(y))] * 2 + [(0, int(y))] * 4}
    else:
        def pair(a, b):
            return (a << 16) | b
        pats = {"hand_square_maxham": [(1, pair(0x3FFF, 0x3FFF)), (1, pair(0xC000, 0xC000))] * 2
                + [(0, pair(0xC000, 0xC000))] * 4}
        if not space.finite_only:
            pats["hand_square_0ffff"] = [(1, pair(0, 0)), (1, pair(0xFFFF, 0xFFFF))] * 2 + [(0, pair(0xFFFF, 0xFFFF))] * 4
    return {n: pattern_from_words(space, p) for n, p in pats.items()}


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--top", required=True)
    ap.add_argument("--config", required=True, help="search dir name under ~/runs/search")
    ap.add_argument("--per-method", type=int, default=5)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    block = BLOCKS[args.block]
    run_dir = Path.home() / "runs" / "orfs" / args.top
    runs = [json.loads(p.read_text()) for p in sorted((Path.home() / "runs/search" / args.config).glob("*_s*.json"))]
    period, target = runs[0]["period"], runs[0]["swing_period"]
    space = Space(block, period=period, finite_only=runs[0]["finite_only"])

    cands, seen = [], set()
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)
    for m, rs in by_method.items():
        picked = [(r["top"][0]["alpha"], r["top"][0]["pattern"], r["seed"]) for r in rs if r["top"]]
        picked += sorted(((t["alpha"], t["pattern"], r["seed"]) for r in rs for t in r["top"][1:]), key=lambda t: -t[0])
        n = 0
        for score, pat, seed in picked:
            key = (m, json.dumps(pat))
            if key in seen or n >= args.per_method:
                continue
            seen.add(key)
            cands.append({"name": f"{args.config}__{m}_s{seed}_{n}", "method": m, "proxy_swing": score, "pattern": pat})
            n += 1
    for name, pat in hand_written(space, block).items():
        cands.append({"name": f"{args.config}__{name}", "method": "hand", "proxy_swing": None,
                      "pattern": pat.astype(int).tolist()})

    out_path = run_dir / f"validate_{args.config}.json"
    if args.reuse:
        data = json.loads(out_path.read_text())
    else:
        stims = [space.to_stimulus(np.array(c["pattern"], bool), c["name"]) for c in cands]
        data = gatesim.run_suite(block, args.top, stims, run_dir, out_name=out_path.name)
        sims = {(s["stim"], s["level"]): s for s in data["sims"]}
        ok = {c["name"]: all(sims[(c["name"], lv)]["accepted"] for lv in gatesim.LEVELS) for c in cands}
        for c in cands:
            if not ok[c["name"]]:
                print(f"REJECTED at validation: {c['name']}: "
                      f"{[sims[(c['name'], lv)]['failed'] for lv in gatesim.LEVELS]}")
        # Two steady-state periods, aligned to the pattern start (stimulus index 0 = cycle RESET_CYCLES).
        first = st.RESET_CYCLES + period * ((st.RESET_CYCLES + gatesim.FILL_CYCLES + 64) // period)
        vcds = {c["name"]: sims[(c["name"], "route")]["vcd"] for c in cands if ok[c["name"]]}
        data["waveforms"] = gatesim.phase_power(run_dir, args.top, "route", vcds, first, 2 * target)
        data["candidates"] = cands
        data["accepted"] = ok
        out_path.write_text(json.dumps(data, indent=2))

    rows = []
    for c in data["candidates"]:
        w = data["waveforms"].get(c["name"])
        if not w:
            continue
        w = np.array(w)
        rows.append({**c, "swing_w": harmonic_amplitude(w, target), "p2p_w": float(w.max() - w.min()),
                     "mean_w": float(w.mean()), "wave": w})

    print(f"\n{args.config}: target period {target} cycles; post-route per-cycle power, 2 periods")
    print(f"{'method':14} {'proxy swing (best, mean±sd seeds)':>34} {'swing uW':>9} {'p2p uW':>8} {'mean uW':>8}  best candidate")
    best = {}
    for m in METHOD_ORDER:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        b = max(rs, key=lambda r: r["swing_w"])
        best[m] = b
        seeds = np.array([r["best"] for r in by_method.get(m, [])]) if m in by_method else np.array([])
        prox = f"{seeds.mean():.4f} ± {seeds.std(ddof=1):.4f}" if len(seeds) > 1 else "-"
        print(f"{m:14} {prox:>34} {b['swing_w'] * 1e6:9.1f} {b['p2p_w'] * 1e6:8.1f} {b['mean_w'] * 1e6:8.1f}  {b['name'].split('__')[1]}")
    scored = [r for r in rows if r["proxy_swing"] is not None]
    if len(scored) > 2:
        rho = spearman([r["proxy_swing"] for r in scored], [r["swing_w"] for r in scored])
        print(f"Spearman(proxy swing, post-route swing) over {len(scored)} search candidates = {rho:.3f}")
    summary = {m: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items() if k != "pattern"}
               for m, b in best.items()}
    (run_dir / f"validate_{args.config}_summary.json").write_text(json.dumps(summary, indent=2))
    plot(args.config, target, best, REPO / "docs" / "img" / f"{args.config}_waveforms.png")


def plot(config, target, best, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"random": "#999999", "random_legal": "#2a6fdb", "ga": "#2b9348", "rl": "#d9480f", "hand": "black"}
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for m, b in best.items():
        w = np.asarray(b["wave"]) * 1e6
        ax.step(np.arange(len(w) + 1), np.append(w, w[-1]), where="post", color=colors[m],
                label=f"{m}: swing {b['swing_w'] * 1e6:.0f} µW", lw=1.6 if m in ("rl", "ga", "hand") else 1)
    ax.set_xlabel("cycle (two periods)")
    ax.set_ylabel("post-route power (µW)")
    ax.set_title(f"{config}: best per-cycle power waveform per method (target period {target})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
