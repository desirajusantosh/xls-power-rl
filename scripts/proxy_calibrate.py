"""Check the Verilator proxy against Icarus VCD alpha on a block's stimulus suite.

  python scripts/proxy_calibrate.py --block mac --top mac_p2 [--level rtl|synth]

Needs power_vcd.json from scripts/vcd_power.py (it holds Icarus alpha per level).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from xpr import gatesim, stimulus as st
from xpr.blocks import BLOCKS
from xpr.proxy import Proxy


def full_words(block, stim: st.Stimulus) -> np.ndarray:
    reset = np.full(st.RESET_CYCLES, 1 << (block.IN_W + 1), np.int64)
    body = (stim.valid.astype(np.int64) << block.IN_W) | stim.words
    return np.concatenate([reset, body])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--top", required=True)
    ap.add_argument("--level", default="rtl", choices=["rtl", "synth"])
    args = ap.parse_args()
    block = BLOCKS[args.block]
    run_dir = Path.home() / "runs" / "orfs" / args.top
    data = json.loads((run_dir / "power_vcd.json").read_text())
    icarus = {s["stim"]: s["alpha"] for s in data["sims"] if s["level"] == args.level}
    icarus_bits = next(s["n_bits"] for s in data["sims"] if s["level"] == args.level)
    win_start = st.RESET_CYCLES + gatesim.FILL_CYCLES

    suite = [s for s in block.suite(2000) if s.regime != "leak"]
    with Proxy.build(block, run_dir, args.top, level=args.level) as p:
        res = p.evaluate([full_words(block, s) for s in suite], win_start)
        seqs = [full_words(block, s)[: win_start + 256] for s in suite] * 50
        t0 = time.perf_counter()
        p.evaluate(seqs, win_start)
        dt = (time.perf_counter() - t0) / len(seqs)

    print(f"{'stimulus':22} {'icarus':>8} {'proxy':>8} {'diff':>8}")
    for s, r in zip(suite, res):
        print(f"{s.name:22} {icarus[s.name]:8.4f} {r.alpha:8.4f} {r.alpha - icarus[s.name]:+8.4f}")
    print(f"bits counted: proxy {res[0].bits}, icarus {icarus_bits}; vld_not_rdy max: {max(r.vld_not_rdy for r in res)}")
    worst = max(abs(r.alpha - icarus[s.name]) for s, r in zip(suite, res))
    print(f"max |diff|: {worst:.4f}")
    print(f"throughput: {dt * 1e3:.3f} ms per 256-cycle evaluation")


if __name__ == "__main__":
    main()
