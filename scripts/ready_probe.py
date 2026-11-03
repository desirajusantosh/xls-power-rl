"""Probe a block's input handshake at RTL: hold valid high with a constant word and
report how often the DUT is ready and how many outputs come back.

    PYTHONPATH=src python scripts/ready_probe.py --block systolic4 --top systolic4_p2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xpr import gatesim
from xpr import stimulus as st
from xpr.blocks import BLOCKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--cycles", type=int, default=300)
    ap.add_argument("--word", type=lambda s: int(s, 0), default=0x1)
    ap.add_argument("--run-dir", type=Path, help="holds <top>.v and clk_period_ps.txt (default ~/runs/orfs/<top>)")
    args = ap.parse_args()
    block = BLOCKS[args.block]
    run_dir = args.run_dir or Path.home() / "runs/orfs" / args.top
    out = run_dir / "probe"
    out.mkdir(exist_ok=True)
    vvp = gatesim.compile_level(block, run_dir, args.top, "rtl")
    stim = st.Stimulus("probe", "average", np.full(args.cycles, args.word, np.int64), np.ones(args.cycles, bool))
    cycles = st.write_hex(stim, block.IN_W, out / "stim.hex")
    log, vcd = out / "rtl.log", out / "rtl.vcd"
    import subprocess
    subprocess.run(["vvp", "-n", str(vvp), f"+stim={out / 'stim.hex'}", f"+cycles={cycles}",
                    f"+win_start=0", f"+win_end={cycles}", f"+vcd={vcd}", f"+log={log}"],
                   check=True, capture_output=True, text=True)
    rows = [line.split() for line in log.read_text().splitlines()]
    ready = np.array([r[1] == "1" for r in rows])
    outs = [i for i, r in enumerate(rows) if r[2] == "1"]
    body = ready[st.RESET_CYCLES:]
    print(f"cycles {len(body)}  ready {int(body.sum())}  accept cycles (first 20): {list(np.flatnonzero(body)[:20])}")
    print(f"outputs {len(outs)}  output cycles (first 20): {outs[:20]}")
    print("ready bits, first 120:", "".join("1" if b else "." for b in body[:120]))


if __name__ == "__main__":
    main()
