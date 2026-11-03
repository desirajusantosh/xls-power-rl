"""Pre-flight check before spending hours in ORFS: simulate a block's stimulus suite on the
XLS RTL and fail if any stimulus is rejected (handshake gates or reference mismatch).

    PYTHONPATH=src python scripts/rtl_check.py --block systolic4 --rtl ~/runs/systolic4/p2/systolic4_p2.v

Works in a scratch dir next to the RTL, so it needs no ORFS run. Exits 1 on any rejection.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from xpr import gatesim
from xpr.blocks import BLOCKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--rtl", required=True, type=Path)
    ap.add_argument("--cycles", type=int, default=500)
    args = ap.parse_args()
    block = BLOCKS[args.block]
    top = args.rtl.stem
    work = args.rtl.parent / "rtl_check"
    shutil.rmtree(work, ignore_errors=True)
    (work / "sim").mkdir(parents=True)
    shutil.copy(args.rtl, work / f"{top}.v")
    (work / "clk_period_ps.txt").write_text("1000\n")  # zero-delay RTL sim: the period is irrelevant

    vvp = gatesim.compile_level(block, work, top, "rtl")
    bad = 0
    for s in block.suite(args.cycles):
        if s.regime == "leak":
            continue
        r = gatesim.simulate(block, vvp, s, work / "sim" / s.name, "rtl")
        bad += not r.accepted
        print(f"{'ok      ' if r.accepted else 'REJECTED'} {s.name:24s} {' '.join(r.failed)}")
    print(f"rtl_check {top}: {bad} rejected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
