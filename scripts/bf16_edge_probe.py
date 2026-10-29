"""Probe bf16_mul's RTL on boundary operands and log where it departs from the reference.

  python scripts/bf16_edge_probe.py --top bf16mul_p2

Runs the edge-case operand grid (subnormal, overflow, rounding, NaN boundaries)
through XLS's RTL and compares to xpr.blocks.bf16_mul.mul. Mismatches are
findings to document, not failures: they show the hardware's actual semantics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xpr import gatesim
from xpr.blocks import bf16_mul as blk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="bf16mul_p2")
    args = ap.parse_args()
    run_dir = Path.home() / "runs" / "orfs" / args.top
    (run_dir / "sim").mkdir(exist_ok=True)
    stim = blk.edge_cases()
    vvp = gatesim.compile_level(blk, run_dir, args.top, "rtl")
    r = gatesim.simulate(blk, vvp, stim, run_dir / "sim" / "edge_cases", "rtl")
    log = (run_dir / "sim/edge_cases/rtl.log").read_text().splitlines()
    got = [int(row.split()[3], 16) for row in log if row.split()[2] == "1"]
    want = blk.reference(stim)
    pairs = [(int(w) >> 16, int(w) & 0xFFFF) for w in stim.words]
    bad = [(a, b, g, w) for (a, b), g, w in zip(pairs, got, want) if not blk.same(g, w)]
    print(f"{len(got)}/{len(want)} outputs observed, {len(bad)} differ from the reference")
    for a, b, g, w in bad[:40]:
        print(f"  {a:04x} * {b:04x}: rtl={g:04x} ref={w:04x}")
    nan_outs = sorted({g for g in got if blk.is_nan(g)})
    print("NaN encodings produced by RTL:", [f"{x:04x}" for x in nan_outs])
    print("gate failures:", [f for f in r.failed if f != "outputs_match_reference"])


if __name__ == "__main__":
    main()
