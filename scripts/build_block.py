"""Build a block's DSLX to pipelined Verilog with XLS, with a provenance manifest.

  python scripts/build_block.py <block> <stages>      # e.g. mac 2, bf16_mul 4, matmul4 3

Writes ~/runs/<block>/p<stages>/{<Top>.ir, <Top>.opt.ir, <top>_p<stages>.v, manifest.json}.
The Verilog then goes to scripts/orfs_run.sh.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xpr.blocks import BLOCKS
from xpr.xlsflow import BuildSpec, build

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("block", choices=BLOCKS)
    ap.add_argument("stages", type=int)
    args = ap.parse_args()
    block = BLOCKS[args.block]
    spec = BuildSpec(REPO / block.DSLX, top=block.TOP, stages=args.stages)
    out = Path.home() / "runs" / block.NAME / f"p{args.stages}"
    manifest = build(spec, out)
    print(f"{out / f'{spec.module}.v'}  (XLS {manifest['xls_release']})")


if __name__ == "__main__":
    main()
