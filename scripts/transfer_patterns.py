"""Copy another config's winning patterns into a search directory as a pseudo-run.

  python scripts/transfer_patterns.py --src matmul4_p3_rerank20 --method llm --dst systolic4_p2_rerank20 \
      --as-method llm_transfer

Blocks with the same interface (e.g. matmul4 and the systolic array) can be validated
on the same input patterns. The result goes through validate_search.py like any run,
so the transferred patterns face the same gates, reference check and post-route power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source search dir under ~/runs/search")
    ap.add_argument("--method", required=True, help="method whose runs to copy (e.g. llm)")
    ap.add_argument("--dst", required=True, help="destination search dir under ~/runs/search")
    ap.add_argument("--as-method", required=True)
    args = ap.parse_args()
    root = Path.home() / "runs" / "search"
    out = root / args.dst
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((root / args.src).glob(f"{args.method}_s*.json")):
        r = json.loads(f.read_text())
        seed = r["seed"]
        r.update({"method": args.as_method, "transferred_from": f"{args.src}/{f.name}",
                  "curve": r.get("curve", [])})
        (out / f"{args.as_method}_s{seed}.json").write_text(json.dumps(r))
        print(f"{f} -> {out / f'{args.as_method}_s{seed}.json'} ({len(r['top'])} patterns)")


if __name__ == "__main__":
    main()
