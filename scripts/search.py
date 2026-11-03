"""Worst-case activity search on the RTL proxy, equal budget for every method.

  python scripts/search.py --block mac --top mac_p2 --methods random random_legal ga rl \
      --seeds 0 1 2 --budget 20000 [--period 8] [--finite-only] [--proxy-level rtl|synth]

Writes ~/runs/search/<top>[_finite][_gate]/<method>_s<seed>.json (curve, top patterns,
reject counts, RL stats) and <method>_s<seed>.rejects.jsonl (reward-hacking log).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from xpr.blocks import BLOCKS
from xpr.gates import RejectLog
from xpr.proxy import Proxy
from xpr.search import baselines, objectives
from xpr.search.space import Scorer, Space

METHODS = {
    "random": baselines.random_search,
    "random_legal": baselines.random_legal,
    "ga": baselines.genetic,
    "cem": baselines.cem,
}


def get_method(name):
    if name in ("rl", "rl_joint"):
        from functools import partial

        from xpr.search.rl import reinforce

        return partial(reinforce, joint_bits=name == "rl_joint")
    return METHODS[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--top", required=True)
    ap.add_argument("--methods", nargs="+", default=["random", "random_legal", "ga", "rl"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--period", type=int, default=8)
    ap.add_argument("--finite-only", action="store_true")
    ap.add_argument("--proxy-level", default="rtl", choices=["rtl", "synth"],
                    help="rtl: XLS Verilog; synth: synthesized netlist (gate-level toggles)")
    ap.add_argument("--objective", default="alpha", choices=["alpha", "swing"],
                    help="alpha: average activity; swing: activity waveform amplitude at --swing-period")
    ap.add_argument("--swing-period", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rerun runs whose result file already exists")
    ap.add_argument("--rerank-k", type=int, default=0,
                    help="after each run, re-score its top-K patterns on the gate (synth) proxy; "
                         "validation then picks by gate alpha. 0 disables.")
    args = ap.parse_args()

    block = BLOCKS[args.block]
    run_dir = Path.home() / "runs" / "orfs" / args.top
    config = (args.top + ("_finite" if args.finite_only else "") + ("_gate" if args.proxy_level == "synth" else "")
              + (f"_didt{args.swing_period}" if args.objective == "swing" else "")
              + (f"_rerank{args.rerank_k}" if args.rerank_k else ""))
    if args.rerank_k and (args.proxy_level != "rtl" or args.objective != "alpha"):
        ap.error("--rerank-k re-scores RTL-proxy alpha searches on the gate proxy")
    objective = objectives.swing(args.swing_period) if args.objective == "swing" else None
    out = Path.home() / "runs" / "search" / config
    out.mkdir(parents=True, exist_ok=True)
    space = Space(block, period=args.period, finite_only=args.finite_only)

    gate = Proxy.build(block, run_dir, args.top, level="synth") if args.rerank_k else None
    with Proxy.build(block, run_dir, args.top, level=args.proxy_level,
                     per_cycle=args.objective == "swing") as proxy:
        for method in args.methods:
            for seed in args.seeds:
                tag = f"{method}_s{seed}"
                # Resumable: a run's JSON is written only when the run completes, so an
                # existing file means it finished; interrupted runs leave no file.
                if (out / f"{tag}.json").exists() and not args.force:
                    print(f"{tag:18} done already, skipping (--force to rerun)")
                    continue
                log_path = out / f"{tag}.rejects.jsonl"
                log_path.unlink(missing_ok=True)
                prov = {"block": block.NAME, "top": args.top, "method": method, "seed": seed,
                        "period": args.period, "finite_only": args.finite_only, "proxy_level": args.proxy_level,
                        "objective": args.objective, "swing_period": args.swing_period}
                scorer = Scorer(space, proxy, args.budget, RejectLog(log_path), prov, objective)
                t0 = time.perf_counter()
                extra = get_method(method)(scorer, __import__("numpy").random.default_rng(seed))
                dt = time.perf_counter() - t0
                result = {
                    **prov, "budget": args.budget, "seconds": dt, "best": scorer.best,
                    "curve": scorer.curve[:: max(1, len(scorer.curve) // 400)] + [scorer.curve[-1]],
                    "rejects": dict(scorer.rejects),
                    "top": [{"alpha": a, "pattern": p.astype(int).tolist()} for a, p in scorer.top_k(10)],
                    "rl_stats": extra if isinstance(extra, list) else None,
                }
                if gate is not None:
                    # Cheap search, expensive re-rank: the top-K by RTL alpha, re-scored on
                    # the gate proxy. Candidates are legal already (top only holds legal ones).
                    top = scorer.top_k(args.rerank_k)
                    res = gate.evaluate([space.words(p) for _, p in top], space.win_start) if top else []
                    reranked = sorted(({"alpha": g.alpha, "rtl_alpha": a, "pattern": p.astype(int).tolist()}
                                       for (a, p), g in zip(top, res)), key=lambda t: -t["alpha"])
                    result["top_rtl"], result["top"] = result["top"], reranked
                    # A run that never found a legal candidate has nothing to re-rank.
                    result["best_rtl"], result["best"] = result["best"], (reranked[0]["alpha"] if reranked else 0.0)
                (out / f"{tag}.json").write_text(json.dumps(result))
                if gate is not None:
                    print(f"{tag:18} reranked on gate proxy: best gate alpha={result['best']:.4f} "
                          f"(RTL best {result['best_rtl']:.4f})")
                legal = 1 - sum(scorer.rejects.values()) / max(1, scorer.used) if scorer.rejects else 1.0
                print(f"{tag:18} best={scorer.best:.4f}  evals={scorer.used}  {dt:6.1f}s  "
                      f"rejects={dict(scorer.rejects)}")
    if gate is not None:
        gate.close()


if __name__ == "__main__":
    main()
