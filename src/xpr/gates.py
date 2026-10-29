"""Fail-closed validity gates for stimulus traces.

A trace is the cycle-by-cycle input stimulus driven into a block. Every gate
must pass before a trace's power counts toward a result or a reward. Rejected
traces that carried a reward are logged as the reward-hacking record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Regime = Literal["idle", "average", "worst"]

X = 2  # data value marking an X/Z bit; legal bits are 0 and 1


@dataclass
class Trace:
    """Input stimulus over T cycles.

    rst, valid, ready: bool arrays of shape (T,). ready is sampled from the DUT.
    data: uint8 array of shape (T, W), values 0, 1, or X.
    window: half-open [start, end) measurement window in cycles.
    """

    rst: np.ndarray
    valid: np.ndarray
    ready: np.ndarray
    data: np.ndarray
    window: tuple[int, int]

    @property
    def cycles(self) -> int:
        return len(self.rst)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]


def input_alpha(trace: Trace) -> float:
    """Mean input toggle rate per bit per cycle inside the window."""
    start, end = trace.window
    seg = trace.data[start:end].astype(np.int16)
    if len(seg) < 2 or seg.shape[1] == 0:
        return 0.0
    return float(np.mean(seg[1:] != seg[:-1]))


def _fires(trace: Trace) -> np.ndarray:
    return np.flatnonzero(trace.valid & trace.ready & ~trace.rst)


def check_trace(trace: Trace, *, latency: int, regime: Regime) -> Verdict:
    """Run every gate. Never short-circuits, so the log shows all failures."""
    T = trace.cycles
    start, end = trace.window
    results: list[GateResult] = []

    shapes_ok = (
        trace.valid.shape == trace.ready.shape == trace.rst.shape == (T,)
        and trace.data.ndim == 2
        and trace.data.shape[0] == T
    )
    results.append(GateResult("shapes", shapes_ok))
    if not shapes_ok:
        return Verdict(tuple(results))

    in_bounds = 0 <= start < end <= T
    results.append(GateResult("window_bounds", in_bounds, f"window={trace.window}, T={T}"))

    xz = np.argwhere(trace.data > 1)
    results.append(
        GateResult("no_xz", len(xz) == 0, f"first X/Z at (cycle, bit)={tuple(xz[0])}" if len(xz) else "")
    )

    # Reset may only be a single block at the start of the trace.
    rst_idx = np.flatnonzero(trace.rst)
    release = int(rst_idx[-1]) + 1 if len(rst_idx) else 0
    rst_ok = len(rst_idx) == 0 or (rst_idx[0] == 0 and len(rst_idx) == release)
    results.append(GateResult("reset_prefix_only", rst_ok, f"reset cycles={rst_idx.tolist()[:8]}"))

    fill_ok = start >= release + latency
    results.append(
        GateResult("window_after_fill", fill_ok, f"start={start}, reset release={release}, latency={latency}")
    )

    # valid/ready: once valid is raised it holds, with stable data, until accepted.
    stall = np.flatnonzero(trace.valid[:-1] & ~trace.ready[:-1] & ~trace.rst[:-1])
    drop = [int(t) for t in stall if not trace.valid[t + 1]]
    change = [int(t) for t in stall if not np.array_equal(trace.data[t], trace.data[t + 1])]
    results.append(GateResult("valid_held_until_ready", not drop, f"valid dropped after cycles {drop[:8]}"))
    results.append(GateResult("data_stable_while_stalled", not change, f"data changed after cycles {change[:8]}"))

    fires = _fires(trace)
    win_valid = trace.valid[start:end]
    if regime == "idle":
        results.append(GateResult("idle_valid_low", not win_valid.any(), "valid asserted inside idle window"))
        held = len(trace.data[start:end]) < 2 or not np.any(np.diff(trace.data[start:end].astype(np.int16), axis=0))
        results.append(GateResult("idle_inputs_held", bool(held), "data toggled inside idle window"))
    else:
        # The pipeline has not drained while an input was accepted within the
        # last `latency` cycles of the window (periodic patterns may idle briefly).
        drain_ok = len(fires) > 0 and (end - 1) - int(fires[-1]) < latency
        last = int(fires[-1]) if len(fires) else None
        results.append(GateResult("window_before_drain", drain_ok, f"end={end}, last accepted input={last}"))
        alpha = input_alpha(trace)
        results.append(GateResult("nonzero_activity", alpha > 0.0, f"input alpha={alpha:.4g}"))

    return Verdict(tuple(results))


class RejectLog:
    """Append-only JSONL record of rejected traces that carried a reward."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, verdict: Verdict, *, reward: float, provenance: dict[str, Any]) -> None:
        entry = {
            "reward": reward,
            "failed": verdict.failed,
            "results": [asdict(r) for r in verdict.results if not r.passed],
            "provenance": provenance,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
