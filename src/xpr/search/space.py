"""Pattern search space and the fail-closed scorer.

A candidate is a (P, W) bit array: P input vectors of W = IN_W + 2 bits, MSB
first {rst, vld, data[IN_W-1:0]}. It is repeated after the reset prefix for
FILL + window cycles; the window starts after reset and pipeline fill.

Scoring: proxy alpha (RTL toggles per bit per cycle) if every gate passes,
else 0. Rejected candidates are logged with the alpha they would have scored:
that is the reward-hacking record.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np

from .. import stimulus as st
from ..gatesim import FILL_CYCLES
from ..gates import GateResult, RejectLog, Verdict
from ..proxy import Proxy, ProxyResult


@dataclass
class Space:
    block: ModuleType
    period: int = 8
    window: int = 256
    finite_only: bool = False  # bf16: reject Inf/NaN operands

    def __post_init__(self):
        if self.period > FILL_CYCLES:
            raise ValueError("period must be <= FILL_CYCLES so a periodic stream never drains")

    @property
    def width(self) -> int:
        return self.block.IN_W + 2

    @property
    def win_start(self) -> int:
        return st.RESET_CYCLES + FILL_CYCLES

    def words(self, pattern: np.ndarray) -> np.ndarray:
        """(P, W) bits -> full per-cycle word sequence including the reset prefix."""
        weights = 1 << np.arange(self.width - 1, -1, -1, dtype=np.int64)
        vec = pattern.astype(np.int64) @ weights
        n = FILL_CYCLES + self.window
        body = np.resize(vec, n)
        reset = np.full(st.RESET_CYCLES, 1 << (self.block.IN_W + 1), np.int64)
        return np.concatenate([reset, body])

    def to_stimulus(self, pattern: np.ndarray, name: str, n: int = 2000) -> st.Stimulus:
        """The pattern as a regular stimulus (for gate-level validation via gatesim)."""
        weights = 1 << np.arange(self.width - 1, -1, -1, dtype=np.int64)
        vec = np.resize(pattern.astype(np.int64) @ weights, n)
        if np.any(vec >> (self.block.IN_W + 1)):
            raise ValueError("pattern drives rst; only legal patterns can become stimuli")
        valid = ((vec >> self.block.IN_W) & 1).astype(bool)
        data = vec & ((1 << self.block.IN_W) - 1)
        return st.Stimulus(name, "worst", data, valid)

    def gates(self, pattern: np.ndarray, r: ProxyResult) -> Verdict:
        rst, vld = pattern[:, 0], pattern[:, 1]
        results = [
            GateResult("reset_prefix_only", not rst.any(), "rst driven inside the pattern"),
            GateResult("inputs_accepted", bool(vld.any()), "valid never high: stream drains"),
            GateResult("valid_held_until_ready", r.vld_not_rdy == 0, f"{r.vld_not_rdy} stalled cycles"),
            GateResult("nonzero_activity", r.alpha > 0.0, f"alpha={r.alpha:.4g}"),
        ]
        if self.finite_only:
            results.append(GateResult("finite_operands", self._finite(pattern), "Inf/NaN operand"))
        return Verdict(tuple(results))

    def _finite(self, pattern: np.ndarray) -> bool:
        # bf16 operands: data = {a[15:0], b[15:0]}; exponent = bits [14:7] of each.
        data = pattern[:, 2:]
        for off in (0, 16):
            exp = data[:, off + 1 : off + 9]
            if np.any(exp.all(axis=1)):
                return False
        return True


@dataclass
class Scorer:
    """Fail-closed scoring with an evaluation budget and a reward-hacking log.

    `objective` maps a proxy result to the score of a legal candidate (default:
    alpha). Gates always look at the raw proxy result, whatever the objective.
    """

    space: Space
    proxy: Proxy
    budget: int
    reject_log: RejectLog | None = None
    provenance: dict = field(default_factory=dict)
    objective: Callable[[ProxyResult], float] | None = None
    used: int = 0
    best: float = 0.0
    curve: list[tuple[int, float]] = field(default_factory=list)  # (evals, best-so-far)
    rejects: Counter = field(default_factory=Counter)
    top: dict[str, tuple[float, np.ndarray]] = field(default_factory=dict)
    _cache: dict[str, tuple[ProxyResult]] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def score(self, patterns: np.ndarray, tag: str = "") -> np.ndarray:
        """patterns: (B, P, W) bool. Every candidate costs one evaluation."""
        patterns = np.asarray(patterns, bool)[: self.remaining]
        keys = [hashlib.blake2b(p.tobytes(), digest_size=12).hexdigest() for p in patterns]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo:
            res = self.proxy.evaluate([self.space.words(patterns[i]) for i in todo], self.space.win_start)
            for i, r in zip(todo, res):
                self._cache[keys[i]] = (r,)
        scores = np.zeros(len(patterns))
        for i, (p, k) in enumerate(zip(patterns, keys)):
            (r,) = self._cache[k]
            verdict = self.space.gates(p, r)
            value = self.objective(r) if self.objective else r.alpha
            self.used += 1
            if verdict.passed:
                scores[i] = value
                if k not in self.top or self.top[k][0] < value:
                    self.top[k] = (value, p.copy())
            else:
                for name in verdict.failed:
                    self.rejects[name] += 1
                # Only rejected attempts that beat the legal best are worth a log line.
                if self.reject_log and value > self.best:
                    self.reject_log.record(verdict, reward=value,
                                           provenance={**self.provenance, "evals": self.used, "tag": tag,
                                                       "pattern": p.astype(int).tolist()})
            if scores[i] > self.best:
                self.best = scores[i]
            self.curve.append((self.used, self.best))
        return scores

    def result(self, pattern: np.ndarray) -> ProxyResult:
        """Proxy result of a pattern this scorer has already evaluated."""
        (r,) = self._cache[hashlib.blake2b(np.asarray(pattern, bool).tobytes(), digest_size=12).hexdigest()]
        return r

    def top_k(self, k: int) -> list[tuple[float, np.ndarray]]:
        return sorted(self.top.values(), key=lambda t: -t[0])[:k]


def run(method: Callable[[Scorer, np.random.Generator], None], scorer: Scorer, seed: int) -> Scorer:
    method(scorer, np.random.default_rng(seed))
    return scorer
