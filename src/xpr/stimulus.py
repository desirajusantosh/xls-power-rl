"""Block-independent stimulus: per-cycle input words for a streaming proc.

A block module (xpr.blocks.*) defines IN_W/OUT_W, how fields pack into a word,
its stimulus suite and its reference model. This module turns any stimulus
into a hex file for the testbench and a gate-checkable trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .gates import Trace

RESET_CYCLES = 4


@dataclass
class Stimulus:
    name: str
    regime: str  # leak | idle | average | worst
    words: np.ndarray  # int64, one input word per cycle (reset prefix excluded)
    valid: np.ndarray  # bool
    fields: dict[str, Any] = field(default_factory=dict)  # decoded fields for the reference model


def _bits(words: np.ndarray, width: int) -> np.ndarray:
    return ((words[:, None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)


def to_trace(stim: Stimulus, in_w: int, ready: np.ndarray, window: tuple[int, int]) -> Trace:
    """Full trace including the reset prefix; `ready` is sampled from the DUT."""
    n = len(stim.words)
    rst = np.zeros(RESET_CYCLES + n, bool)
    rst[:RESET_CYCLES] = True
    valid = np.concatenate([np.zeros(RESET_CYCLES, bool), stim.valid])
    words = np.concatenate([np.zeros(RESET_CYCLES, np.int64), stim.words])
    return Trace(rst, valid, ready.astype(bool), _bits(words, in_w), window)


def write_hex(stim: Stimulus, in_w: int, path: Path) -> int:
    """{rst, vld, data} per cycle, reset prefix first. Returns the cycle count."""
    digits = (in_w + 2 + 3) // 4
    lines = [f"{1 << (in_w + 1):0{digits}x}"] * RESET_CYCLES
    lines += [f"{(int(v) << in_w) | int(w):0{digits}x}" for v, w in zip(stim.valid, stim.words)]
    path.write_text("\n".join(lines) + "\n")
    return len(lines)


def held(n: int, name: str, regime: str) -> Stimulus:
    """Inputs held at zero with valid low (idle), or the same with the clock stopped (leak)."""
    return Stimulus(name, regime, np.zeros(n, np.int64), np.zeros(n, bool))


def flip_bits(n: int, width: int, p: float, seed: int) -> np.ndarray:
    """Random walk over `width`-bit words: each bit flips with probability p per cycle."""
    rng = np.random.default_rng(seed)
    bits = np.cumsum(rng.random((n, width)) < p, axis=0) % 2
    return bits @ (1 << np.arange(width - 1, -1, -1, dtype=np.int64))
