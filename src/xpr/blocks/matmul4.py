"""INT8 4x4 weight-stationary matrix-vector engine (`Matmul4` proc).

Input word (35 bits): { load[34], row[33:32], x[31:0] }, x = s8[4] with x[0] in
the LEAST significant byte. XLS flattens struct fields first-field-MSB but arrays
element-0-LSB; verified against the RTL (see decisions.md).
  load = 1: W[row] <- x;  load = 0: emit y = x . W, y[c] = sum_r x[r] * W[r][c].
Output word (128 bits): s32[4] with y[0] in the least significant word.
Weights reset to zero.
"""

from __future__ import annotations

import numpy as np

from ..stimulus import Stimulus, flip_bits, held

NAME = "matmul4"
DSLX = "designs/matmul4/matmul4.x"
TOP = "Matmul4"
IN_W = 35
OUT_W = 128


def _s8(u: int) -> int:
    return u - 256 if u >= 128 else u


def pack_x(x) -> int:
    return int(sum((int(v) & 0xFF) << (8 * i) for i, v in enumerate(x)))


def word(load: bool, row: int, x) -> int:
    return (int(load) << 34) | ((row & 3) << 32) | pack_x(x)


def unpack(w: int) -> tuple[bool, int, list[int]]:
    return bool((w >> 34) & 1), (w >> 32) & 3, [_s8((w >> (8 * i)) & 0xFF) for i in range(4)]


def reference(stim: Stimulus) -> list[int]:
    """Expected s32[4] output words, in order (weights start at zero after reset)."""
    W = [[0] * 4 for _ in range(4)]
    out = []
    for w, v in zip(stim.words, stim.valid):
        if not v:
            continue
        load, row, x = unpack(int(w))
        if load:
            W[row] = list(x)
        else:
            y = [sum(x[r] * W[r][c] for r in range(4)) for c in range(4)]
            out.append(sum((yc & 0xFFFFFFFF) << (32 * c) for c, yc in enumerate(y)))
    return out


def same(got: int, want: int) -> bool:
    return got == want


def _with_weights(name: str, regime: str, weights, acts: np.ndarray) -> Stimulus:
    """Weight-load preamble (4 rows), then activation vectors; all valid."""
    words = [word(True, r, weights[r]) for r in range(4)] + [word(False, 0, a) for a in acts]
    return Stimulus(name, regime, np.array(words, np.int64), np.ones(len(words), bool))


def synthetic_weights(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.round(rng.normal(0, 25, (4, 4))), -128, 127).astype(int)


def bitflip(n: int, p: float, seed: int) -> Stimulus:
    """Activation bits flip with probability p per cycle; synthetic weights."""
    w = flip_bits(n - 4, 32, p, seed)
    acts = np.array([[_s8((int(v) >> (8 * i)) & 0xFF) for i in range(4)] for v in w])
    return _with_weights(f"bitflip_p{p:g}_s{seed}", "average", synthetic_weights(seed), acts)


def synthetic_layer(n: int, seed: int) -> Stimulus:
    """ReLU activations about N(0,40) times weights about N(0,25). Synthetic placeholder."""
    rng = np.random.default_rng(seed + 100)
    acts = np.clip(np.round(np.maximum(rng.normal(0, 40, (n - 4, 4)), 0)), 0, 127).astype(int)
    return _with_weights(f"synthetic_layer_s{seed}", "average", synthetic_weights(seed), acts)


def alternating(n: int, wval: int, x0: int, x1: int, name: str) -> Stimulus:
    """Hand-crafted: every weight = wval; all four activations alternate x0, x1."""
    acts = np.array([[x0] * 4 if i % 2 == 0 else [x1] * 4 for i in range(n - 4)])
    return _with_weights(name, "worst", np.full((4, 4), wval), acts)


def suite(n: int = 2000) -> list[Stimulus]:
    return [
        held(n, "leak", "leak"),
        held(n, "idle", "idle"),
        *(bitflip(n, p, seed=1) for p in (0.02, 0.05, 0.1, 0.25, 0.5)),
        synthetic_layer(n, seed=1),
        alternating(n, -86, 0x55, -86, "alt_55_aa"),
        alternating(n, -128, -128, 127, "alt_min_max"),
        alternating(n, -1, 0, -1, "alt_0_ff"),
    ]
