"""INT8 MAC (`Mac` proc): data[16:9] = a (s8), data[8:1] = b (s8), data[0] = last."""

from __future__ import annotations

import numpy as np

from ..stimulus import Stimulus, flip_bits, held

NAME = "mac"
DSLX = "designs/mac/mac.x"
TOP = "Mac"
IN_W = 17
OUT_W = 32


def pack(a, b, last) -> np.ndarray:
    a = np.asarray(a, np.int64) & 0xFF
    b = np.asarray(b, np.int64) & 0xFF
    return (a << 9) | (b << 1) | np.asarray(last, np.int64)


def _to_s8(u: np.ndarray) -> np.ndarray:
    return np.where(u >= 128, u - 256, u)


def stream(name: str, regime: str, a, b, dot_len: int) -> Stimulus:
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    last = (np.arange(len(a)) % dot_len) == dot_len - 1
    return Stimulus(name, regime, pack(a, b, last), np.ones(len(a), bool), {"a": a, "b": b, "last": last})


def unpack(w: int) -> tuple[int, int, bool]:
    a, b = (w >> 9) & 0xFF, (w >> 1) & 0xFF
    return (a - 256 if a >= 128 else a), (b - 256 if b >= 128 else b), bool(w & 1)


def reference(stim: Stimulus) -> list[int]:
    """Accumulated dot products, emitted on `last`, as unsigned 32-bit words.

    Decodes the input words themselves, so every stimulus is checked, including
    search patterns that carry no decoded fields.
    """
    out, acc = [], 0
    for w, v in zip(stim.words, stim.valid):
        if not v:
            continue
        a, b, last = unpack(int(w))
        acc += a * b
        if last:
            out.append(acc & 0xFFFFFFFF)
            acc = 0
    return out


def same(got: int, want: int) -> bool:
    return got == want


def bitflip(n: int, p: float, seed: int, dot_len: int = 64) -> Stimulus:
    """Each bit of a and b flips with probability p per cycle (controls input toggle rate)."""
    w = flip_bits(n, 16, p, seed)
    return stream(f"bitflip_p{p:g}_s{seed}", "average", _to_s8(w >> 8), _to_s8(w & 0xFF), dot_len)


def synthetic_layer(n: int, seed: int, dot_len: int = 64) -> Stimulus:
    """Synthetic INT8 layer: ReLU activations (about half zeros) times Gaussian weights.

    Placeholder until real activations and weights from a model layer are in place.
    """
    rng = np.random.default_rng(seed)
    act = np.clip(np.round(np.maximum(rng.normal(0, 40, n), 0)), 0, 127)
    w = np.clip(np.round(rng.normal(0, 25, n)), -128, 127)
    return stream(f"synthetic_layer_s{seed}", "average", act, w, dot_len)


def alternating(n: int, x: int, y: int, name: str, dot_len: int = 64) -> Stimulus:
    """Hand-crafted worst case: a and b alternate between x and y every cycle."""
    seq = np.where(np.arange(n) % 2 == 0, x, y)
    return stream(name, "worst", seq, seq.copy(), dot_len)


def suite(n: int = 2000) -> list[Stimulus]:
    return [
        held(n, "leak", "leak"),
        held(n, "idle", "idle"),
        *(bitflip(n, p, seed=1) for p in (0.02, 0.05, 0.1, 0.25, 0.5)),
        synthetic_layer(n, seed=1),
        alternating(n, 0, -1, "alt_0_ff"),
        alternating(n, 0x55, -86, "alt_55_aa"),
        alternating(n, -128, 127, "alt_min_max"),
    ]
