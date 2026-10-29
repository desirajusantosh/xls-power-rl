"""bf16 multiplier (`Bf16Mul` proc, stock `bfloat16::mul`): data = {a[15:0], b[15:0]}.

Reference model follows apfloat::mul's semantics: subnormal inputs are treated
as zero, round-to-nearest-even, any NaN matches any NaN (RTL emits 0x7FC0).
Results below the normal range are rounded at subnormal precision first and
flushed only if still subnormal, so a product just under the min normal rounds
up to it. That last rule is not in the doc comment; it was found on the RTL.
"""

from __future__ import annotations

import numpy as np

from ..stimulus import Stimulus, flip_bits, held

NAME = "bf16_mul"
DSLX = "designs/bf16_mul/bf16_mul.x"
TOP = "Bf16Mul"
IN_W = 32
OUT_W = 16

QNAN = 0x7FC0


def is_nan(x: int) -> bool:
    return (x >> 7) & 0xFF == 0xFF and x & 0x7F != 0


def mul(a: int, b: int) -> int:
    sa, ea, fa = a >> 15, (a >> 7) & 0xFF, a & 0x7F
    sb, eb, fb = b >> 15, (b >> 7) & 0xFF, b & 0x7F
    sign = (sa ^ sb) << 15
    a_inf, b_inf = ea == 0xFF and fa == 0, eb == 0xFF and fb == 0
    a_zero, b_zero = ea == 0, eb == 0  # subnormals count as zero
    if is_nan(a) or is_nan(b) or (a_inf and b_zero) or (b_inf and a_zero):
        return QNAN
    if a_inf or b_inf:
        return sign | 0x7F80
    if a_zero or b_zero:
        return sign

    prod = (0x80 | fa) * (0x80 | fb)  # in [2^14, 2^16)
    shift = 8 if prod >= 1 << 15 else 7
    exp = ea + eb - 127 + (shift - 7)
    if exp <= 0:
        # Below the normal range, XLS rounds at subnormal precision first and
        # flushes only if the rounded result is still subnormal (verified on RTL
        # by scripts/bf16_edge_probe.py). Rounding can carry up to the min normal.
        shift += 1 - exp
        exp = 1
    mant = _rne_shift(prod, shift)
    if mant == 0x100:
        mant, exp = 0x80, exp + 1
    if exp >= 0xFF:
        return sign | 0x7F80
    if mant < 0x80:
        return sign  # still subnormal after rounding: flush to zero
    return sign | (exp << 7) | (mant & 0x7F)


def _rne_shift(x: int, shift: int) -> int:
    """x >> shift, rounded to nearest, ties to even."""
    if shift >= 40:
        return 0
    q, rem = x >> shift, x & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    return q + (rem > half or (rem == half and q & 1))


def reference(stim: Stimulus) -> list[int]:
    return [mul(int(w) >> 16, int(w) & 0xFFFF) for w, v in zip(stim.words, stim.valid) if v]


def same(got: int, want: int) -> bool:
    return got == want or (is_nan(got) and is_nan(want))


def from_float(x: np.ndarray) -> np.ndarray:
    """float -> bf16 bits, round-to-nearest-even from float32."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def pairs(name: str, regime: str, a, b) -> Stimulus:
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    return Stimulus(name, regime, (a << 16) | b, np.ones(len(a), bool))


def bitflip(n: int, p: float, seed: int) -> Stimulus:
    w = flip_bits(n, 32, p, seed)
    return pairs(f"bitflip_p{p:g}_s{seed}", "average", w >> 16, w & 0xFFFF)


def synthetic_layer(n: int, seed: int) -> Stimulus:
    """ReLU activations about N(0,1) times weights about N(0,0.05). Synthetic placeholder."""
    rng = np.random.default_rng(seed)
    act = np.maximum(rng.normal(0, 1, n), 0)
    w = rng.normal(0, 0.05, n)
    return pairs(f"synthetic_layer_s{seed}", "average", from_float(act), from_float(w))


def alternating(n: int, x: int, y: int, name: str) -> Stimulus:
    """a and b both alternate x, y, x, ...; the product alternates x*x, y*y.

    (Feeding b the reversed sequence would make a*b constant: inputs flip every
    bit while the datapath barely moves. That was an early bug; see decisions.md.)
    """
    seq = np.where(np.arange(n) % 2 == 0, x, y)
    return pairs(name, "worst", seq, seq.copy())


def suite(n: int = 2000) -> list[Stimulus]:
    return [
        held(n, "leak", "leak"),
        held(n, "idle", "idle"),
        *(bitflip(n, p, seed=1) for p in (0.02, 0.05, 0.1, 0.25, 0.5)),
        synthetic_layer(n, seed=1),
        # 1.9921875 <-> -2.0: all 16 bits flip, both operands stay normal.
        alternating(n, 0x3FFF, 0xC000, "alt_maxham_normal"),
        # +0 <-> NaN pattern: all bits flip but the datapath takes special-case paths.
        alternating(n, 0x0000, 0xFFFF, "alt_0_ffff"),
    ]


def edge_cases() -> Stimulus:
    """Operand pairs around the subnormal, overflow, rounding and NaN boundaries."""
    vals = [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x0081, 0x00FF, 0x3F00, 0x3F80, 0x3F81,
            0x3FFF, 0x4000, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3E80, 0x1F80, 0x2000, 0x1FFF]
    a, b = zip(*[(x, y) for x in vals for y in vals])
    return pairs("edge_cases", "average", a, b)
