import numpy as np
import pytest

from xpr import stimulus as st
from xpr.blocks import BLOCKS, bf16_mul, mac
from xpr.gates import check_trace, input_alpha


def test_mac_word_packing_matches_xls_struct_layout():
    s = mac.stream("x", "average", [-128], [3], dot_len=1)
    assert int(s.words[0]) == (0x80 << 9) | (0x03 << 1) | 1


def test_hex_has_reset_prefix(tmp_path):
    s = st.held(10, "idle", "idle")
    n = st.write_hex(s, mac.IN_W, tmp_path / "s.hex")
    lines = (tmp_path / "s.hex").read_text().split()
    assert n == 10 + st.RESET_CYCLES == len(lines)
    assert all(int(l, 16) >> (mac.IN_W + 1) == 1 for l in lines[: st.RESET_CYCLES])
    assert all(int(l, 16) == 0 for l in lines[st.RESET_CYCLES :])


def test_hex_width_fits_32_bit_inputs(tmp_path):
    s = bf16_mul.pairs("x", "average", [0xFFFF], [0xFFFF])
    st.write_hex(s, bf16_mul.IN_W, tmp_path / "s.hex")
    last = (tmp_path / "s.hex").read_text().split()[-1]
    assert int(last, 16) == (1 << 32) | 0xFFFFFFFF


def test_mac_reference():
    s = mac.stream("x", "average", [1, 2, 3, 4], [4, -5, 6, 1], dot_len=2)
    assert mac.reference(s) == [(4 - 10) & 0xFFFFFFFF, 22]


def test_mac_reference_needs_no_decoded_fields():
    # Search patterns become stimuli with only raw words; the reference must still check them.
    s = mac.stream("x", "average", [-128, 127], [-128, 1], dot_len=2)
    raw = st.Stimulus("raw", "worst", s.words.copy(), s.valid.copy())
    assert mac.reference(raw) == mac.reference(s) == [16384 + 127]


def test_matmul4_reference_matches_dslx_test():
    from xpr.blocks import matmul4 as mm

    W = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    words = [mm.word(True, r, W[r]) for r in range(4)]
    words += [mm.word(False, 0, [1, 0, 0, -1]), mm.word(True, 0, [-128, 127, 0, -1]), mm.word(False, 0, [1, 0, 0, 0])]
    s = st.Stimulus("t", "average", np.array(words, np.int64), np.ones(len(words), bool))
    as_s32 = lambda y: [((o >> (32 * c)) & 0xFFFFFFFF) for c in range(4) for o in [y]]  # y[0] in the LSBs
    out = mm.reference(s)
    assert len(out) == 2
    assert as_s32(out[0]) == [(-12) & 0xFFFFFFFF] * 4
    assert as_s32(out[1]) == [x & 0xFFFFFFFF for x in (-128, 127, 0, -1)]


def test_matmul4_unpack_roundtrip():
    from xpr.blocks import matmul4 as mm

    assert mm.unpack(mm.word(True, 2, [-128, 127, 0, -1])) == (True, 2, [-128, 127, 0, -1])
    # XLS flattens arrays element-0-LSB: x[0] = -128 is the low byte.
    assert mm.word(False, 0, [-128, 0, 0, 0]) == 0x80


def test_flip_bits_tracks_p():
    for p in (0.05, 0.25):
        s = mac.bitflip(4000, p, seed=0)
        n = len(s.words) + st.RESET_CYCLES
        t = st.to_trace(s, mac.IN_W, np.ones(n, bool), (st.RESET_CYCLES + 8, n - 1))
        # 16 of 17 bits flip with probability p; `last` adds a little.
        assert abs(input_alpha(t) - p * 16 / 17) < 0.02


@pytest.mark.parametrize("block", BLOCKS.values(), ids=BLOCKS.keys())
def test_suite_passes_gates_with_always_ready(block):
    for s in block.suite(200):
        n = len(s.words) + st.RESET_CYCLES
        t = st.to_trace(s, block.IN_W, np.ones(n, bool), (st.RESET_CYCLES + 8, n - 1))
        regime = "idle" if s.regime in ("idle", "leak") else s.regime
        v = check_trace(t, latency=8, regime=regime)
        assert v.passed, (s.name, v.failed)


# bf16 reference: (a, b, expected) in raw bits.
@pytest.mark.parametrize("a,b,want", [
    (0x3FC0, 0xC000, 0xC040),  # 1.5 * -2 = -3
    (0x3F80, 0x3F80, 0x3F80),  # 1 * 1
    (0x0080, 0x3F00, 0x0000),  # min normal * 0.5 -> subnormal -> flushed
    (0x0001, 0x7180, 0x0000),  # subnormal input treated as zero
    (0x8001, 0x3F80, 0x8000),  # negative subnormal -> -0
    (0x7F7F, 0x4000, 0x7F80),  # max normal * 2 -> inf
    (0x3F81, 0x3F81, 0x3F82),  # (1+2^-7)^2 = 1 + 2^-6 + 2^-14 -> rounds to 1+2^-6
    (0x3FFF, 0x3FFF, 0x407E),  # 1.9921875^2 = 3.968... -> RNE
    # Found on XLS RTL: (255/256)*2^-126 rounds up to the min normal instead of flushing.
    (0x00FF, 0x3F00, 0x0080),
    (0x2000, 0x1FFF, 0x0080),
    (0x0080, 0x3E80, 0x0000),  # min normal * 0.25: still subnormal after rounding -> 0
])
def test_bf16_reference(a, b, want):
    assert bf16_mul.mul(a, b) == want


def test_bf16_nan_rules():
    assert bf16_mul.is_nan(bf16_mul.mul(0x7F80, 0x0000))  # inf * 0
    assert bf16_mul.is_nan(bf16_mul.mul(0x7FC1, 0x3F80))  # NaN propagates
    assert bf16_mul.same(0x7FC0, 0xFFC1) and not bf16_mul.same(0x7F80, 0x7FC0)


def test_bf16_reference_matches_float32_for_normal_range():
    rng = np.random.default_rng(0)
    a = bf16_mul.from_float(rng.normal(0, 4, 2000))
    b = bf16_mul.from_float(rng.normal(0, 4, 2000))
    to_f = lambda x: (np.asarray(x, np.uint32) << 16).view(np.float32)
    # bf16 x bf16 is exact in float32, so one RNE rounding of the float32 product must agree.
    want = bf16_mul.from_float(to_f(a) * to_f(b))
    got = [bf16_mul.mul(int(x), int(y)) for x, y in zip(a, b)]
    assert got == list(want)
