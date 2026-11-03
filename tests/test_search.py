from dataclasses import dataclass

import numpy as np
import pytest

from xpr import stimulus as st
from xpr.blocks import bf16_mul, mac
from xpr.gatesim import FILL_CYCLES
from xpr.proxy import ProxyResult
from xpr.search import baselines
from xpr.search.space import Scorer, Space


class FakeProxy:
    """Alpha = fraction of data bits that differ between consecutive pattern vectors."""

    def __init__(self, in_w):
        self.in_w = in_w

    def evaluate(self, sequences, win_start):
        out = []
        for s in sequences:
            data = np.asarray(s[win_start:]) & ((1 << self.in_w) - 1)
            flips = sum(bin(int(a) ^ int(b)).count("1") for a, b in zip(data[:-1], data[1:]))
            a = flips / (self.in_w * (len(data) - 1))
            out.append(ProxyResult(a, flips, self.in_w, len(data) - 1, 0, 0))
        return out


def legal(P, W, rng):
    x = rng.random((P, W)) < 0.5
    x[:, 0], x[:, 1] = False, True
    return x


def test_words_layout_and_reset_prefix():
    sp = Space(mac, period=2, window=8)
    pat = np.zeros((2, sp.width), bool)
    pat[:, 1] = True  # valid
    pat[1, -1] = True  # data bit 0 in the second vector
    w = sp.words(pat)
    assert len(w) == st.RESET_CYCLES + FILL_CYCLES + 8
    assert all(x == 1 << (mac.IN_W + 1) for x in w[: st.RESET_CYCLES])
    assert list(w[st.RESET_CYCLES : st.RESET_CYCLES + 2]) == [1 << mac.IN_W, (1 << mac.IN_W) | 1]


def test_rst_in_pattern_is_rejected_and_logged(tmp_path):
    from xpr.gates import RejectLog

    sp = Space(mac, period=4)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=10, reject_log=RejectLog(tmp_path / "r.jsonl"))
    rng = np.random.default_rng(0)
    good, bad = legal(4, sp.width, rng), legal(4, sp.width, rng)
    bad[2, 0] = True
    s = sc.score(np.stack([good, bad]))
    assert s[0] > 0 and s[1] == 0
    assert sc.rejects["reset_prefix_only"] == 1
    assert "reset_prefix_only" in (tmp_path / "r.jsonl").read_text()


def test_valid_never_high_is_rejected():
    sp = Space(mac, period=4)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=10)
    x = legal(4, sp.width, np.random.default_rng(1))
    x[:, 1] = False
    assert sc.score(x[None])[0] == 0 and sc.rejects["inputs_accepted"] == 1


def test_finite_only_rejects_inf_nan_operands():
    sp = Space(bf16_mul, period=2, finite_only=True)
    sc = Scorer(sp, FakeProxy(bf16_mul.IN_W), budget=10)
    x = legal(2, sp.width, np.random.default_rng(2))
    x[:, 2 + 1 : 2 + 9] = False  # a exponent 0
    x[:, 2 + 17 : 2 + 25] = False  # b exponent 0
    x[1, 2 + 1 : 2 + 9] = True  # a exponent all ones in vector 1
    assert sc.score(x[None])[0] == 0 and sc.rejects["finite_operands"] == 1


def test_budget_counts_duplicates_and_stops():
    sp = Space(mac, period=2)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=5)
    x = legal(2, sp.width, np.random.default_rng(3))
    s = sc.score(np.stack([x] * 8))
    assert len(s) == 5 and sc.used == 5 and sc.remaining == 0
    assert len(sc.top) == 1


@pytest.mark.parametrize("method", [baselines.random_search, baselines.random_legal, baselines.genetic,
                                    baselines.cem])
def test_methods_respect_budget_and_improve(method):
    sp = Space(mac, period=4, window=32)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=640)
    method(sc, np.random.default_rng(0))
    assert sc.used == 640
    assert sc.best > 0
    assert sc.curve[-1][1] == sc.best


def test_ga_beats_random_legal_on_toy_objective():
    sp = Space(mac, period=4, window=32)
    res = {}
    for name, m in [("rl", baselines.random_legal), ("ga", baselines.genetic)]:
        sc = Scorer(sp, FakeProxy(mac.IN_W), budget=3000)
        m(sc, np.random.default_rng(0))
        res[name] = sc.best
    assert res["ga"] > res["rl"]


def test_harmonic_amplitude_of_square_and_flat_waves():
    from xpr.search.objectives import harmonic_amplitude

    sq = np.tile([1.0] * 4 + [0.0] * 4, 32)
    # 0/1 square wave, period 8, 50% duty: amplitude = (2/8) * |sum_{k<4} e^{-2 pi i k/8}| ~= 0.653
    assert harmonic_amplitude(sq, 8) == pytest.approx(0.25 * abs(sum(np.exp(-2j * np.pi * k / 8) for k in range(4))))
    assert harmonic_amplitude(np.full(256, 0.7), 8) == pytest.approx(0.0, abs=1e-12)
    # A period-2 alternation has no period-8 component.
    assert harmonic_amplitude(np.tile([1.0, 0.0], 128), 8) == pytest.approx(0.0, abs=1e-12)


def test_swing_objective_scores_scorer_candidates():
    from xpr.search.objectives import swing

    class WaveProxy(FakeProxy):
        def evaluate(self, sequences, win_start):
            base = super().evaluate(sequences, win_start)
            out = []
            for s, r in zip(sequences, base):
                data = np.asarray(s[win_start:]) & ((1 << self.in_w) - 1)
                cyc = [bin(int(a) ^ int(b)).count("1") for a, b in zip(data[:-1], data[1:])]
                out.append(ProxyResult(r.alpha, r.toggles, r.bits, r.window_cycles, 0, 0, tuple(cyc)))
            return out

    sp = Space(mac, period=8, window=64)
    sc = Scorer(sp, WaveProxy(mac.IN_W), budget=10, objective=swing(8))
    rng = np.random.default_rng(5)
    bursty = legal(8, sp.width, rng)
    bursty[4:, 2:] = bursty[3, 2:]  # data frozen for the second half of the period
    steady = legal(8, sp.width, rng)
    steady[:, 2:] = np.tile(np.array([[0], [1]] * 4, bool), (1, sp.width - 2))  # flips every cycle
    s = sc.score(np.stack([bursty, steady]))
    assert s[0] > s[1]


def test_cem_learns_to_avoid_rst_and_beats_random_legal():
    sp = Space(mac, period=4, window=32)
    res = {}
    for name, m in [("rl", baselines.random_legal), ("cem", baselines.cem)]:
        sc = Scorer(sp, FakeProxy(mac.IN_W), budget=3000)
        m(sc, np.random.default_rng(0))
        res[name] = sc
    assert res["cem"].best > res["rl"].best
    # CEM learns to stop driving rst: well under half its budget is rejected
    # (uniform sampling over a 4-vector pattern would reject about 94%).
    assert res["cem"].rejects["reset_prefix_only"] < 1500


def test_cem_ignores_all_rejected_batches():
    # A wide port makes legal candidates rare at first (1 in 2^P without rst);
    # CEM must not drift toward rst while every candidate is rejected.
    sp = Space(mac, period=8, window=32)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=4000)
    baselines.cem(sc, np.random.default_rng(3))
    assert sc.best > 0
    assert sc.rejects["reset_prefix_only"] < 4000


def test_to_stimulus_refuses_rst_patterns():
    sp = Space(mac, period=2)
    x = legal(2, sp.width, np.random.default_rng(4))
    s = sp.to_stimulus(x, "p", n=10)
    assert s.valid.all() and len(s.words) == 10
    x[0, 0] = True
    with pytest.raises(ValueError):
        sp.to_stimulus(x, "p", n=10)
