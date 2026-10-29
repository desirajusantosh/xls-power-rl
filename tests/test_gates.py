import json

import numpy as np

from xpr.gates import X, RejectLog, Trace, check_trace, input_alpha

T, W, RST, LAT = 40, 8, 2, 3


def make(regime="average", seed=0):
    rng = np.random.default_rng(seed)
    rst = np.zeros(T, bool)
    rst[:RST] = True
    valid = np.ones(T, bool)
    ready = np.ones(T, bool)
    data = rng.integers(0, 2, (T, W), dtype=np.uint8)
    if regime == "idle":
        valid[:] = False
        data[:] = 0
    return Trace(rst, valid, ready, data, window=(RST + LAT, T - 1))


def test_legal_average_trace_passes():
    v = check_trace(make(), latency=LAT, regime="average")
    assert v.passed, v.failed


def test_legal_idle_trace_passes():
    v = check_trace(make("idle"), latency=LAT, regime="idle")
    assert v.passed, v.failed


def test_rejects_xz():
    t = make()
    t.data[10, 3] = X
    assert check_trace(t, latency=LAT, regime="worst").failed == ["no_xz"]


def test_rejects_reset_toggle_midstream():
    t = make()
    t.rst[20] = True
    assert "reset_prefix_only" in check_trace(t, latency=LAT, regime="worst").failed


def test_rejects_window_overlapping_fill():
    t = make()
    t.window = (RST + LAT - 1, T - 1)
    assert check_trace(t, latency=LAT, regime="average").failed == ["window_after_fill"]


def test_rejects_window_past_drain():
    t = make()
    t.valid[30:] = False
    assert "window_before_drain" in check_trace(t, latency=LAT, regime="average").failed


def test_rejects_valid_dropped_during_stall():
    t = make()
    t.ready[15] = False
    t.valid[16] = False
    assert "valid_held_until_ready" in check_trace(t, latency=LAT, regime="worst").failed


def test_rejects_data_change_during_stall():
    t = make()
    t.ready[15] = False
    t.data[16] = 1 - t.data[15]
    assert "data_stable_while_stalled" in check_trace(t, latency=LAT, regime="worst").failed


def test_stall_with_held_data_is_legal():
    t = make()
    t.ready[15] = False
    t.data[16] = t.data[15]
    assert check_trace(t, latency=LAT, regime="worst").passed


def test_rejects_zero_activity_outside_idle():
    t = make()
    t.data[:] = 1
    assert check_trace(t, latency=LAT, regime="average").failed == ["nonzero_activity"]


def test_idle_rejects_valid_high():
    t = make("idle")
    t.valid[20] = True
    assert "idle_valid_low" in check_trace(t, latency=LAT, regime="idle").failed


def test_input_alpha_max_hamming():
    t = make()
    t.data[::2] = 0
    t.data[1::2] = 1
    assert input_alpha(t) == 1.0


def test_reject_log_appends_jsonl(tmp_path):
    t = make()
    t.data[10, 0] = X
    v = check_trace(t, latency=LAT, regime="worst")
    log = RejectLog(tmp_path / "logs" / "rejects.jsonl")
    log.record(v, reward=3.5, provenance={"agent": "rl", "step": 7})
    log.record(v, reward=1.0, provenance={"agent": "ga", "step": 2})
    lines = (tmp_path / "logs" / "rejects.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["failed"] == ["no_xz"] and first["reward"] == 3.5
