"""Search objectives over a proxy result.

alpha:  mean toggle rate (average power).
swing:  amplitude of the per-cycle activity waveform at a target period, in
        toggles per bit per cycle. A periodic current swing at the power-delivery
        network's resonance is what a di/dt stressmark aims for; the target period
        stands in for that resonance (8 cycles = 250 MHz at 2 GHz).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..proxy import ProxyResult

Objective = Callable[[ProxyResult], float]


def alpha(r: ProxyResult) -> float:
    return r.alpha


def waveform(r: ProxyResult) -> np.ndarray:
    """Per-cycle toggles per bit."""
    if r.per_cycle is None:
        raise ValueError("swing objectives need a proxy built with per_cycle=True")
    return np.asarray(r.per_cycle, float) / r.bits


def harmonic_amplitude(x: np.ndarray, period: int) -> float:
    """Amplitude of the sinusoid at `period` cycles fitted to x (whole periods only)."""
    n = len(x) - len(x) % period
    x = x[:n]
    phase = np.exp(-2j * np.pi * np.arange(n) / period)
    return float(2 * abs(np.dot(x, phase)) / n)


def swing(period: int = 8) -> Objective:
    def objective(r: ProxyResult) -> float:
        return harmonic_amplitude(waveform(r), period)

    objective.__name__ = f"swing_p{period}"
    return objective


def peak_to_peak(r: ProxyResult, period: int) -> float:
    """Max minus min of the period-averaged waveform (reporting only)."""
    x = waveform(r)
    n = len(x) - len(x) % period
    phases = x[:n].reshape(-1, period).mean(axis=0)
    return float(phases.max() - phases.min())
