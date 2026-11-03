"""Fit P(alpha) = P_leak + P_intercept + slope * alpha from measured runs.

P_leak comes only from clock-stopped runs. P_intercept and slope come from an
OLS fit of (P_total - P_leak) against alpha over clocked runs. Both constant
terms cannot be separated by regression alone; see docs/decisions.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Run:
    """One power measurement.

    power_w: total power reported for the measurement window, in watts.
    alpha:   mean toggle rate per net per cycle in the window (ignored when not clocked).
    clocked: False for a clock-stopped (leakage-only) run.
    """

    power_w: float
    alpha: float
    clocked: bool = True
    label: str = ""


@dataclass(frozen=True)
class PowerModel:
    p_leak: float
    p_intercept: float
    slope: float
    r2: float
    n_leak: int
    n_dyn: int
    intercept_stderr: float = math.nan
    slope_stderr: float = math.nan
    residuals: tuple[float, ...] = field(default=(), repr=False)

    def predict(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return self.p_leak + self.p_intercept + self.slope * alpha


def fit(runs: Sequence[Run]) -> PowerModel:
    leak = [r for r in runs if not r.clocked]
    dyn = [r for r in runs if r.clocked]
    if not leak:
        raise ValueError("need at least one clock-stopped run to measure P_leak")
    if len(dyn) < 2:
        raise ValueError("need at least two clocked runs to fit intercept and slope")

    p_leak = float(np.mean([r.power_w for r in leak]))
    alpha = np.array([r.alpha for r in dyn], dtype=float)
    y = np.array([r.power_w for r in dyn], dtype=float) - p_leak
    if np.ptp(alpha) == 0.0:
        raise ValueError("clocked runs all have the same alpha; slope is unidentifiable")

    X = np.column_stack([np.ones_like(alpha), alpha])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    n = len(dyn)
    if n > 2:
        sigma2 = ss_res / (n - 2)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        intercept_se, slope_se = (float(math.sqrt(v)) for v in np.diag(cov))
    else:
        intercept_se = slope_se = math.nan

    return PowerModel(
        p_leak=p_leak,
        p_intercept=intercept,
        slope=slope,
        r2=r2,
        n_leak=len(leak),
        n_dyn=n,
        intercept_stderr=intercept_se,
        slope_stderr=slope_se,
        residuals=tuple(float(v) for v in resid),
    )
