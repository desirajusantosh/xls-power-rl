import math

import numpy as np
import pytest

from xpr.powerfit import Run, fit


def test_recovers_exact_coefficients():
    leak, icpt, slope = 1e-6, 2e-4, 5e-3
    runs = [Run(leak, 0.0, clocked=False)]
    runs += [Run(leak + icpt + slope * a, a) for a in (0.0, 0.1, 0.25, 0.4)]
    m = fit(runs)
    assert m.p_leak == pytest.approx(leak)
    assert m.p_intercept == pytest.approx(icpt)
    assert m.slope == pytest.approx(slope)
    assert m.r2 == pytest.approx(1.0)
    assert m.predict(0.3) == pytest.approx(leak + icpt + slope * 0.3)


def test_noisy_fit_reports_stderr():
    rng = np.random.default_rng(0)
    alphas = np.linspace(0.0, 0.5, 20)
    runs = [Run(1e-6, 0.0, clocked=False)]
    runs += [Run(1e-6 + 2e-4 + 5e-3 * a + rng.normal(0, 1e-6), a) for a in alphas]
    m = fit(runs)
    assert abs(m.slope - 5e-3) < 4 * m.slope_stderr
    assert m.n_dyn == 20 and m.n_leak == 1
    assert len(m.residuals) == 20


def test_two_points_have_no_stderr():
    m = fit([Run(0.0, 0.0, clocked=False), Run(1.0, 0.0), Run(2.0, 1.0)])
    assert math.isnan(m.slope_stderr)


def test_requires_leakage_run():
    with pytest.raises(ValueError, match="clock-stopped"):
        fit([Run(1.0, 0.0), Run(2.0, 1.0)])


def test_requires_distinct_alpha():
    with pytest.raises(ValueError, match="unidentifiable"):
        fit([Run(0.0, 0.0, clocked=False), Run(1.0, 0.2), Run(1.1, 0.2)])
