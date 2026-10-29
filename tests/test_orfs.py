import pytest

from xpr.orfs import parse_power

LOG = """\
read_sdc /work/results/asap7/mac_p2/base/6_final.sdc
XPR_POWER stage=route activity=0.0 internal=1.851629e-04 switching=5.219383e-05 leakage=6.866727e-08 total=2.374255e-04
XPR_POWER stage=route activity=0.1 internal=3.333591e-04 switching=1.921590e-04 leakage=6.876651e-08 total=5.255868e-04
Group                  Internal  Switching    Leakage      Total
"""


def test_parses_only_tagged_lines():
    pts = parse_power(LOG)
    assert [p.activity for p in pts] == [0.0, 0.1]
    assert pts[0].stage == "route"
    assert pts[1].total_w == pytest.approx(5.255868e-04)


def test_components_sum_to_total():
    for p in parse_power(LOG):
        assert p.internal_w + p.switching_w + p.leakage_w == pytest.approx(p.total_w, rel=1e-5)


def test_empty_input():
    assert parse_power("no power here") == []
