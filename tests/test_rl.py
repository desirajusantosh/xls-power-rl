import numpy as np
import pytest

torch = pytest.importorskip("torch")  # RL is an optional extra ([rl]); CI installs [dev] only

from xpr.blocks import mac
from xpr.search.rl import JointPolicy, Policy, reinforce
from xpr.search.space import Scorer, Space

from test_search import FakeProxy


@pytest.mark.parametrize("cls", [Policy, JointPolicy])
def test_policy_sample_shapes_and_logprob(cls):
    pol = cls(width=6, period=3, hidden=16)
    x, logp, ent = pol.sample(5, torch.Generator().manual_seed(0))
    assert x.shape == (5, 3, 6) and x.dtype == torch.bool
    assert logp.shape == ent.shape == (5,)
    assert torch.all(logp <= 0) and torch.all(ent >= 0)
    logp.mean().backward()  # differentiable


@pytest.mark.parametrize("joint", [False, True])
def test_reinforce_respects_budget_and_learns_rst(joint):
    sp = Space(mac, period=2, window=16)
    sc = Scorer(sp, FakeProxy(mac.IN_W), budget=640)
    reinforce(sc, np.random.default_rng(0), batch=32, hidden=16, joint_bits=joint)
    assert sc.used == 640 and sc.best > 0
