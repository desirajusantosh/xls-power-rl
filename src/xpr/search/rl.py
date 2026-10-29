"""RL policy for worst-case activity: autoregressive GRU over input vectors, REINFORCE.

Episode: the policy emits the P vectors of a pattern one at a time.
  state  = previous vector (W bits), its Hamming distance to the one before
           (an input toggle statistic), and the position in the pattern; the
           GRU carries the rest of the history
  action = next vector, one Bernoulli per bit of {rst, vld, data}
  reward = fail-closed proxy score of the finished pattern (0 if any gate fails)
Update: REINFORCE with a batch-normalized advantage and an entropy bonus.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .space import Scorer


class Policy(nn.Module):
    def __init__(self, width: int, period: int, hidden: int = 128):
        super().__init__()
        self.width, self.period = width, period
        self.cell = nn.GRUCell(width + 1 + period, hidden)
        self.head = nn.Linear(hidden, width)
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def sample(self, batch: int, gen: torch.Generator):
        """Returns patterns (B, P, W) bool, summed log-probs (B,), summed entropies (B,)."""
        W, P = self.width, self.period
        h = self.h0.expand(batch, -1)
        prev = torch.zeros(batch, W)
        prev2 = torch.zeros(batch, W)
        logp = torch.zeros(batch)
        ent = torch.zeros(batch)
        out = []
        for t in range(P):
            ham = (prev != prev2).float().mean(dim=1, keepdim=True)
            pos = torch.zeros(batch, P)
            pos[:, t] = 1.0
            h = self.cell(torch.cat([prev, ham, pos], dim=1), h)
            dist = torch.distributions.Bernoulli(logits=self.head(h))
            x = torch.bernoulli(dist.probs.detach(), generator=gen)
            logp = logp + dist.log_prob(x).sum(dim=1)
            ent = ent + dist.entropy().sum(dim=1)
            out.append(x)
            prev2, prev = prev, x
        return torch.stack(out, dim=1).bool(), logp, ent


def reinforce(scorer: Scorer, rng: np.random.Generator, batch: int = 64, lr: float = 3e-3,
              entropy_coef: float = 0.01, hidden: int = 128) -> list[dict]:
    """Train until the scorer's budget is spent. Returns per-update stats."""
    torch.set_num_threads(1)
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    policy = Policy(scorer.space.width, scorer.space.period, hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    stats = []
    while scorer.remaining > 0:
        n = min(batch, scorer.remaining)
        patterns, logp, ent = policy.sample(n, gen)
        r = torch.as_tensor(scorer.score(patterns.numpy(), tag="rl"), dtype=torch.float32)
        adv = (r - r.mean()) / (r.std() + 1e-8) if n > 1 else r - r.mean()
        loss = -(adv * logp).mean() - entropy_coef * ent.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        stats.append({"evals": scorer.used, "mean_reward": float(r.mean()), "max_reward": float(r.max()),
                      "legal_frac": float((r > 0).float().mean()), "entropy": float(ent.detach().mean()),
                      "best": scorer.best})
    return stats
