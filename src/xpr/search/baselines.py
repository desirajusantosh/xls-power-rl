"""Baselines: random search (raw and legal-only), a genetic algorithm, and CEM."""

from __future__ import annotations

import numpy as np

from .space import Scorer

BATCH = 64


def random_search(scorer: Scorer, rng: np.random.Generator) -> None:
    """Uniform bits over the whole input port, rst and valid included."""
    P, W = scorer.space.period, scorer.space.width
    while scorer.remaining > 0:
        scorer.score(rng.random((BATCH, P, W)) < 0.5, tag="random")


def random_legal(scorer: Scorer, rng: np.random.Generator) -> None:
    """Uniform data bits with rst=0 and valid=1: random search given the protocol."""
    P, W = scorer.space.period, scorer.space.width
    while scorer.remaining > 0:
        x = rng.random((BATCH, P, W)) < 0.5
        x[:, :, 0], x[:, :, 1] = False, True
        scorer.score(x, tag="random_legal")


def genetic(scorer: Scorer, rng: np.random.Generator, pop: int = 64, elite: int = 2,
            tournament: int = 3, crossover: float = 0.9) -> None:
    """Steady generational GA over (P, W) patterns.

    Selection: tournament. Crossover: uniform over whole vectors (keeps each
    cycle's word intact). Mutation: independent bit flips at rate 1/(P*W) * 2.
    """
    P, W = scorer.space.period, scorer.space.width
    mut = 2.0 / (P * W)
    x = rng.random((pop, P, W)) < 0.5
    fit = scorer.score(x, tag="ga_init")
    x = x[: len(fit)]
    while scorer.remaining > 0:
        order = np.argsort(-fit)
        children = [x[i].copy() for i in order[:elite]]
        while len(children) < pop:
            a = x[max(rng.integers(0, len(x), tournament), key=lambda i: fit[i])]
            b = x[max(rng.integers(0, len(x), tournament), key=lambda i: fit[i])]
            child = np.where((rng.random(P) < 0.5)[:, None], a, b) if rng.random() < crossover else a.copy()
            child ^= rng.random((P, W)) < mut
            children.append(child)
        new = np.array(children[elite:])
        new_fit = scorer.score(new, tag="ga")
        x = np.concatenate([x[order[:elite]], new[: len(new_fit)]])
        fit = np.concatenate([fit[order[:elite]], new_fit])


def cem(scorer: Scorer, rng: np.random.Generator, batch: int = 64, elite_frac: float = 0.125,
        smoothing: float = 0.3, floor: float = 0.02) -> None:
    """Cross-entropy method over independent per-bit Bernoullis, one per (vector, bit).

    The ablation for RL: same batch size and "learn a sampling distribution" idea,
    but with no conditioning between vectors and no hidden state. Each step moves the
    probabilities toward the elite fraction's bit frequencies; `floor` keeps every
    bit explorable (probabilities stay within [floor, 1 - floor]).

    Elites come only from legal candidates (score > 0). A batch with none leaves the
    distribution unchanged: rejected candidates carry no information, and ranking
    them would drag the distribution into the illegal region (observed: rst -> 1).
    """
    P, W = scorer.space.period, scorer.space.width
    prob = np.full((P, W), 0.5)
    n_elite = max(1, int(batch * elite_frac))
    while scorer.remaining > 0:
        x = rng.random((batch, P, W)) < prob
        s = scorer.score(x, tag="cem")
        x = x[: len(s)]
        legal = np.flatnonzero(s > 0)
        if len(legal) == 0:
            continue
        elite = x[legal[np.argsort(-s[legal])[:n_elite]]]
        prob = (1 - smoothing) * prob + smoothing * elite.mean(axis=0)
        prob = np.clip(prob, floor, 1 - floor)
