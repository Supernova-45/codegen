from __future__ import annotations

import math

from posterior.particle_posterior import ParticlePosterior


def binary_entropy(p: float) -> float:
    p = max(1e-12, min(1.0 - 1e-12, p))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def eig_for_test(p_pass: float, epsilon: float) -> float:
    noisy_prob = epsilon + (1.0 - 2.0 * epsilon) * p_pass
    return binary_entropy(noisy_prob) - binary_entropy(epsilon)


def select_max_eig(
    tests: list[str],
    outcomes_by_test: list[list[bool]],
    posterior: ParticlePosterior,
    epsilon: float,
) -> tuple[int, list[float]]:
    scores: list[float] = []
    for outcomes in outcomes_by_test:
        p_pass = 0.0
        for w, out in zip(posterior.weights, outcomes, strict=True):
            p_pass += w * (1.0 if out else 0.0)
        scores.append(eig_for_test(p_pass, epsilon))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores
