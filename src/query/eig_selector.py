from __future__ import annotations

import math
from typing import Any

from posterior.particle_posterior import ParticlePosterior


def binary_entropy(p: float) -> float:
    p = max(1e-12, min(1.0 - 1e-12, p))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def eig_for_test(p_pass: float, epsilon: float) -> float:
    noisy_prob = epsilon + (1.0 - 2.0 * epsilon) * p_pass
    return binary_entropy(noisy_prob) - binary_entropy(epsilon)


def select_max_eig(
    tests: list[str],
    outcomes_by_test: list[list[bool | None]],
    posterior: ParticlePosterior,
    epsilon: float,
    undefined_likelihood: float = 1.0,
    discriminative_weight: float = 0.0,
    runtime_error_penalty: float = 0.0,
) -> tuple[int, list[float], list[dict[str, Any]]]:
    scores: list[float] = []
    details: list[dict[str, Any]] = []
    for outcomes in outcomes_by_test:
        p_pass = _weighted_defined_pass_probability(posterior, outcomes)
        base_eig = eig_for_test(p_pass, epsilon)
        defined_weight = _defined_weight(posterior, outcomes)
        undefined_weight = max(0.0, 1.0 - defined_weight)
        balance = 4.0 * p_pass * (1.0 - p_pass)
        quality = (1.0 - discriminative_weight) + (discriminative_weight * balance)
        robustness = 1.0 - (runtime_error_penalty * undefined_weight)
        robustness = max(0.0, robustness)
        score = base_eig * defined_weight * quality * robustness
        scores.append(score)
        details.append(
            {
                "base_eig": base_eig,
                "defined_weight": defined_weight,
                "undefined_weight": undefined_weight,
                "balance": balance,
                "quality_multiplier": quality,
                "robustness_multiplier": robustness,
                "composite_score": score,
                "undefined_likelihood": undefined_likelihood,
            }
        )
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, scores, details


def _weighted_defined_pass_probability(
    posterior: ParticlePosterior,
    outcomes: list[bool | None],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for w, out in zip(posterior.weights, outcomes, strict=True):
        if out is None:
            continue
        denominator += w
        if out:
            numerator += w
    if denominator <= 0.0:
        return 0.5
    return numerator / denominator


def _defined_weight(
    posterior: ParticlePosterior,
    outcomes: list[bool | None],
) -> float:
    weight = 0.0
    for w, out in zip(posterior.weights, outcomes, strict=True):
        if out is not None:
            weight += w
    return max(0.0, min(1.0, weight))
