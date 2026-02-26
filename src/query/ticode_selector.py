from __future__ import annotations

from posterior.particle_posterior import TestOutcome


def ticode_scores(outcomes: list[list[TestOutcome]], candidates: list[int]) -> list[float]:
    scores = []
    for o in outcomes:
        trues = 0
        falses = 0
        for idx in candidates:
            outcome = o[idx]
            if outcome:
                trues += 1
            elif outcome is False:
                falses += 1
        if trues == 0 or falses == 0:
            scores.append(0)
        else:
            scores.append(min(trues, falses) / max(trues, falses))
    return scores