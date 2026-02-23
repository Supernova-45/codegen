from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ParticlePosterior:
    candidates: list[str]
    weights: list[float]

    @classmethod
    def uniform(cls, candidates: Sequence[str]) -> "ParticlePosterior":
        if not candidates:
            raise ValueError("Need at least one candidate.")
        n = len(candidates)
        return cls(candidates=list(candidates), weights=[1.0 / n] * n)

    def map_index(self) -> int:
        return max(range(len(self.weights)), key=lambda i: self.weights[i])

    def map_confidence(self) -> float:
        return max(self.weights)

    def normalize(self) -> None:
        z = sum(self.weights)
        if z <= 0:
            n = len(self.weights)
            self.weights = [1.0 / n] * n
            return
        self.weights = [w / z for w in self.weights]

    def update(self, outcomes: list[bool], observed: bool, epsilon: float) -> None:
        new_weights: list[float] = []
        for w, outcome in zip(self.weights, outcomes, strict=True):
            p = (1.0 - epsilon) if outcome == observed else epsilon
            new_weights.append(w * p)
        self.weights = new_weights
        self.normalize()

    def expected_map_after_question(self, outcomes: list[bool], epsilon: float) -> float:
        p_obs_true = 0.0
        for w, outcome in zip(self.weights, outcomes, strict=True):
            p_obs_true += w * ((1 - epsilon) if outcome else epsilon)
        p_obs_false = 1.0 - p_obs_true

        post_true = self._posterior_given_observation(outcomes, True, epsilon)
        post_false = self._posterior_given_observation(outcomes, False, epsilon)

        return p_obs_true * max(post_true) + p_obs_false * max(post_false)

    def _posterior_given_observation(
        self, outcomes: list[bool], observed: bool, epsilon: float
    ) -> list[float]:
        ws: list[float] = []
        for w, outcome in zip(self.weights, outcomes, strict=True):
            likelihood = (1 - epsilon) if outcome == observed else epsilon
            ws.append(w * likelihood)
        z = sum(ws)
        if z <= 0:
            return [1.0 / len(ws)] * len(ws)
        return [x / z for x in ws]
