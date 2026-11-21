"""Bayesian Knowledge Tracing: per-concept mastery from correct/incorrect signals."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BKT:
    p_init: float = 0.2
    p_learn: float = 0.15
    p_slip: float = 0.1
    p_guess: float = 0.2

    def update(self, p_known: float | None, correct: bool) -> float:
        p = self.p_init if p_known is None else p_known
        if correct:
            num = p * (1 - self.p_slip)
            den = num + (1 - p) * self.p_guess
        else:
            num = p * self.p_slip
            den = num + (1 - p) * (1 - self.p_guess)
        posterior = num / den if den else p
        return posterior + (1 - posterior) * self.p_learn

    def decay(self, p_known: float, days_idle: float, half_life_days: float = 60.0) -> float:
        """Forgetting: drift back towards p_init without practice."""
        w = 0.5 ** (days_idle / half_life_days)
        return self.p_init + (p_known - self.p_init) * w
