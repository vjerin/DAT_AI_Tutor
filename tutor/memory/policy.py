"""Rules that decide whether a proposed memory operation is stored."""
from __future__ import annotations

import re

from ..schema import MemoryItem, MemoryOp, MemoryType

SENSITIVE = re.compile(
    r"\b(diagnos\w*|disorder|depress\w*|anxiety|adhd|autis\w*|medication|therapy|therapist|illness|disease|"
    r"pregnan\w*|disab\w*|religio\w*|church|mosque|temple|caste|ethnic\w*|race|politic\w*|vote|party member|"
    r"salary|income|debt|loan|bank account|credit card|sexual\w*|gay|lesbian|transgender|password|aadhaar|ssn)\b",
    re.I,
)


class Decision:
    COMMIT, PENDING, REJECT = "commit", "pending", "reject"


class MemoryPolicy:
    def __init__(self, cfg: dict):
        self.allowed = set(cfg.get("allowed_types", ["goal", "mastery", "misconception", "preference"]))
        self.min_conf = cfg.get("min_confidence", 0.6)
        self.min_obs = cfg.get("misconception_min_observations", 2)
        self.goal_explicit = cfg.get("goal_requires_explicit", True)
        self.ttl_days = cfg.get("ttl_days", {})

    def ttl_s(self, type_: MemoryType) -> float | None:
        d = self.ttl_days.get(type_.value)
        return d * 86400 if d else None

    def decide(self, op: MemoryOp, existing: MemoryItem | None) -> tuple[str, str]:
        if op.op in ("NOOP",):
            return Decision.REJECT, "noop"
        if op.op == "DELETE":
            return Decision.COMMIT, "learner request"
        if op.type is None or op.type.value not in self.allowed:
            return Decision.REJECT, "type not allowed"
        if SENSITIVE.search(f"{op.key} {op.value}"):
            return Decision.REJECT, "sensitive category"
        if op.type == MemoryType.MASTERY:
            return Decision.COMMIT, "mastery observation"
        if op.type == MemoryType.GOAL and self.goal_explicit and not op.explicit:
            return Decision.REJECT, "goals must be stated by the learner"
        if op.type == MemoryType.MISCONCEPTION:
            obs = (existing.observations + 1) if existing else 1
            if obs < self.min_obs:
                return Decision.PENDING, f"needs {self.min_obs} observations ({obs} so far)"
            return Decision.COMMIT, f"observed {obs} times"
        if op.confidence < self.min_conf and not op.explicit:
            return Decision.PENDING, "low confidence"
        return Decision.COMMIT, "explicit" if op.explicit else "confident"
