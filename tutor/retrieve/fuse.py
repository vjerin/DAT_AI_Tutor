"""Reciprocal Rank Fusion (Cormack et al., SIGIR 2009), optionally weighted."""
from __future__ import annotations

from typing import Optional


def rrf(results: dict[str, list[tuple[str, float]]], k: int = 60,
        weights: Optional[dict[str, float]] = None) -> list[tuple[str, float, dict[str, int], dict[str, float]]]:
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    for channel, ranked in results.items():
        w = (weights or {}).get(channel, 1.0)
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
            ranks.setdefault(doc_id, {})[channel] = rank
            scores.setdefault(doc_id, {})[channel] = score
    order = sorted(fused.items(), key=lambda x: -x[1])
    return [(d, s, ranks[d], scores[d]) for d, s in order]
