"""Rerankers: lexical (default), cross-encoder, or none."""
from __future__ import annotations

import math

from ..schema import RetrievedItem
from ..text import light_stem, numbers, query_terms, tokenize


class NoReranker:
    def rerank(self, query: str, items: list[RetrievedItem]) -> list[RetrievedItem]:
        if items:
            top = max(i.fused_score for i in items) or 1.0
            for it in items:
                it.rerank_score = it.fused_score / top * 0.5
        return items


class LexicalReranker:
    def __init__(self, idf=None):
        self.idf = idf or (lambda t: 1.0)

    def score(self, query: str, text: str) -> float:
        q = query_terms(query) or [light_stem(t) for t in tokenize(query)]
        if not q:
            return 0.0
        d_toks = [light_stem(t) for t in tokenize(text)]
        d = set(d_toks)
        weights = {t: self.idf(t) for t in set(q)}
        total = sum(weights.values()) or 1.0
        covered = sum(w for t, w in weights.items() if t in d) / total
        # phrase bonus: query bigrams appearing contiguously in the document
        q_all = [light_stem(t) for t in tokenize(query)]
        q_bi = set(zip(q_all, q_all[1:]))
        d_bi = set(zip(d_toks, d_toks[1:]))
        phrase = (len(q_bi & d_bi) / len(q_bi)) if q_bi else 0.0
        nums_q, nums_d = numbers(query), numbers(text)
        num = (len(nums_q & nums_d) / len(nums_q)) if nums_q else 0.0
        length_pen = 1.0 / (1.0 + math.log1p(max(0, len(d_toks) - 120) / 120))
        return float(min(1.0, (0.75 * covered + 0.2 * phrase + 0.05 * num) * (0.85 + 0.15 * length_pen)))

    def rerank(self, query, items):
        for it in items:
            it.rerank_score = self.score(query, it.unit.search_text())
        return sorted(items, key=lambda i: -(i.rerank_score or 0))


class CrossEncoderReranker:  # pragma: no cover - needs model download
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query, items):
        if not items:
            return items
        logits = self.model.predict([(query, it.unit.search_text()) for it in items])
        for it, z in zip(items, logits):
            it.rerank_score = float(1 / (1 + math.exp(-float(z))))
        return sorted(items, key=lambda i: -(i.rerank_score or 0))


def make_reranker(cfg, bm25=None):
    kind = cfg.get("retrieval.reranker", "lexical")
    if kind == "none":
        return NoReranker()
    if kind == "lexical":
        return LexicalReranker(bm25.idf if bm25 else None)
    if kind == "cross_encoder":
        return CrossEncoderReranker(cfg.get("retrieval.reranker_model"))
    raise ValueError(f"unknown reranker {kind!r}")
