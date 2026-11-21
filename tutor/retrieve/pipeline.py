"""Hybrid retrieval: parallel channels -> (weighted) RRF -> rerank -> pack (M2)."""
from __future__ import annotations

import time
from typing import Optional

from ..config import Config
from ..index.manager import IndexManager
from ..schema import EvidenceUnit, RetrievedItem
from ..store import EvidenceStore
from .fuse import rrf
from .rerank import make_reranker


class Retriever:
    def __init__(self, cfg: Config, store: EvidenceStore, index: IndexManager):
        self.cfg, self.store, self.index = cfg, store, index
        self.reranker = make_reranker(cfg, index.bm25)
        self._units: dict[str, EvidenceUnit] = {}
        self.refresh()

    def refresh(self) -> None:
        """Reload evidence metadata (call after ingestion or version changes)."""
        self._units = {u.evidence_id: u for u in self.store.all_units()}

    def allowed_ids(self) -> set[str]:
        c = self.cfg
        mods = c.get("retrieval.modalities")
        out = set()
        for u in self._units.values():
            if not c.get("retrieval.include_superseded", False) and not u.is_current:
                continue
            if c.get("retrieval.drop_flagged", True) and "suspected_injection" in u.flags:
                continue
            if mods and u.modality.value not in mods:
                continue
            out.add(u.evidence_id)
        return out

    def retrieve(self, query: str, query_class: str = "conceptual",
                 allowed: Optional[set[str]] = None) -> tuple[list[RetrievedItem], dict]:
        c = self.cfg
        t0 = time.perf_counter()
        allowed = self.allowed_ids() if allowed is None else allowed
        channels = list(c.get("retrieval.channels", ["sparse", "dense", "visual"]))
        k = {"sparse": c.get("retrieval.k_sparse", 50), "dense": c.get("retrieval.k_dense", 50),
             "visual": c.get("retrieval.k_visual", 20)}
        results = self.index.search(query, allowed, channels, k)
        t1 = time.perf_counter()
        weights = None
        if c.get("retrieval.weighted_rrf", True):
            weights = (c.get("retrieval.class_weights") or {}).get(query_class)
        fused = rrf(results, k=c.get("retrieval.rrf_k", 60), weights=weights)
        top_n = c.get("retrieval.rerank_top_n", 30)
        cands = [RetrievedItem(unit=self._units[d], fused_score=s, channel_ranks=r, channel_scores=sc)
                 for d, s, r, sc in fused[:top_n] if d in self._units]
        cands = self.reranker.rerank(query, cands)
        t2 = time.perf_counter()
        final = self.pack(cands, query_class, allowed)
        info = {
            "channels": {ch: [d for d, _ in res[:10]] for ch, res in results.items()},
            "candidates": [{"id": it.unit.evidence_id, "fused": round(it.fused_score, 5),
                            "rerank": None if it.rerank_score is None else round(it.rerank_score, 4),
                            "ranks": it.channel_ranks} for it in cands[:15]],
            "n_allowed": len(allowed),
            "timings_ms": {"retrieve": (t1 - t0) * 1000, "fuse_rerank": (t2 - t1) * 1000},
            "all_ranked": [it.unit.evidence_id for it in cands],
        }
        return final, info

    def pack(self, cands: list[RetrievedItem], query_class: str, allowed: Optional[set[str]] = None) -> list[RetrievedItem]:
        """Keep the top ``final_k`` with light diversity, and attach the frame that
        was on screen for top transcript hits on visual questions."""
        k = self.cfg.get("retrieval.final_k", 6)
        out: list[RetrievedItem] = []
        seen_text: set[str] = set()
        per_slide: dict[tuple, int] = {}
        for it in cands:
            u = it.unit
            key = u.text.strip().lower()[:200]
            if key in seen_text:
                continue
            sk = (u.source_id, u.source_version, u.slide_no)
            if per_slide.get(sk, 0) >= 2:
                continue
            seen_text.add(key)
            per_slide[sk] = per_slide.get(sk, 0) + 1
            out.append(it)
            if len(out) >= k:
                break
        if query_class == "visual" and out and not any(i.unit.is_visual for i in out[:3]):
            mods = self.cfg.get("retrieval.modalities")
            if not mods or "slide" in mods or "code" in mods:
                for it in out[:2]:
                    for nb in self.store.neighbours(it.unit):
                        if nb.is_visual and nb.is_current and (allowed is None or nb.evidence_id in allowed) and nb.evidence_id not in {o.unit.evidence_id for o in out}:
                            out.insert(1, RetrievedItem(unit=nb, fused_score=it.fused_score * 0.9,
                                                        rerank_score=(it.rerank_score or 0) * 0.9,
                                                        channel_ranks={"companion": 1}))
                            break
                out = out[:k]
        return out
