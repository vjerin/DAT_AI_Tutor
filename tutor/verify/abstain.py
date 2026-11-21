"""Answer / qualify / abstain decision from retrieval and verification features."""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from ..schema import AnswerSegment, RetrievedItem, Status

FEATURES = ["top_rerank", "margin", "top_fused", "agreement", "frac_supported", "mean_support",
            "n_cited_units", "n_segments", "gen_abstained", "visual_query", "top_is_visual",
            "coverage", "oov_frac", "answer_coverage"]


def coverage_features(query: str, cited_texts: list[str], answer: str, idf, vocab) -> dict[str, float]:
    """How much of the question is actually present in the cited evidence / corpus.

    * coverage        : IDF-weighted share of question terms found in the cited evidence
    * oov_frac        : share of question terms that never occur anywhere in the corpus
                        (strong off-corpus signal: "Adam", "Kubernetes", "batch size")
    * answer_coverage : IDF-weighted share of question terms found in the answer itself
    """
    from ..text import content_terms, query_terms
    q = query_terms(query)
    if not q:
        return {"coverage": 0.0, "oov_frac": 0.0, "answer_coverage": 0.0}
    w = {t: (idf(t) if idf else 1.0) for t in q}
    tot = sum(w.values()) or 1.0
    ev = content_terms(" ".join(cited_texts))
    ans = content_terms(answer)
    oov = [t for t in q if vocab is not None and t not in vocab]
    return {"coverage": sum(v for t, v in w.items() if t in ev) / tot,
            "oov_frac": len(oov) / len(q),
            "answer_coverage": sum(v for t, v in w.items() if t in ans) / tot}


def extract_features(items: list[RetrievedItem], segments: list[AnswerSegment], gen_status: Status,
                     query_class: str) -> dict[str, float]:
    rr = [i.rerank_score or 0.0 for i in items]
    top = rr[0] if rr else 0.0
    fifth = rr[4] if len(rr) > 4 else (rr[-1] if rr else 0.0)
    agreement = 0.0
    if items:
        ranks = items[0].channel_ranks
        agreement = sum(1 for r in ranks.values() if r <= 10) / 3.0
    supp = [s.support_score for s in segments if s.support_score is not None]
    cited = {c for s in segments for c in s.citations}
    return {
        "top_rerank": top,
        "margin": top - fifth,
        "top_fused": (items[0].fused_score * 60 / 3) if items else 0.0,   # ~0..1 for 3 channels
        "agreement": min(1.0, agreement),
        "frac_supported": (sum(1 for s in segments if s.supported) / len(segments)) if segments else 0.0,
        "mean_support": float(np.mean(supp)) if supp else 0.0,
        "n_cited_units": float(len(cited)),
        "n_segments": float(len(segments)),
        "gen_abstained": 1.0 if gen_status == Status.ABSTAINED else 0.0,
        "visual_query": 1.0 if query_class == "visual" else 0.0,
        "top_is_visual": 1.0 if items and items[0].unit.is_visual else 0.0,
        "coverage": 0.0, "oov_frac": 0.0, "answer_coverage": 0.0,
    }


class HeuristicAbstainer:
    def __init__(self, answer_threshold: float = 0.55, qualify_threshold: float = 0.35):
        self.t_answer, self.t_qualify = answer_threshold, qualify_threshold

    def confidence(self, f: dict[str, float]) -> float:
        if f["gen_abstained"] or f["n_segments"] == 0:
            return 0.0
        z = (0.55 * f["coverage"] + 0.15 * f["answer_coverage"] + 0.15 * f["top_rerank"]
             + 0.10 * f["frac_supported"] + 0.05 * f["agreement"])
        z *= (1.0 - f["oov_frac"]) ** 1.5          # unknown terms: the question is probably off-corpus
        return float(max(0.0, min(1.0, z)))

    def decide(self, f: dict[str, float]) -> tuple[Status, float]:
        p = self.confidence(f)
        if p >= self.t_answer:
            return Status.ANSWERED, p
        if p >= self.t_qualify:
            return Status.QUALIFIED, p
        return Status.ABSTAINED, p


class CalibratedAbstainer(HeuristicAbstainer):
    def __init__(self, model, answer_threshold: float, qualify_threshold: float):
        super().__init__(answer_threshold, qualify_threshold)
        self.model = model

    def confidence(self, f):
        if f["gen_abstained"] or f["n_segments"] == 0:
            return 0.0
        x = np.array([[f[k] for k in FEATURES]])
        return float(self.model.predict_proba(x)[0, 1])

    @classmethod
    def load(cls, path: str | Path, answer_threshold=None, qualify_threshold=None) -> "CalibratedAbstainer":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["model"], answer_threshold or d["answer_threshold"], qualify_threshold or d["qualify_threshold"])


def make_abstainer(cfg):
    a = cfg.section("abstention")
    if a.get("type") == "calibrated" and a.get("path") and Path(a["path"]).exists():
        return CalibratedAbstainer.load(a["path"])
    return HeuristicAbstainer(a.get("answer_threshold", 0.55), a.get("qualify_threshold", 0.35))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
