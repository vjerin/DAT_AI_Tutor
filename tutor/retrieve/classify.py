"""Query type classification (rules or a trained classifier)."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

CLASSES = ["conceptual", "exact_term", "procedural", "visual", "cross_lecture", "temporal"]

_TEMPORAL = re.compile(r"\b(when (does|did|is)|at what (time|point)|which (part|lecture|minute)|where in the (lecture|video)|timestamp|what time)\b", re.I)
_VISUAL = re.compile(r"\b(slides?|diagram|plot|figure|graph|chart|table|on (the )?screen|shown|picture|image|visible|axis|axes|drawn|board|written|displayed|code (on|in) the)\b", re.I)
_CODE = re.compile(r"([A-Za-z_]\w*\.\w+|\w+_\w+|\w+\(\)|`[^`]+`|\"[^\"]+\"|\b[a-z]+[A-Z]\w+\b)")
_EXACT = re.compile(r"\b(error|exception|traceback|function|method|argument|parameter|keyword|flag|api|syntax|called|named)\b", re.I)
_PROC = re.compile(r"\b(how (do|can|should) (i|we|you)|how to|steps?|procedure|implement|walk me through)\b", re.I)
_CROSS = re.compile(r"\b(compare|comparison|difference between|differ|versus|vs\.?|across (the )?lectures|relate|relationship between)\b", re.I)


class RuleClassifier:
    def predict(self, query: str) -> str:
        if _TEMPORAL.search(query):
            return "temporal"
        if _VISUAL.search(query):
            return "visual"
        if _CODE.search(query) or _EXACT.search(query):
            return "exact_term"
        if _CROSS.search(query):
            return "cross_lecture"
        if _PROC.search(query):
            return "procedural"
        return "conceptual"


class LearnedClassifier:
    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    @classmethod
    def train(cls, queries: list[str], labels: list[str]) -> "LearnedClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ])
        pipe = Pipeline([("f", feats), ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])
        pipe.fit(queries, labels)
        return cls(pipe)

    def predict(self, query: str) -> str:
        return str(self.pipeline.predict([query])[0])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedClassifier":
        with open(path, "rb") as f:
            return cls(pickle.load(f))


def make_classifier(cfg):
    if cfg.get("classifier.type", "rules") == "learned" and cfg.get("classifier.path"):
        return LearnedClassifier.load(cfg.get("classifier.path"))
    return RuleClassifier()
