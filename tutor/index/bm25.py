"""Okapi BM25 with a code-aware tokenizer."""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from ..text import light_stem, tokenize


def analyze(text: str) -> list[str]:
    return [light_stem(t) for t in tokenize(text)]


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.avgdl = 0.0

    def build(self, docs: Iterable[tuple[str, str]]) -> "BM25Index":
        self.ids, self.doc_len = [], []
        self.postings = defaultdict(list)
        for i, (doc_id, text) in enumerate(docs):
            toks = analyze(text)
            self.ids.append(doc_id)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        return self

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        N = len(self.ids)
        return math.log(1 + (N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 50, allowed: Optional[set[str]] = None) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(analyze(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for i, tf in plist:
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out = []
        for i, s in ranked:
            doc_id = self.ids[i]
            if allowed is not None and doc_id not in allowed:
                continue
            out.append((doc_id, float(s)))
            if len(out) >= k:
                break
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "ids": self.ids, "doc_len": self.doc_len,
                         "postings": dict(self.postings), "avgdl": self.avgdl}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls(d["k1"], d["b"])
        idx.ids, idx.doc_len, idx.avgdl = d["ids"], d["doc_len"], d["avgdl"]
        idx.postings = defaultdict(list, d["postings"])
        return idx
