"""Builds, saves and queries the sparse, dense and visual indexes.

All versions are indexed; filtering to current evidence happens at query time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..schema import EvidenceUnit
from .bm25 import BM25Index
from .encoders import ClipEncoder, LSAEncoder, TextEncoder, make_text_encoder


class FlatIndex:
    def __init__(self, ids: list[str], vecs: np.ndarray):
        self.ids = ids
        self.vecs = vecs.astype(np.float32) if len(ids) else np.zeros((0, 1), np.float32)

    def search(self, q: np.ndarray, k: int, allowed: Optional[set[str]] = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.vecs @ q.astype(np.float32)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if allowed is not None and self.ids[i] not in allowed:
                continue
            out.append((self.ids[i], float(sims[i])))
            if len(out) >= k:
                break
        return out


class IndexManager:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bm25: Optional[BM25Index] = None
        self.encoder: Optional[TextEncoder] = None
        self.dense: Optional[FlatIndex] = None
        self.visual: Optional[FlatIndex] = None
        self.visual_kind = cfg.get("index.visual", "text_proxy")
        self.clip: Optional[ClipEncoder] = None

    def build(self, units: list[EvidenceUnit]) -> dict:
        texts = [u.search_text() for u in units]
        ids = [u.evidence_id for u in units]
        self.bm25 = BM25Index().build(zip(ids, texts))
        self.encoder = make_text_encoder(self.cfg)
        if self.encoder is not None and units:
            self.encoder.fit(texts)
            self.dense = FlatIndex(ids, self.encoder.encode_docs(texts))
        vis = [u for u in units if u.is_visual]
        if vis and self.visual_kind == "clip":
            self.clip = ClipEncoder(self.cfg.get("index.visual_model"))
            self.visual = FlatIndex([u.evidence_id for u in vis], self.clip.encode_images([u.frame_uri for u in vis]))
        elif vis and self.visual_kind == "text_proxy" and self.encoder is not None:
            self.visual = FlatIndex([u.evidence_id for u in vis], self.encoder.encode_docs([u.search_text() for u in vis]))
        self.save()
        return {"units": len(units), "visual_units": len(vis), "dense": self.encoder.name if self.encoder else None,
                "visual": self.visual_kind if vis else None}

    def save(self) -> None:
        if self.bm25:
            self.bm25.save(self.root / "bm25.pkl")
        meta = {"dense": self.encoder.name if self.encoder else None, "visual": self.visual_kind}
        if self.encoder is not None:
            if isinstance(self.encoder, LSAEncoder):
                self.encoder.save(self.root / "lsa.pkl")
        if self.dense is not None:
            np.save(self.root / "dense.npy", self.dense.vecs)
            (self.root / "dense_ids.json").write_text(json.dumps(self.dense.ids))
        if self.visual is not None:
            np.save(self.root / "visual.npy", self.visual.vecs)
            (self.root / "visual_ids.json").write_text(json.dumps(self.visual.ids))
        (self.root / "meta.json").write_text(json.dumps(meta))

    def load(self) -> "IndexManager":
        if not (self.root / "bm25.pkl").exists():
            raise FileNotFoundError(f"no index in {self.root}; run `tutor index` first")
        self.bm25 = BM25Index.load(self.root / "bm25.pkl")
        meta = json.loads((self.root / "meta.json").read_text())
        if meta.get("dense") == "lsa":
            self.encoder = LSAEncoder.load(self.root / "lsa.pkl")
        elif meta.get("dense"):
            self.encoder = make_text_encoder(self.cfg)
        if (self.root / "dense.npy").exists():
            self.dense = FlatIndex(json.loads((self.root / "dense_ids.json").read_text()), np.load(self.root / "dense.npy"))
        if (self.root / "visual.npy").exists():
            self.visual = FlatIndex(json.loads((self.root / "visual_ids.json").read_text()), np.load(self.root / "visual.npy"))
            if meta.get("visual") == "clip":
                self.clip = ClipEncoder(self.cfg.get("index.visual_model"))
        self.visual_kind = meta.get("visual", self.visual_kind)
        return self

    def search(self, query: str, allowed: Optional[set[str]], channels: list[str],
               k: dict[str, int]) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        if "sparse" in channels and self.bm25:
            out["sparse"] = self.bm25.search(query, k.get("sparse", 50), allowed)
        qvec = None
        if ("dense" in channels or "visual" in channels) and self.encoder is not None:
            qvec = self.encoder.encode_query(query)
        if "dense" in channels and self.dense is not None and qvec is not None:
            out["dense"] = [(i, s) for i, s in self.dense.search(qvec, k.get("dense", 50), allowed) if s > 0]
        if "visual" in channels and self.visual is not None:
            vq = self.clip.encode_query(query) if self.clip is not None else qvec
            if vq is not None:
                out["visual"] = [(i, s) for i, s in self.visual.search(vq, k.get("visual", 20), allowed) if s > 0]
        return out

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*") if p.is_file())
