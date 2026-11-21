"""Dense text and image encoders: LSA (default), sentence-transformers, CLIP."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def lsa_analyzer(text: str) -> list[str]:
    from .bm25 import analyze
    toks = analyze(text)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class TextEncoder:
    name = "base"

    def fit(self, texts: Sequence[str]) -> "TextEncoder":
        return self

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_docs([text])[0]

    def save(self, path: Path) -> None:
        pass


class LSAEncoder(TextEncoder):
    name = "lsa"

    def __init__(self, dims: int = 128):
        self.dims = dims
        self.vec = None
        self.svd = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer=lsa_analyzer, sublinear_tf=True, min_df=1)
        X = self.vec.fit_transform(texts)
        k = min(self.dims, X.shape[0] - 1, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0) if k >= 2 else None
        if self.svd is not None:
            self.svd.fit(X)
        return self

    def encode_docs(self, texts):
        X = self.vec.transform(texts)
        Z = self.svd.transform(X) if self.svd is not None else X.toarray()
        return l2norm(np.asarray(Z, dtype=np.float32))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"dims": self.dims, "vec": self.vec, "svd": self.svd}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        e = cls(d["dims"])
        e.vec, e.svd = d["vec"], d["svd"]
        return e


class SentenceTransformerEncoder(TextEncoder):  # pragma: no cover - needs model download
    name = "sentence_transformers"

    def __init__(self, model_name: str = "BAAI/bge-m3", query_prefix: str = "", doc_prefix: str = "",
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size

    def encode_docs(self, texts):
        return np.asarray(self.model.encode([self.doc_prefix + t for t in texts], batch_size=self.batch_size,
                                            normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def encode_query(self, text):
        return np.asarray(self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0], dtype=np.float32)


class ClipEncoder:  # pragma: no cover - needs model download
    """Image-text encoder (sentence-transformers CLIP checkpoints)."""
    name = "clip"

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_images(self, paths):
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in paths]
        return np.asarray(self.model.encode(imgs, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def encode_query(self, text):
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def make_text_encoder(cfg) -> TextEncoder | None:
    kind = cfg.get("index.dense", "lsa")
    if kind == "none":
        return None
    if kind == "lsa":
        return LSAEncoder(cfg.get("index.lsa_dims", 128))
    if kind == "sentence_transformers":
        return SentenceTransformerEncoder(cfg.get("index.dense_model"), cfg.get("index.query_prefix", ""))
    raise ValueError(f"unknown dense encoder {kind!r}")
