"""YAML configuration with `base:` inheritance and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "ingest": {
        "frame_fps": 1.0,
        "slide_change_threshold": 0.01,    # fraction of changed thumbnail pixels that starts a new slide
        "min_slide_ms": 2000,
        "ocr": "tesseract",                # tesseract | paddle | none
        "captioner": "none",               # none | vlm
        "asr": "transcript",               # transcript (use provided captions) | faster_whisper
        "asr_model": "large-v3",
        "segment_policy": "slide",         # slide | fixed
        "fixed_window_ms": 30000,
        "max_segment_ms": 90000,
        "min_segment_ms": 6000,
        "scan_injection": True,
    },
    "index": {
        "dense": "lsa",                    # lsa | sentence_transformers | none
        "dense_model": "BAAI/bge-m3",
        "query_prefix": "",
        "lsa_dims": 128,
        "visual": "text_proxy",            # text_proxy | clip | none
        "visual_model": "clip-ViT-B-32",
    },
    "retrieval": {
        "channels": ["sparse", "dense", "visual"],
        "k_sparse": 50,
        "k_dense": 50,
        "k_visual": 20,
        "rrf_k": 60,
        "class_weights": {                 # per-query-class channel weights for weighted RRF
            "exact_term": {"sparse": 1.5, "dense": 1.0, "visual": 0.8},
            "visual": {"sparse": 1.0, "dense": 1.0, "visual": 1.6},
        },
        "weighted_rrf": True,
        "reranker": "lexical",             # lexical | cross_encoder | none
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "rerank_top_n": 30,
        "final_k": 6,
        "include_superseded": False,
        "drop_flagged": True,              # drop evidence flagged as suspected injection
    },
    "classifier": {"type": "rules", "path": None},
    "rewrite": {"type": "heuristic"},      # heuristic | llm
    "generator": {
        "type": "extractive",              # extractive | llm
        "max_sentences": 3,
        "min_sentence_score": 0.25,
    },
    "llm": {
        "provider": "openai_compatible",   # openai_compatible (vLLM, Ollama, OpenAI) | anthropic
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen/Qwen3-8B",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.0,
        "max_tokens": 700,
        "timeout_s": 120,
        "vision": False,                   # send frame images for visual evidence
    },
    "verifier": {
        "type": "lexical",                 # lexical | nli | llm | none
        "nli_model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "support_threshold": 0.6,
        "drop_unsupported": True,
    },
    "abstention": {
        "type": "heuristic",               # heuristic | calibrated
        "path": None,
        "answer_threshold": 0.55,
        "qualify_threshold": 0.35,
    },
    "guard": {"type": "heuristic", "model": "meta-llama/Llama-Prompt-Guard-2-86M", "threshold": 0.5},
    "memory": {
        "enabled": True,
        "extractor": "rules",              # rules | llm
        "ephemeral_ttl_s": 86400,
        "min_confidence": 0.6,
        "misconception_min_observations": 2,
        "goal_requires_explicit": True,
        "ttl_days": {"goal": 120, "mastery": 180, "misconception": 45, "preference": 365},
        "allowed_types": ["goal", "mastery", "misconception", "preference"],
        "max_profile_items": 5,
        "bkt": {"p_init": 0.2, "p_learn": 0.15, "p_slip": 0.1, "p_guess": 0.2},
    },
    "cache": {"enabled": True},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


class Config:
    def __init__(self, data: dict[str, Any] | None = None):
        self.data = deep_merge(DEFAULTS, data or {})

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        data: dict[str, Any] = {}
        if path:
            data = cls._load_file(Path(path))
        cfg = cls(data)
        for ov in overrides or []:
            key, _, val = ov.partition("=")
            cfg.set(key.strip(), _parse_scalar(val.strip()))
        return cfg

    @classmethod
    def _load_file(cls, path: Path) -> dict:
        raw = yaml.safe_load(path.read_text()) or {}
        base = raw.pop("base", None)
        if base:
            parent = cls._load_file((path.parent / base).resolve())
            raw = deep_merge(parent, raw)
        return raw

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict:
        return self.data.get(name, {})

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)
