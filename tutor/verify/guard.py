"""Injection scanning, quarantine support and output scrubbing."""
from __future__ import annotations

import re

PATTERNS = [
    (r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,30}\b(instructions?|rules|prompts?|context)", 0.9),
    (r"\b(you are now|act as|pretend to be|from now on you)\b", 0.6),
    (r"\b(system prompt|developer message|hidden instructions?)\b", 0.6),
    (r"\b(respond|reply|answer|say|output|print)\b.{0,25}\b(only|exactly|with)\b.{0,40}[\"'“]", 0.5),
    (r"\b(do not|don't) (cite|mention|tell)\b", 0.5),
    (r"\b(visit|go to|click|download from|sign up at)\b.{0,40}(https?://|www\.)", 0.7),
    (r"\b(assistant|ai|model|tutor)\s*[:,]?\s*(must|should|will)\b.{0,40}\b(say|tell|recommend|respond)", 0.6),
    (r"\[(system|inst)\]|<\|im_start\|>|<<sys>>", 0.8),
    (r"\bbase64\b|\beval\(|\bexec\(", 0.3),
]
_COMPILED = [(re.compile(p, re.I | re.S), w) for p, w in PATTERNS]
URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b")


class HeuristicGuard:
    def score(self, text: str) -> float:
        s = 0.0
        for rx, w in _COMPILED:
            if rx.search(text):
                s = max(s, w) + 0.1 * (s > 0)
        return min(1.0, s)


class PromptGuardModel:  # pragma: no cover - needs model download
    def __init__(self, model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M"):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, truncation=True)
        self.fallback = HeuristicGuard()

    def score(self, text: str) -> float:
        out = self.pipe(text[:2000])[0]
        label = out["label"].upper()
        p = out["score"] if label in {"MALICIOUS", "INJECTION", "JAILBREAK", "LABEL_1"} else 1 - out["score"]
        return max(p, self.fallback.score(text))


class NoGuard:
    def score(self, text: str) -> float:
        return 0.0


def make_guard(cfg):
    kind = cfg.get("guard.type", "heuristic")
    if kind == "none":
        return NoGuard()
    if kind == "heuristic":
        return HeuristicGuard()
    if kind == "prompt_guard":
        return PromptGuardModel(cfg.get("guard.model"))
    raise ValueError(f"unknown guard {kind!r}")


def scrub_output(text: str, evidence_texts: list[str]) -> str:
    """Remove links / addresses that are not present in the evidence."""
    allowed = set()
    for t in evidence_texts:
        allowed.update(m.group(0).rstrip(".,)") for m in URL_RE.finditer(t))

    def repl(m):
        tok = m.group(0).rstrip(".,)")
        return m.group(0) if tok in allowed else "[link removed]"
    return URL_RE.sub(repl, text)
