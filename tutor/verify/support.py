"""Checks whether each answer sentence is supported by its cited evidence."""
from __future__ import annotations

from ..generate.llm import parse_json
from ..text import content_terms, numbers


class LexicalVerifier:
    def score(self, claim: str, evidence: list[str]) -> float:
        ev = " \n".join(evidence)
        terms = content_terms(claim)
        if not terms:
            return 1.0
        ev_terms = content_terms(ev)
        cov = len(terms & ev_terms) / len(terms)
        claim_nums = numbers(claim)
        if claim_nums and not claim_nums <= numbers(ev):
            cov = min(cov, 0.2)
        return cov


class NLIVerifier:  # pragma: no cover - needs model download
    def __init__(self, model_name: str):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True)

    def score(self, claim, evidence):
        best = 0.0
        for premise in evidence:
            out = self.pipe({"text": premise[:2000], "text_pair": claim})
            scores = out[0] if isinstance(out[0], list) else out
            ent = next((s["score"] for s in scores if s["label"].lower().startswith("entail")), 0.0)
            best = max(best, ent)
        return best


JUDGE_PROMPT = """Does the EVIDENCE fully support the CLAIM? Judge only from the evidence.
Return JSON: {{"supported": true|false, "score": <0..1>}}

EVIDENCE:
{evidence}

CLAIM: {claim}"""


class LLMJudgeVerifier:
    def __init__(self, llm):
        self.llm = llm

    def score(self, claim, evidence):
        raw = self.llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(
            evidence="\n---\n".join(e[:1500] for e in evidence), claim=claim)}], json_mode=True, max_tokens=60)
        try:
            d = parse_json(raw)
            return float(d.get("score", 1.0 if d.get("supported") else 0.0))
        except Exception:  # noqa: BLE001
            return 0.0


class NoVerifier:
    def score(self, claim, evidence):
        return 1.0


def make_verifier(cfg, llm=None):
    kind = cfg.get("verifier.type", "lexical")
    if kind == "none":
        return NoVerifier()
    if kind == "lexical":
        return LexicalVerifier()
    if kind == "nli":
        return NLIVerifier(cfg.get("verifier.nli_model"))
    if kind == "llm":
        if llm is None:
            raise ValueError("verifier.type=llm needs an llm section")
        return LLMJudgeVerifier(llm)
    raise ValueError(f"unknown verifier {kind!r}")
