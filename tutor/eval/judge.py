"""Answer correctness judges (lexical and LLM-based, with an on-disk cache)."""
from __future__ import annotations

import json
from pathlib import Path

from ..generate.llm import parse_json
from ..text import numbers, sha256_text
from .metrics import gold_recall


class LexicalJudge:
    name = "lexical"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def judge(self, question: str, gold: str, answer: str) -> float:
        if not answer.strip():
            return 0.0
        gn = numbers(gold)
        if gn and not gn <= numbers(answer):
            return 0.0
        return 1.0 if gold_recall(answer, gold) >= self.threshold else 0.0


JUDGE_PROMPT = """You grade a tutor's answer against a reference answer.
Score 1 if the answer is correct and consistent with the reference (extra correct detail is fine),
0 if it is wrong, contradicts the reference or misses its key point.
Return JSON: {{"score": 0 or 1, "reason": "<short>"}}

Question: {q}
Reference: {gold}
Answer: {answer}"""


class LLMJudge:
    name = "llm"

    def __init__(self, llm, cache_dir: str | Path, model_name: str = "judge"):
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def judge(self, question, gold, answer):
        key = sha256_text(self.model_name, JUDGE_PROMPT, question, gold, answer)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())["score"]
        raw = self.llm.complete([{"role": "user", "content": JUDGE_PROMPT.format(q=question, gold=gold, answer=answer)}],
                                json_mode=True, max_tokens=80)
        try:
            score = float(parse_json(raw).get("score", 0))
        except Exception:  # noqa: BLE001
            score = 0.0
        f.write_text(json.dumps({"score": score, "raw": raw}))
        return score
