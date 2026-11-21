"""Turn follow-up questions into standalone queries using session history."""
from __future__ import annotations

import re

from ..text import tokenize

FOLLOWUP = re.compile(
    r"^\s*(and|what about|how about|why|so|then|also|but)\b|\b(it|that|this|those|these|they|them|the (first|second|third|last|other) one|same)\b",
    re.I,
)

REWRITE_PROMPT = (
    "Rewrite the learner's last question as a standalone search query using the conversation. "
    "Keep technical terms exactly. Output only the query.\n\nConversation:\n{history}\n\nLast question: {q}"
)


class HeuristicRewriter:
    def rewrite(self, query: str, history: list[dict]) -> str:
        if not history or not FOLLOWUP.search(query) or len(tokenize(query)) > 8:
            return query
        prev = next((h["content"] for h in reversed(history) if h.get("role") == "user"), "")
        q_terms = set(tokenize(query))
        extra = [t for t in tokenize(prev) if t not in q_terms][:6]
        return f"{query} ({' '.join(extra)})" if extra else query


class LLMRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query: str, history: list[dict]) -> str:
        if not history:
            return query
        hist = "\n".join(f"{h['role']}: {h['content'][:300]}" for h in history[-6:])
        out = self.llm.complete([{"role": "user", "content": REWRITE_PROMPT.format(history=hist, q=query)}],
                                json_mode=False, max_tokens=80)
        return out.strip().strip('"') or query


def make_rewriter(cfg, llm=None):
    if cfg.get("rewrite.type") == "llm" and llm is not None:
        return LLMRewriter(llm)
    return HeuristicRewriter()
