"""Turns a learner message into proposed memory operations."""
from __future__ import annotations

import re

from ..generate.llm import parse_json
from ..schema import MemoryOp, MemoryType

GOAL = re.compile(r"\b(?:my goal is(?: to)?|i want to (?:learn|understand|master|get better at)|i'?m (?:preparing|studying) for|i need to (?:learn|pass|understand)|i am (?:preparing|studying) for)\s+(.{3,120}?)(?:[.!?]|$)", re.I)
PREF = re.compile(r"\b(?:i prefer|please (?:always )?(?:use|give me|explain (?:it )?with)|i like (?:it )?when you (?:use|give)|i learn best (?:with|from))\s+(.{3,80}?)(?:[.!?]|$)", re.I)
MISC = re.compile(r"\b(?:i thought|i assumed|i believed|isn'?t it (?:true )?that|doesn'?t (?:it|that) mean)\s+(.{3,120}?)(?:[.!?]|$)", re.I)
POS = re.compile(r"\b(got it|makes sense( now)?|i understand( it)? now|that'?s clear|that helps|understood)\b", re.I)
NEG = re.compile(r"\b(still (don'?t|do not) (get|understand)|i'?m (confused|lost)|doesn'?t make sense|i don'?t understand)\b", re.I)
FORGET = re.compile(r"\b(?:forget|delete|remove|don'?t remember)\b.{0,20}?\b(?:that )?(?:my |the )?(goal|preference|misconception|that)\b(?:\s+(?:about|of|on)\s+(.{2,60}?))?(?:[.!?]|$)", re.I)
CORRECT = re.compile(r"\b(?:actually|no,?)\s+my goal is(?: to)?\s+(.{3,120}?)(?:[.!?]|$)", re.I)


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()[:60]


class RuleExtractor:
    def extract(self, user_text: str, concepts: list[str], prev_concepts: list[str]) -> list[MemoryOp]:
        ops: list[MemoryOp] = []
        m = CORRECT.search(user_text)
        if m:
            g = m.group(1).strip()
            ops.append(MemoryOp(op="UPDATE", type=MemoryType.GOAL, key="__current_goal__",
                                value=f"Goal: {g}", confidence=0.95, explicit=True))
        else:
            m = GOAL.search(user_text)
            if m:
                g = m.group(1).strip()
                ops.append(MemoryOp(op="ADD", type=MemoryType.GOAL, key=_norm_key(g), value=f"Goal: {g}",
                                    confidence=0.9, explicit=True))
        m = PREF.search(user_text)
        if m:
            p = m.group(1).strip()
            ops.append(MemoryOp(op="ADD", type=MemoryType.PREFERENCE, key=_norm_key(p), value=f"Prefers {p}",
                                confidence=0.85, explicit=True))
        m = MISC.search(user_text)
        if m and concepts:
            ops.append(MemoryOp(op="ADD", type=MemoryType.MISCONCEPTION, key=concepts[0],
                                value=f"May believe that {m.group(1).strip()} ({concepts[0]})", confidence=0.6))
        if POS.search(user_text):
            for c in (prev_concepts or concepts)[:2]:
                ops.append(MemoryOp(op="OBSERVE", type=MemoryType.MASTERY, key=c, value=c, correct=True, confidence=0.7))
        elif NEG.search(user_text):
            for c in (concepts or prev_concepts)[:2]:
                ops.append(MemoryOp(op="OBSERVE", type=MemoryType.MASTERY, key=c, value=c, correct=False, confidence=0.7))
        m = FORGET.search(user_text)
        if m:
            what = (m.group(1) or "").lower()
            about = (m.group(2) or "").strip()
            t = {"goal": MemoryType.GOAL, "preference": MemoryType.PREFERENCE,
                 "misconception": MemoryType.MISCONCEPTION}.get(what)
            ops.append(MemoryOp(op="DELETE", type=t, key=_norm_key(about), value=about, explicit=True))
        return ops


EXTRACT_PROMPT = """You maintain a tutor's memory about a learner. From the learner's message, propose
operations. Allowed types: goal (only if stated), preference, misconception (what they wrongly believe),
mastery (correct=true/false about a concept). Never store health, religion, ethnicity, politics,
finances or other sensitive data. Return JSON: {{"ops": [{{"op": "ADD|UPDATE|DELETE|OBSERVE|NOOP",
"type": "...", "key": "...", "value": "...", "confidence": 0..1, "explicit": true|false, "correct": null|true|false}}]}}

Concepts in this turn: {concepts}
Learner message: {text}"""


class LLMExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, user_text, concepts, prev_concepts):
        raw = self.llm.complete([{"role": "user", "content": EXTRACT_PROMPT.format(
            concepts=", ".join(concepts) or "none", text=user_text)}], json_mode=True, max_tokens=300)
        try:
            data = parse_json(raw)
        except Exception:  # noqa: BLE001
            return []
        ops = []
        for o in data.get("ops", []):
            try:
                ops.append(MemoryOp(**{k: v for k, v in o.items() if v is not None}))
            except Exception:  # noqa: BLE001
                continue
        return ops
