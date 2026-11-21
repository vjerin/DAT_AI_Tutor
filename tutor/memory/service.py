"""Learner memory: session state, durable learner model, correction and deletion."""
from __future__ import annotations

import time
from typing import Callable

from ..schema import MemoryItem, MemoryOp, MemoryType
from .bkt import BKT
from .extractor import LLMExtractor, RuleExtractor
from .policy import Decision, MemoryPolicy
from .store import DurableStore, EphemeralStore, new_id


class MemoryService:
    def __init__(self, cfg: dict, db_path: str, llm=None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.clock = clock
        self.ephemeral = EphemeralStore(cfg.get("ephemeral_ttl_s", 86400), clock)
        self.durable = DurableStore(db_path, clock)
        self.policy = MemoryPolicy(cfg)
        b = cfg.get("bkt", {})
        self.bkt = BKT(**b) if b else BKT()
        self.extractor = LLMExtractor(llm) if cfg.get("extractor") == "llm" and llm else RuleExtractor()
        self._on_delete: list[Callable[[str], None]] = []
        self._last_concepts: dict[str, list[str]] = {}

    def on_delete(self, cb: Callable[[str], None]) -> None:
        self._on_delete.append(cb)

    def history(self, learner_id: str, session_id: str) -> list[dict]:
        return self.ephemeral.history(learner_id, session_id)

    def record_turn(self, learner_id: str, session_id: str, role: str, content: str) -> None:
        self.ephemeral.append_turn(learner_id, session_id, role, content)

    def observe(self, learner_id: str, session_id: str, user_text: str, concepts: list[str],
                turn_id: str, evidence_ids: list[str]) -> list[dict]:
        prev = self._last_concepts.get(f"{learner_id}:{session_id}", [])
        ops = self.extractor.extract(user_text, concepts, prev)
        self._last_concepts[f"{learner_id}:{session_id}"] = concepts
        results = []
        for op in ops:
            op.source_turn = turn_id
            results.append(self.apply(learner_id, op, evidence_ids))
        return results

    def apply(self, learner_id: str, op: MemoryOp, evidence_ids: list[str] | None = None) -> dict:
        now = self.clock()
        if op.op == "DELETE":
            n = 0
            for it in self.durable.list(learner_id, include_pending=True):
                if op.type and it.type != op.type:
                    continue
                if op.key and op.key not in it.key and op.key not in it.value.lower():
                    continue
                self.delete(it.item_id)
                n += 1
            return {"op": "DELETE", "decision": Decision.COMMIT, "deleted": n}

        if op.op == "UPDATE" and op.type == MemoryType.GOAL and op.key == "__current_goal__":
            goals = [g for g in self.durable.list(learner_id, include_pending=True) if g.type == MemoryType.GOAL]
            if goals:
                g = goals[0]
                return {"op": "UPDATE", "decision": Decision.COMMIT, "item_id": self.correct(g.item_id, op.value).item_id}
            op = op.model_copy(update={"op": "ADD", "key": op.value.lower()[:60]})

        existing = self.durable.find(learner_id, op.type, op.key) if op.type else None
        decision, reason = self.policy.decide(op, existing)
        if decision == Decision.REJECT:
            return {"op": op.op, "type": op.type.value if op.type else None, "decision": decision, "reason": reason}
        prov = {"turn": op.source_turn, "evidence": (evidence_ids or [])[:3], "ts": now}
        ttl = self.policy.ttl_s(op.type)
        if op.type == MemoryType.MASTERY:
            p_prev = existing.mastery_p if existing else None
            p = self.bkt.update(p_prev, bool(op.correct))
            item = existing or MemoryItem(item_id=new_id(), learner_id=learner_id, type=op.type, key=op.key,
                                          value="", created_at=now, updated_at=now)
            item.mastery_p = p
            item.value = f"Mastery of '{op.key}': {p:.2f}"
            item.confidence = p
            item.observations = (existing.observations + 1) if existing else 1
        elif existing:
            item = existing
            item.observations += 1
            item.confidence = max(item.confidence, op.confidence)
            item.value = op.value or item.value
        else:
            item = MemoryItem(item_id=new_id(), learner_id=learner_id, type=op.type, key=op.key, value=op.value,
                              confidence=op.confidence, created_at=now, updated_at=now)
        item.updated_at = now
        item.expires_at = (now + ttl) if ttl else None
        item.status = "active" if decision == Decision.COMMIT else "pending"
        item.provenance = (item.provenance + [prov])[-10:]
        self.durable.upsert(item, action=f"{op.op.lower()}:{decision}")
        return {"op": op.op, "type": op.type.value, "decision": decision, "reason": reason, "item_id": item.item_id}

    def profile(self, learner_id: str, concepts: list[str]) -> list[MemoryItem]:
        items = self.durable.list(learner_id)
        cset = {c.lower() for c in concepts}

        def relevant(it: MemoryItem) -> bool:
            if it.type in (MemoryType.GOAL, MemoryType.PREFERENCE):
                return True
            k = it.key.lower()
            return k in cset or any(k in c or c in k for c in cset)

        chosen = [it for it in items if relevant(it)]
        order = {MemoryType.MISCONCEPTION: 0, MemoryType.MASTERY: 1, MemoryType.PREFERENCE: 2, MemoryType.GOAL: 3}
        chosen.sort(key=lambda it: (order[it.type], -it.updated_at))
        return chosen[: self.cfg.get("max_profile_items", 5)]

    def profile_lines(self, items: list[MemoryItem]) -> list[str]:
        lines = []
        for it in items:
            if it.type == MemoryType.MASTERY and it.mastery_p is not None:
                level = "low" if it.mastery_p < 0.4 else ("medium" if it.mastery_p < 0.75 else "high")
                lines.append(f"Mastery of {it.key}: {level}")
            else:
                lines.append(it.value)
        return lines

    def list(self, learner_id: str, include_pending: bool = True) -> list[MemoryItem]:
        return self.durable.list(learner_id, include_pending=include_pending)

    def correct(self, item_id: str, value: str) -> MemoryItem:
        it = self.durable.get(item_id)
        if not it:
            raise KeyError(item_id)
        old_phrases = self._phrases(it) if it.value != value else []
        it.value = value
        it.updated_at = self.clock()
        it.status = "active"
        it.confidence = 1.0
        self.durable.upsert(it, action="correct")
        for phrase in old_phrases:          # the corrected (wrong) version must not linger in session state
            if phrase.lower() not in value.lower():
                self.ephemeral.scrub(it.learner_id, phrase)
        for cb in self._on_delete:          # cached answers used the old value
            cb(item_id)
        return it

    @staticmethod
    def _phrases(it: MemoryItem) -> list[str]:
        """Text fragments that could hold the memory's content in session state."""
        v = it.value
        for prefix in ("Goal: ", "Prefers ", "May believe that "):
            if v.startswith(prefix):
                v = v[len(prefix):]
        v = v.split(" (")[0].strip()
        out = [it.value, v]
        if it.type != MemoryType.MASTERY and it.key and not it.key.startswith("__"):
            out.append(it.key)
        return [p for p in out if len(p) >= 4]

    def delete(self, item_id: str) -> bool:
        it = self.durable.delete(item_id)
        if not it:
            return False
        for phrase in self._phrases(it):
            self.ephemeral.scrub(it.learner_id, phrase)
        for cb in self._on_delete:
            cb(item_id)
        return True

    def delete_learner(self, learner_id: str) -> int:
        items = self.durable.list(learner_id, include_pending=True)
        for it in items:
            self.delete(it.item_id)
        return len(items)

    def is_forgotten(self, learner_id: str, text: str) -> bool:
        return not self.durable.raw_contains(text) and not self.ephemeral.contains(learner_id, text)
