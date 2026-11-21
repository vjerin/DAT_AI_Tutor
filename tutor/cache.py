"""Answer cache keyed on query, learner profile and config.

Entries are dropped when a cited source gets a new version or a memory item
they used is corrected or deleted. Abstentions are dropped on any corpus change.
"""
from __future__ import annotations

import threading
from typing import Optional

from .schema import TutorResponse
from .text import normalize, sha256_text


class AnswerCache:
    def __init__(self):
        self._lock = threading.RLock()
        self.entries: dict[str, TutorResponse] = {}
        self.by_source: dict[str, set[str]] = {}
        self.by_memory: dict[str, set[str]] = {}
        self.stats = {"hits": 0, "misses": 0, "invalidated": 0}

    @staticmethod
    def key(query: str, profile_ids: list[str], config_name: str) -> str:
        return sha256_text(normalize(query), "|".join(sorted(profile_ids)), config_name)

    def get(self, key: str) -> Optional[TutorResponse]:
        with self._lock:
            r = self.entries.get(key)
            self.stats["hits" if r else "misses"] += 1
            return r.model_copy(deep=True) if r else None

    def put(self, key: str, resp: TutorResponse, memory_ids: list[str]) -> None:
        with self._lock:
            self.entries[key] = resp.model_copy(deep=True)
            for c in resp.citations.values():
                self.by_source.setdefault(c.source_id, set()).add(key)
            for s in resp.suggestions:
                self.by_source.setdefault(s.source_id, set()).add(key)
            if not resp.citations:
                # abstentions depend on the whole corpus: any update may make them answerable
                self.by_source.setdefault("*", set()).add(key)
            for m in memory_ids:
                self.by_memory.setdefault(m, set()).add(key)

    def _drop(self, keys: set[str]) -> int:
        n = 0
        for k in keys:
            if self.entries.pop(k, None) is not None:
                n += 1
        self.stats["invalidated"] += n
        return n

    def invalidate_source(self, source_id: str, *_args) -> int:
        with self._lock:
            return self._drop(self.by_source.pop(source_id, set()) | self.by_source.pop("*", set()))

    def invalidate_memory(self, item_id: str) -> int:
        with self._lock:
            return self._drop(self.by_memory.pop(item_id, set()))

    def clear(self) -> None:
        with self._lock:
            self.entries.clear()
            self.by_source.clear()
            self.by_memory.clear()
