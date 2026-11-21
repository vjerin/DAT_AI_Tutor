"""Session store with expiry and SQLite-backed durable store with audit log."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Callable, Optional

from ..schema import MemoryItem, MemoryType


class EphemeralStore:
    def __init__(self, ttl_s: int = 86400, clock: Callable[[], float] = time.time):
        self.ttl_s, self.clock = ttl_s, clock
        self._data: dict[str, dict] = {}
        self._lock = threading.RLock()

    def _key(self, learner_id: str, session_id: str) -> str:
        return f"{learner_id}:{session_id}"

    def get(self, learner_id: str, session_id: str) -> dict:
        with self._lock:
            k = self._key(learner_id, session_id)
            v = self._data.get(k)
            if v and v["expires_at"] < self.clock():
                del self._data[k]
                v = None
            if v is None:
                v = {"turns": [], "summary": "", "topics": [], "expires_at": self.clock() + self.ttl_s}
                self._data[k] = v
            return v

    def append_turn(self, learner_id: str, session_id: str, role: str, content: str, max_turns: int = 12) -> None:
        with self._lock:
            s = self.get(learner_id, session_id)
            s["turns"].append({"role": role, "content": content, "ts": self.clock()})
            if len(s["turns"]) > max_turns:
                dropped = s["turns"][:-max_turns]
                s["turns"] = s["turns"][-max_turns:]
                s["summary"] = (s["summary"] + " " + " ".join(t["content"][:80] for t in dropped if t["role"] == "user")).strip()[-1500:]
            s["expires_at"] = self.clock() + self.ttl_s

    def history(self, learner_id: str, session_id: str) -> list[dict]:
        return list(self.get(learner_id, session_id)["turns"])

    def scrub(self, learner_id: str, needle: str) -> int:
        """Remove text containing ``needle`` from every session of a learner (deletion cascade)."""
        n = 0
        needle_l = needle.lower()
        with self._lock:
            for k, s in self._data.items():
                if not k.startswith(learner_id + ":"):
                    continue
                before = len(s["turns"])
                s["turns"] = [t for t in s["turns"] if needle_l not in t["content"].lower()]
                n += before - len(s["turns"])
                if needle_l in s["summary"].lower():
                    s["summary"] = ""
                    n += 1
        return n

    def contains(self, learner_id: str, needle: str) -> bool:
        needle_l = needle.lower()
        for k, s in self._data.items():
            if k.startswith(learner_id + ":"):
                if needle_l in s["summary"].lower() or any(needle_l in t["content"].lower() for t in s["turns"]):
                    return True
        return False

    def purge_expired(self) -> int:
        now = self.clock()
        with self._lock:
            dead = [k for k, v in self._data.items() if v["expires_at"] < now]
            for k in dead:
                del self._data[k]
        return len(dead)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
  item_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, type TEXT NOT NULL, key TEXT NOT NULL,
  value TEXT NOT NULL, confidence REAL, observations INTEGER, mastery_p REAL, provenance TEXT,
  created_at REAL, updated_at REAL, expires_at REAL, status TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_learner ON memory_items(learner_id, type, key);
CREATE TABLE IF NOT EXISTS memory_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, learner_id TEXT, item_id TEXT, action TEXT, detail TEXT
);
"""


class DurableStore:
    def __init__(self, db_path: str, clock: Callable[[], float] = time.time):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.clock = clock
        self._lock = threading.RLock()

    def _row(self, r) -> MemoryItem:
        d = dict(r)
        d["provenance"] = json.loads(d["provenance"] or "[]")
        d["type"] = MemoryType(d["type"])
        return MemoryItem(**d)

    def find(self, learner_id: str, type_: MemoryType, key: str) -> Optional[MemoryItem]:
        r = self.conn.execute("SELECT * FROM memory_items WHERE learner_id=? AND type=? AND key=?",
                              (learner_id, type_.value, key)).fetchone()
        return self._row(r) if r else None

    def get(self, item_id: str) -> Optional[MemoryItem]:
        r = self.conn.execute("SELECT * FROM memory_items WHERE item_id=?", (item_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, learner_id: str, include_pending: bool = False) -> list[MemoryItem]:
        now = self.clock()
        rows = self.conn.execute("SELECT * FROM memory_items WHERE learner_id=? ORDER BY updated_at DESC",
                                 (learner_id,)).fetchall()
        items = []
        for r in rows:
            it = self._row(r)
            if it.expires_at and it.expires_at < now:
                continue
            if it.status == "pending" and not include_pending:
                continue
            items.append(it)
        return items

    def upsert(self, item: MemoryItem, action: str, detail: dict | None = None) -> MemoryItem:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO memory_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item.item_id, item.learner_id, item.type.value, item.key, item.value, item.confidence,
                 item.observations, item.mastery_p, json.dumps(item.provenance), item.created_at,
                 item.updated_at, item.expires_at, item.status))
            # audit entries never carry memory content, so deletion leaves no copy behind
            self.audit(item.learner_id, item.item_id, action, detail or {"type": item.type.value})
            self.conn.commit()
        return item

    def delete(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            it = self.get(item_id)
            if not it:
                return None
            self.conn.execute("DELETE FROM memory_items WHERE item_id=?", (item_id,))
            # content-free audit entry: we record that something was deleted, not what
            self.audit(it.learner_id, item_id, "delete", {"type": it.type.value})
            self.conn.commit()
            return it

    def delete_learner(self, learner_id: str) -> int:
        with self._lock:
            n = self.conn.execute("DELETE FROM memory_items WHERE learner_id=?", (learner_id,)).rowcount
            self.audit(learner_id, "*", "delete_all", {"count": n})
            self.conn.commit()
            return n

    def audit(self, learner_id: str, item_id: str, action: str, detail: dict) -> None:
        self.conn.execute("INSERT INTO memory_audit (ts, learner_id, item_id, action, detail) VALUES (?,?,?,?,?)",
                          (self.clock(), learner_id, item_id, action, json.dumps(detail)))

    def audit_log(self, learner_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT ts, item_id, action, detail FROM memory_audit WHERE learner_id=? ORDER BY id",
                                 (learner_id,)).fetchall()
        return [dict(r) for r in rows]

    def raw_contains(self, needle: str) -> bool:
        """For deletion tests: is ``needle`` anywhere in the durable tables?"""
        like = f"%{needle}%"
        a = self.conn.execute("SELECT 1 FROM memory_items WHERE value LIKE ? OR key LIKE ? OR provenance LIKE ?",
                              (like, like, like)).fetchone()
        b = self.conn.execute("SELECT 1 FROM memory_audit WHERE detail LIKE ?", (like,)).fetchone()
        return bool(a or b)


def new_id() -> str:
    return "mem_" + uuid.uuid4().hex[:12]
