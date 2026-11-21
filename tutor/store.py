"""SQLite source registry and evidence store.

A new content hash for an existing source creates a new version; older
versions and their evidence are marked as superseded, never overwritten.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .schema import EvidenceUnit, Modality, Source


class Workspace:
    """Folder that holds the database, indexes, frames and caches."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("indexes", "frames", "cache", "models", "runs"):
            (self.root / sub).mkdir(exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.root / "tutor.db"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT NOT NULL, version INTEGER NOT NULL, course TEXT, title TEXT,
  media_uri TEXT, slides_uri TEXT, transcript_uri TEXT, url TEXT, licence TEXT,
  content_hash TEXT NOT NULL, ingested_at REAL, is_current INTEGER DEFAULT 1,
  superseded_by INTEGER, PRIMARY KEY (source_id, version)
);
CREATE TABLE IF NOT EXISTS evidence (
  evidence_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_version INTEGER NOT NULL,
  course TEXT, modality TEXT, t_start_ms INTEGER, t_end_ms INTEGER, slide_no INTEGER,
  is_current INTEGER DEFAULT 1, flags TEXT, body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_evidence_source ON evidence(source_id, source_version);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, detail TEXT
);
"""


class EvidenceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._listeners: list = []   # callbacks(source_id, old_version, new_version)

    def on_new_version(self, callback) -> None:
        self._listeners.append(callback)

    def latest_source(self, source_id: str) -> Optional[Source]:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE source_id=? ORDER BY version DESC LIMIT 1", (source_id,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def get_source(self, source_id: str, version: int) -> Optional[Source]:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE source_id=? AND version=?", (source_id, version)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def list_sources(self, current_only: bool = False) -> list[Source]:
        q = "SELECT * FROM sources" + (" WHERE is_current=1" if current_only else "") + " ORDER BY source_id, version"
        return [self._row_to_source(r) for r in self.conn.execute(q).fetchall()]

    def register_source(self, src: dict) -> tuple[Source, bool]:
        """Register a source; returns (source, is_new_version).

        If the content hash matches the latest version nothing changes.
        """
        with self._lock:
            latest = self.latest_source(src["source_id"])
            if latest and latest.content_hash == src["content_hash"]:
                return latest, False
            version = (latest.version + 1) if latest else 1
            s = Source(version=version, ingested_at=time.time(), is_current=True, **src)
            self.conn.execute(
                "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.source_id, s.version, s.course, s.title, s.media_uri, s.slides_uri,
                 s.transcript_uri, s.url, s.licence, s.content_hash, s.ingested_at, 1, None),
            )
            if latest:
                self.conn.execute(
                    "UPDATE sources SET is_current=0, superseded_by=? WHERE source_id=? AND version<?",
                    (version, s.source_id, version),
                )
                self.conn.execute(
                    "UPDATE evidence SET is_current=0 WHERE source_id=? AND source_version<?",
                    (s.source_id, version),
                )
            self._log("new_version" if latest else "new_source", {"source_id": s.source_id, "version": version})
            self.conn.commit()
        if latest:
            for cb in self._listeners:
                cb(s.source_id, latest.version, version)
        return s, True

    @staticmethod
    def _row_to_source(r: sqlite3.Row) -> Source:
        d = dict(r)
        d["is_current"] = bool(d["is_current"])
        return Source(**d)

    def add_units(self, units: Iterable[EvidenceUnit]) -> int:
        n = 0
        with self._lock:
            for u in units:
                self.conn.execute(
                    "INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (u.evidence_id, u.source_id, u.source_version, u.course, u.modality.value,
                     u.t_start_ms, u.t_end_ms, u.slide_no, int(u.is_current), json.dumps(u.flags),
                     u.model_dump_json()),
                )
                n += 1
            self.conn.commit()
        return n

    def get(self, evidence_id: str) -> Optional[EvidenceUnit]:
        row = self.conn.execute("SELECT body, is_current FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        if not row:
            return None
        u = EvidenceUnit.model_validate_json(row["body"])
        u.is_current = bool(row["is_current"])
        return u

    def get_many(self, ids: Iterable[str]) -> dict[str, EvidenceUnit]:
        out = {}
        for i in ids:
            u = self.get(i)
            if u:
                out[i] = u
        return out

    def all_units(self, current_only: bool = False, course: Optional[str] = None) -> list[EvidenceUnit]:
        q, args = "SELECT body, is_current FROM evidence WHERE 1=1", []
        if current_only:
            q += " AND is_current=1"
        if course:
            q += " AND course=?"
            args.append(course)
        q += " ORDER BY source_id, source_version, t_start_ms, evidence_id"
        out = []
        for r in self.conn.execute(q, args).fetchall():
            u = EvidenceUnit.model_validate_json(r["body"])
            u.is_current = bool(r["is_current"])
            out.append(u)
        return out

    def current_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT evidence_id FROM evidence WHERE is_current=1")}

    def units_for_source(self, source_id: str, version: Optional[int] = None) -> list[EvidenceUnit]:
        q, args = "SELECT body, is_current FROM evidence WHERE source_id=?", [source_id]
        if version is not None:
            q += " AND source_version=?"
            args.append(version)
        q += " ORDER BY t_start_ms, evidence_id"
        res = []
        for r in self.conn.execute(q, args).fetchall():
            u = EvidenceUnit.model_validate_json(r["body"])
            u.is_current = bool(r["is_current"])
            res.append(u)
        return res

    def neighbours(self, unit: EvidenceUnit, modality: Optional[Modality] = None) -> list[EvidenceUnit]:
        """Units of the same source/version overlapping the unit's time span."""
        if unit.t_start_ms is None:
            return []
        out = []
        for u in self.units_for_source(unit.source_id, unit.source_version):
            if u.evidence_id == unit.evidence_id or u.t_start_ms is None:
                continue
            if modality and u.modality != modality:
                continue
            if u.t_start_ms < (unit.t_end_ms or 0) and (u.t_end_ms or 0) > unit.t_start_ms:
                out.append(u)
        return out

    def count(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT modality, is_current, COUNT(*) c FROM evidence GROUP BY modality, is_current"
        ).fetchall()
        return {f"{r['modality']}{'' if r['is_current'] else ' (superseded)'}": r["c"] for r in rows}

    def _log(self, kind: str, detail: dict) -> None:
        self.conn.execute("INSERT INTO events (ts, kind, detail) VALUES (?,?,?)", (time.time(), kind, json.dumps(detail)))
