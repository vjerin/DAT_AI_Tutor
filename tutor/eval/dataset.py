"""Evaluation dataset schema (JSONL, one question per line)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvalItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qid: str
    split: str = "test"
    cls: str = Field("conceptual", alias="class")
    question: str
    answerable: bool = True
    gold_answer: Optional[str] = None
    gold_evidence: list[str] = Field(default_factory=list)
    gold_source: Optional[str] = None             # source_id, for window-based matching across versions
    gold_window_ms: Optional[list[int]] = None    # [start, end]
    expected_version: Optional[int] = None        # stale-version items
    redteam: Optional[str] = None                 # injection type
    canary: Optional[str] = None                  # string that must NOT appear in the answer
    notes: Optional[str] = None


def load_dataset(path: str | Path, split: Optional[str] = None, classes: Optional[list[str]] = None) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        it = EvalItem.model_validate(json.loads(line))
        if split and it.split != split:
            continue
        if classes and it.cls not in classes:
            continue
        items.append(it)
    return items


def save_dataset(items: list[EvalItem], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(i.model_dump(by_alias=True, exclude_none=True)) for i in items) + "\n")
