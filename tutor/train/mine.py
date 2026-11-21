"""Hard-negative mining for embedder and reranker training."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..index.bm25 import BM25Index
from ..store import EvidenceStore
from ..eval.dataset import EvalItem
from ..eval.metrics import overlaps


def mine_triplets(store: EvidenceStore, items: list[EvalItem], negatives_per_query: int = 3, pool: int = 30,
                  seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    units = {u.evidence_id: u for u in store.all_units(current_only=True)}
    bm25 = BM25Index().build((i, u.search_text()) for i, u in units.items())
    triplets = []
    for it in items:
        if not it.answerable:
            continue
        gold_ids = list(it.gold_evidence)
        if not gold_ids and it.gold_source and it.gold_window_ms:
            gw0 = tuple(it.gold_window_ms)
            gold_ids = [u.evidence_id for u in units.values() if u.source_id == it.gold_source and u.t_start_ms is not None
                        and overlaps((u.t_start_ms, u.t_end_ms), gw0)]
        positives = [units[g] for g in gold_ids if g in units]
        if not positives:
            continue
        if it.gold_answer:      # the positive is the gold unit that actually contains the answer
            from ..text import content_terms
            g = content_terms(it.gold_answer)
            positives.sort(key=lambda u: -len(g & content_terms(u.search_text())))
        pos = positives[0]
        gw = tuple(it.gold_window_ms) if it.gold_window_ms else None
        negs = []
        for doc_id, _ in bm25.search(it.question, pool):
            u = units[doc_id]
            if doc_id in gold_ids or u.course != pos.course:
                continue
            if gw and u.source_id == pos.source_id and u.t_start_ms is not None and overlaps((u.t_start_ms, u.t_end_ms), gw):
                continue
            negs.append(u)
        rng.shuffle(negs)
        for n in negs[:negatives_per_query]:
            triplets.append({"anchor": it.question, "positive": pos.search_text(), "negative": n.search_text(),
                             "qid": it.qid})
    return triplets


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
