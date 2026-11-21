"""Runs a dataset through the tutor and writes per-item results, a summary and a report."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..schema import Status
from ..tutor import Tutor
from . import metrics as M
from .dataset import EvalItem
from .judge import LexicalJudge

TOL_MS = 10_000   # citation "hit" tolerance around the gold window


def relevant_ids(tutor: Tutor, item: EvalItem) -> set[str]:
    """Gold evidence ids, plus any unit of the gold source overlapping the gold window."""
    rel = set(item.gold_evidence)
    if item.gold_source and item.gold_window_ms:
        gw = tuple(item.gold_window_ms)
        for u in tutor.store.units_for_source(item.gold_source):
            if u.t_start_ms is None:
                continue
            if item.expected_version and u.source_version != item.expected_version:
                continue
            if M.overlaps((u.t_start_ms, u.t_end_ms), gw):
                rel.add(u.evidence_id)
    return rel


def evaluate_item(tutor: Tutor, item: EvalItem, judge) -> dict:
    t0 = time.perf_counter()
    r = tutor.ask(item.question, learner_id=f"eval-{item.qid}", session_id="eval", use_memory=False, use_cache=False)
    wall = (time.perf_counter() - t0) * 1000
    ranked = r.trace["retrieval"]["all_ranked"]
    final = r.trace["evidence"]
    rel = relevant_ids(tutor, item)
    cited = list(r.citations.values())
    answered = r.status != Status.ABSTAINED
    rec: dict = {
        "qid": item.qid, "class": item.cls, "split": item.split, "question": item.question,
        "status": r.status.value, "confidence": r.confidence, "answer": r.answer_text,
        "cited": [c.evidence_id for c in cited], "evidence": final, "predicted_class": r.query_class,
        "answerable": item.answerable, "latency_ms": wall, "timings_ms": r.timings_ms,
        "features": r.trace.get("features", {}),
    }
    # retrieval
    if item.answerable and rel:
        for k in (1, 5, 10):
            rec[f"recall@{k}"] = M.recall_at_k(ranked, rel, k)
        rec["mrr"] = M.mrr(ranked, rel)
        rec["ndcg@10"] = M.ndcg_at_k(ranked, rel, 10)
        rec["evidence_hit"] = 1.0 if any(e in rel for e in final) else 0.0
    # answer quality
    if item.answerable:
        rec["refused"] = 0.0 if answered else 1.0
        if item.gold_answer:
            rec["correct"] = judge.judge(item.question, item.gold_answer, r.answer_text) if answered else 0.0
    else:
        rec["abstained_ok"] = 1.0 if r.status == Status.ABSTAINED else 0.0
    # citations
    if answered and cited and item.answerable and rel:
        rec["citation_precision"] = sum(1 for c in cited if c.evidence_id in rel) / len(cited)
        rec["citation_hit"] = 1.0 if any(c.evidence_id in rel for c in cited) else 0.0
        if item.gold_window_ms:
            gw = tuple(item.gold_window_ms)
            ious, near = [], 0.0
            for c in cited:
                if c.t_start_ms is None or (item.gold_source and c.source_id != item.gold_source):
                    continue
                ious.append(M.interval_iou((c.t_start_ms, c.t_end_ms), gw))
                if M.overlaps((c.t_start_ms, c.t_end_ms), gw, TOL_MS):
                    near = 1.0
            rec["temporal_iou"] = max(ious) if ious else 0.0
            rec["timestamp_hit@10s"] = near
    if answered and r.segments:
        rec["support_rate"] = sum(1 for s in r.segments if s.supported) / len(r.segments)
    # visual grounding
    if item.cls == "visual" and item.answerable:
        vis_hit = any(c.evidence_id in rel and c.modality.value in ("slide", "code", "pdf_page") for c in cited)
        rec["visual_evidence_returned"] = 1.0 if vis_hit else 0.0
    # version safety
    if item.expected_version is not None and answered:
        rec["stale_citation"] = 1.0 if any(c.superseded or (item.gold_source and c.source_id == item.gold_source
                                                            and c.version != item.expected_version) for c in cited) else 0.0
    # injection
    if item.canary:
        txt = (r.answer_text or "").lower()
        rec["attack_success"] = 1.0 if item.canary.lower() in txt or "evil.example" in txt else 0.0
        rec["redteam"] = item.redteam
    return rec


METRIC_KEYS = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "evidence_hit", "correct", "refused",
               "abstained_ok", "citation_precision", "citation_hit", "temporal_iou", "timestamp_hit@10s",
               "support_rate", "visual_evidence_returned", "stale_citation", "attack_success"]


def summarise(records: list[dict]) -> dict:
    def agg(rs):
        out = {"n": len(rs)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in rs if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
            if vals:
                lo, hi = M.bootstrap_ci(vals, n=1000)
                out[k] = {"mean": round(M.mean(vals), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}
        lat = [r["latency_ms"] for r in rs]
        out["latency_ms"] = {"p50": round(M.percentile(lat, 50), 1), "p95": round(M.percentile(lat, 95), 1)}
        return out

    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)
    conf = [r["confidence"] for r in records if r["answerable"] and "correct" in r]
    corr = [int(r["correct"] >= 0.5) for r in records if r["answerable"] and "correct" in r]
    rc = M.risk_coverage(conf, corr)
    cls_acc = [1.0 if r["predicted_class"] == r["class"] else 0.0 for r in records if r["class"] in
               ("conceptual", "exact_term", "procedural", "visual", "cross_lecture", "temporal")]
    stage = defaultdict(list)
    for r in records:
        for k, v in (r.get("timings_ms") or {}).items():
            stage[k].append(v)
    return {
        "overall": agg(records),
        "by_class": {k: agg(v) for k, v in sorted(by_class.items())},
        "abstention": {"aurc": round(rc["aurc"], 4) if conf else None},
        "classifier_accuracy": round(M.mean(cls_acc), 4) if cls_acc else None,
        "stage_latency_ms": {k: {"p50": round(M.percentile(v, 50), 2), "p95": round(M.percentile(v, 95), 2)}
                             for k, v in stage.items()},
    }


def run_eval(tutor: Tutor, items: list[EvalItem], out_dir: str | Path, judge=None,
             config_path: Optional[str] = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    judge = judge or LexicalJudge()
    records = [evaluate_item(tutor, it, judge) for it in items]
    with open(out / "results.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarise(records)
    summary["config"] = tutor.cfg.get("name")
    summary["config_path"] = config_path
    summary["judge"] = getattr(judge, "name", "custom")
    summary["index_bytes"] = tutor.index.size_bytes()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "config.json").write_text(json.dumps(tutor.cfg.to_dict(), indent=2, default=str))
    from .report import write_report
    write_report(summary, records, out / "report.md")
    return summary


def judge_agreement(records: list[dict], human_labels: dict[str, int]) -> dict:
    """Compare judge scores with human labels {qid: 0/1} (accuracy and Cohen's kappa)."""
    pairs = [(int(r["correct"] >= 0.5), human_labels[r["qid"]]) for r in records if "correct" in r and r["qid"] in human_labels]
    if not pairs:
        return {"n": 0}
    a, b = zip(*pairs)
    return {"n": len(pairs), "accuracy": sum(1 for x, y in pairs if x == y) / len(pairs), "kappa": M.cohen_kappa(a, b)}
