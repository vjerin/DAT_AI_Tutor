"""Markdown reports for single runs and configuration comparisons."""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics as M

HEADLINE = ["recall@5", "mrr", "correct", "citation_hit", "timestamp_hit@10s", "temporal_iou", "support_rate",
            "refused", "abstained_ok", "visual_evidence_returned", "stale_citation", "attack_success"]


def _fmt(v) -> str:
    if not isinstance(v, dict):
        return "-"
    lo, hi = v["ci95"]
    if lo != lo:  # nan
        return f"{v['mean']:.2f}"
    return f"{v['mean']:.2f} [{lo:.2f}, {hi:.2f}]"


def write_report(summary: dict, records: list[dict], path: str | Path) -> None:
    lines = [f"# Evaluation report: `{summary.get('config')}`", ""]
    o = summary["overall"]
    lines += [f"Items: **{o['n']}** · judge: {summary.get('judge')} · latency p50/p95: "
              f"{o['latency_ms']['p50']} / {o['latency_ms']['p95']} ms · index size: {summary.get('index_bytes', 0)/1e6:.2f} MB",
              f"Query-classifier accuracy: {summary.get('classifier_accuracy')} · abstention AURC: {summary['abstention']['aurc']}", ""]
    lines += ["## Overall (mean [95% bootstrap CI])", "", "| metric | value | n |", "|---|---|---|"]
    for k in HEADLINE:
        if k in o:
            lines.append(f"| {k} | {_fmt(o[k])} | {o[k]['n']} |")
    lines += ["", "## By question class", ""]
    cols = ["recall@5", "correct", "citation_hit", "timestamp_hit@10s", "refused", "abstained_ok", "visual_evidence_returned", "stale_citation", "attack_success"]
    lines.append("| class | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for cls, s in summary["by_class"].items():
        lines.append(f"| {cls} | {s['n']} | " + " | ".join(f"{s[c]['mean']:.2f}" if c in s else "-" for c in cols) + " |")
    lines += ["", "## Stage latency (ms)", "", "| stage | p50 | p95 |", "|---|---|---|"]
    for k, v in summary.get("stage_latency_ms", {}).items():
        lines.append(f"| {k} | {v['p50']} | {v['p95']} |")
    fails = [r for r in records if r.get("correct") == 0.0 or r.get("abstained_ok") == 0.0 or r.get("attack_success") == 1.0
             or r.get("stale_citation") == 1.0]
    lines += ["", f"## Failures ({len(fails)})", ""]
    for r in fails[:25]:
        lines.append(f"- **{r['qid']}** ({r['class']}, {r['status']}): {r['question']}  ")
        lines.append(f"  answer: {r['answer'][:200] or '(none)'} · cited: {', '.join(r['cited']) or '-'}")
    Path(path).write_text("\n".join(lines) + "\n")


def compare_runs(run_a: str | Path, run_b: str | Path, metrics: list[str] | None = None) -> str:
    """Paired comparison of two runs on the same dataset (bootstrap CI + p-value)."""
    load = lambda p: {json.loads(l)["qid"]: json.loads(l) for l in Path(p, "results.jsonl").read_text().splitlines() if l}  # noqa: E731
    A, B = load(run_a), load(run_b)
    qids = sorted(set(A) & set(B))
    metrics = metrics or ["recall@5", "mrr", "correct", "citation_hit", "timestamp_hit@10s", "abstained_ok", "refused", "attack_success"]
    out = [f"# Paired comparison: {Path(run_a).name} -> {Path(run_b).name}", "",
           "| metric | n | A | B | B-A | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    for m in metrics:
        a = [A[q].get(m, float("nan")) for q in qids]
        b = [B[q].get(m, float("nan")) for q in qids]
        res = M.paired_bootstrap(a, b)
        if res["n"] < 2:
            continue
        ma, mb = M.mean(a), M.mean(b)
        out.append(f"| {m} | {res['n']} | {ma:.3f} | {mb:.3f} | {res['diff']:+.3f} | "
                   f"[{res['ci'][0]:+.3f}, {res['ci'][1]:+.3f}] | {res['p']:.3f} |")
    la = [A[q]["latency_ms"] for q in qids]
    lb = [B[q]["latency_ms"] for q in qids]
    out += ["", f"Latency p50: {M.percentile(la, 50):.1f} -> {M.percentile(lb, 50):.1f} ms; "
                f"p95: {M.percentile(la, 95):.1f} -> {M.percentile(lb, 95):.1f} ms"]
    return "\n".join(out) + "\n"
