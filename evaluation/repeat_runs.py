"""Repeats the main evaluation to measure run-to-run variance (quality and latency)."""
import argparse
import json
import statistics
import tempfile
from pathlib import Path

from tutor.eval.dataset import load_dataset
from tutor.eval.runner import run_eval
from tutor.tutor import Tutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="data/demo_course/eval.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="evaluation/results/repeat_runs.json")
    a = ap.parse_args()
    items = load_dataset(a.dataset)
    rows = []
    for i in range(a.runs):
        t = Tutor.open(a.workspace, a.config)
        with tempfile.TemporaryDirectory() as d:
            s = run_eval(t, items, Path(d))
        o = s["overall"]
        rows.append({"run": i + 1, "correct": o["correct"]["mean"], "recall@5": o["recall@5"]["mean"],
                     "abstained_ok": o["abstained_ok"]["mean"], "p50_ms": o["latency_ms"]["p50"],
                     "p95_ms": o["latency_ms"]["p95"]})
    agg = {k: {"mean": statistics.mean(r[k] for r in rows), "stdev": statistics.pstdev(r[k] for r in rows)}
           for k in rows[0] if k != "run"}
    Path(a.out).write_text(json.dumps({"runs": rows, "aggregate": agg}, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
