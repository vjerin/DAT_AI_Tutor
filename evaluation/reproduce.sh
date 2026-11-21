#!/usr/bin/env bash
# Regenerates every number in docs/Project_Report.pdf. Run from the repository root.
set -euo pipefail
WS=${WS:-workspace}
OUT=${OUT:-evaluation/results}
DATA=data/demo_course

python -m tutor.cli ingest --workspace "$WS" --config configs/default.yaml --manifest $DATA/manifest_v1.yaml
python -m tutor.cli ingest --workspace "$WS" --config configs/default.yaml --manifest $DATA/manifest_v2.yaml

python -m tutor.cli eval --workspace "$WS" --config configs/default.yaml --dataset $DATA/eval.jsonl --out "$OUT/solution_all"
python -m tutor.cli eval --workspace "$WS" --config configs/default.yaml --dataset $DATA/eval.jsonl --split heldout --out "$OUT/solution_heldout"
python -m tutor.cli eval --workspace "$WS" --config configs/baseline.yaml --dataset $DATA/eval.jsonl --out "$OUT/baseline_all"

python -m tutor.cli sweep --workspace "$WS" --dataset $DATA/eval.jsonl --out "$OUT/sweep" \
  --configs configs/baseline.yaml configs/default.yaml configs/bm25_only.yaml configs/dense_only.yaml \
            configs/hybrid_no_rerank.yaml configs/transcript_only.yaml
python -m tutor.cli sweep --dataset $DATA/eval.jsonl --out "$OUT/chunking" \
  --configs configs/default.yaml configs/fixed30.yaml --manifests $DATA/manifest_v1.yaml $DATA/manifest_v2.yaml

python -m tutor.cli redteam --workspace "$WS" --config configs/default.yaml --dataset $DATA/eval.jsonl \
  --out "$OUT/redteam_defended" --redteam-workspace "${WS}_rt_defended"
python -m tutor.cli redteam --workspace "$WS" --config configs/no_defence.yaml --dataset $DATA/eval.jsonl \
  --out "$OUT/redteam_undefended" --redteam-workspace "${WS}_rt_undefended"

python -m tutor.cli memtest | tee "$OUT/memtest.txt"
PYTHONPATH=. python evaluation/repeat_runs.py --workspace "$WS" --runs 3 --out "$OUT/repeat_runs.json"
python evaluation/environment.py > "$OUT/environment.txt"
python -m tutor.cli compare "$OUT/baseline_all" "$OUT/solution_all" | tee "$OUT/baseline_vs_solution.md"
