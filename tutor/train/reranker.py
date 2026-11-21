"""Cross-encoder reranker fine-tuning.

python -m tutor.train.reranker --triplets runs/triplets.jsonl --out models/rerank-ft
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def train(triplets_path: str, base: str, out: str, epochs: int = 1, batch_size: int = 16,
          lr: float = 2e-5) -> str:  # pragma: no cover - needs GPU / model download
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample

    rows = [json.loads(l) for l in Path(triplets_path).read_text().splitlines() if l.strip()]
    seen_pos = set()
    examples = []
    for r in rows:
        if (r["anchor"], r["positive"]) not in seen_pos:
            examples.append(InputExample(texts=[r["anchor"], r["positive"]], label=1.0))
            seen_pos.add((r["anchor"], r["positive"]))
        examples.append(InputExample(texts=[r["anchor"], r["negative"]], label=0.0))
    random.Random(0).shuffle(examples)
    model = CrossEncoder(base, num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=int(0.1 * len(loader)),
              optimizer_params={"lr": lr}, show_progress_bar=True)
    model.save(out)
    return out


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="models/rerank-ft")
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    print(train(a.triplets, a.base, a.out, a.epochs))
