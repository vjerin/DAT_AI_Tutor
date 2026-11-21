"""Contrastive fine-tuning of the dense embedder.

python -m tutor.train.embedder --triplets runs/triplets.jsonl --base BAAI/bge-m3 --out models/emb-ft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(triplets_path: str, base: str, out: str, epochs: int = 1, batch_size: int = 32, lr: float = 2e-5,
          use_lora: bool = False) -> str:  # pragma: no cover - needs GPU / model download
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    rows = [json.loads(l) for l in Path(triplets_path).read_text().splitlines() if l.strip()]
    ds = Dataset.from_list([{"anchor": r["anchor"], "positive": r["positive"], "negative": r["negative"]} for r in rows])
    split = ds.train_test_split(test_size=0.1, seed=0)
    model = SentenceTransformer(base)
    if use_lora:
        from peft import LoraConfig, TaskType
        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1))
    args = SentenceTransformerTrainingArguments(
        output_dir=out, num_train_epochs=epochs, per_device_train_batch_size=batch_size, learning_rate=lr,
        warmup_ratio=0.1, bf16=True, eval_strategy="steps", eval_steps=100, save_steps=100, save_total_limit=2,
        logging_steps=20, seed=0, load_best_model_at_end=True)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=split["train"],
                                         eval_dataset=split["test"], loss=MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save_pretrained(out)
    return out


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--base", default="BAAI/bge-m3")
    ap.add_argument("--out", default="models/emb-ft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    print(train(a.triplets, a.base, a.out, a.epochs, a.batch_size, a.lr, a.lora))
