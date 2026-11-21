"""T4: train the query-type classifier (TF-IDF word+char n-grams + logistic regression)."""
from __future__ import annotations

from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from ..eval.dataset import EvalItem
from ..retrieve.classify import CLASSES, LearnedClassifier, RuleClassifier


def train_classifier(items: list[EvalItem], out_path: str | Path, seed: int = 0) -> dict:
    data = [(i.question, i.cls) for i in items if i.cls in CLASSES]
    X, y = zip(*data)
    strat = y if min(y.count(c) for c in set(y)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(list(X), list(y), test_size=0.25, random_state=seed, stratify=strat)
    clf = LearnedClassifier.train(Xtr, ytr)
    pred = [clf.predict(q) for q in Xte]
    rules = [RuleClassifier().predict(q) for q in Xte]
    clf_full = LearnedClassifier.train(list(X), list(y))     # refit on everything for deployment
    clf_full.save(out_path)
    return {
        "n_train": len(Xtr), "n_test": len(Xte),
        "macro_f1_learned": f1_score(yte, pred, average="macro"),
        "macro_f1_rules": f1_score(yte, rules, average="macro"),
        "report": classification_report(yte, pred, zero_division=0),
        "path": str(out_path),
    }
