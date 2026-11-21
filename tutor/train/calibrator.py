"""Fits the abstention calibrator on dev-split outcomes."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..eval.dataset import EvalItem
from ..eval.judge import LexicalJudge
from ..schema import Status
from ..tutor import Tutor
from ..verify.abstain import FEATURES, HeuristicAbstainer


def collect(tutor: Tutor, items: list[EvalItem], judge=None) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    judge = judge or LexicalJudge()
    saved = tutor.abstainer
    tutor.abstainer = HeuristicAbstainer(answer_threshold=0.0, qualify_threshold=0.0)   # always answer
    X, y, answerable = [], [], []
    try:
        for it in items:
            r = tutor.ask(it.question, learner_id="calib", use_memory=False, use_cache=False)
            f = r.trace["features"]
            X.append([f[k] for k in FEATURES])
            if it.answerable and it.gold_answer:
                ok = r.status != Status.ABSTAINED and judge.judge(it.question, it.gold_answer, r.answer_text) >= 0.5
            else:
                ok = False          # any answer to an unanswerable question is wrong
            y.append(1 if ok else 0)
            answerable.append(it.answerable)
    finally:
        tutor.abstainer = saved
    return np.array(X, dtype=float), np.array(y), answerable


def fit_calibrator(tutor: Tutor, dev_items: list[EvalItem], out_path: str | Path,
                   max_over_refusal: float = 0.15, qualify_gap: float = 0.2) -> dict:
    X, y, answerable = collect(tutor, dev_items)
    if len(set(y.tolist())) < 2:
        raise ValueError("calibration needs both correct and incorrect outcomes in the dev set")
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X, y)
    p = model.predict_proba(X)[:, 1]
    # pick the highest threshold whose refusal rate on *correctly answerable* items stays under target
    pos = np.array([pi for pi, yi, a in zip(p, y, answerable) if a and yi == 1])
    best = 0.5
    for t in np.linspace(0.95, 0.05, 91):
        refusal = float(np.mean(pos < t)) if len(pos) else 0.0
        if refusal <= max_over_refusal:
            best = float(t)
            break
    neg = [pi for pi, a in zip(p, answerable) if not a]
    abstain_rate_unans = float(np.mean(np.array(neg) < best)) if neg else None
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "answer_threshold": best,
                     "qualify_threshold": max(0.0, best - qualify_gap), "features": FEATURES}, f)
    return {"n": int(len(y)), "positives": int(y.sum()), "answer_threshold": round(best, 3),
            "qualify_threshold": round(max(0.0, best - qualify_gap), 3),
            "dev_abstain_rate_on_unanswerable": abstain_rate_unans,
            "coef": dict(zip(FEATURES, np.round(model.coef_[0], 3).tolist())), "path": str(out_path)}
