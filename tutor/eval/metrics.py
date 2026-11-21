"""Metric functions for the evaluation harness (M7, M8)."""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def overlaps(a: tuple[int, int], b: tuple[int, int], tol_ms: int = 0) -> bool:
    return a[0] - tol_ms < b[1] and b[0] - tol_ms < a[1]


def _norm_tokens(s: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", s.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = _norm_tokens(pred), _norm_tokens(gold)
    if not p or not g:
        return 0.0
    common = 0
    gc = {}
    for t in g:
        gc[t] = gc.get(t, 0) + 1
    for t in p:
        if gc.get(t, 0) > 0:
            common += 1
            gc[t] -= 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def gold_recall(pred: str, gold: str) -> float:
    """Fraction of gold tokens present in the prediction (answers are often longer than gold)."""
    p, g = set(_norm_tokens(pred)), _norm_tokens(gold)
    return sum(1 for t in g if t in p) / len(g) if g else 0.0


def mean(xs: Iterable[float]) -> float:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def bootstrap_ci(xs: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(float(np.mean([v[rng.randrange(len(v))] for _ in v])) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 0) -> dict:
    """Two configurations on the same items: mean difference (b - a), CI and p-value."""
    pairs = [(x, y) for x, y in zip(a, b) if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return {"n": len(pairs), "diff": float("nan"), "ci": (float("nan"), float("nan")), "p": float("nan")}
    diffs = [y - x for x, y in pairs]
    obs = float(np.mean(diffs))
    rng = random.Random(seed)
    boots = sorted(float(np.mean([diffs[rng.randrange(len(diffs))] for _ in diffs])) for _ in range(n))
    # two-sided p: how often a sign-flipped resample is at least as extreme
    centred = [d - obs for d in diffs]
    extreme = sum(1 for _ in range(n) if abs(np.mean([centred[rng.randrange(len(centred))] for _ in centred])) >= abs(obs))
    return {"n": len(pairs), "diff": obs, "ci": (boots[int(0.025 * n)], boots[int(0.975 * n) - 1]),
            "p": (extreme + 1) / (n + 1)}


def mcnemar(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test for paired binary outcomes."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p": 1.0}
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b01": b01, "b10": b10, "p": min(1.0, 2 * p)}


def risk_coverage(conf: Sequence[float], correct: Sequence[int]) -> dict:
    """Risk-coverage curve and AURC for selective answering (abstention, E4)."""
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct)[order]
    n = len(c)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan")}
    cov, risk = [], []
    errs = 0
    for i in range(n):
        errs += 1 - c[i]
        cov.append((i + 1) / n)
        risk.append(errs / (i + 1))
    return {"coverage": cov, "risk": risk, "aurc": float(np.mean(risk))}


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")
