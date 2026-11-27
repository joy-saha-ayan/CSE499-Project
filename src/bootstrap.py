"""Paired bootstrap confidence intervals for macro-F1 (pure numpy, no torch).

Two models are scored on the same test rows. Rows are resampled with replacement, and both models
are scored on the same resample. This keeps the paired structure, so the interval is for the
difference, which is the quantity that matters when comparing fusion with text-only.

Macro-F1 here is computed from a confusion matrix over a fixed label set, so a class that is absent
from a resample still counts as F1 = 0 for that resample, the same convention as scikit-learn with
zero_division=0.
"""
import numpy as np


def macro_f1_np(y_true, y_pred, n_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(cm, (y_true, y_pred), 1.0)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def f1_class_np(y_true, y_pred, n_classes: int, cls: int) -> float:
    """F1 of one class (one-vs-rest), 0 when the class has no true or predicted rows."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _score(y, p, n_classes, cls):
    return macro_f1_np(y, p, n_classes) if cls is None else f1_class_np(y, p, n_classes, cls)


def paired_bootstrap_macro_f1(y_true, pred_a, pred_b, n_classes: int, n_boot: int = 2000,
                              seed: int = 42, alpha: float = 0.05, cls: int | None = None) -> dict:
    """Return the scores of A and B, the difference B - A, its CI and P(B > A) over resamples.

    cls=None scores macro-F1. An integer cls scores the one-vs-rest F1 of that class instead
    (for example Satire, the smallest class).
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("labels and both prediction vectors must have the same length")
    if len(y) == 0:
        raise ValueError("empty test set")

    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = _score(y[idx], b[idx], n_classes, cls) - _score(y[idx], a[idx], n_classes, cls)

    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(n),
        "metric": "macro_f1" if cls is None else f"f1_class_{cls}",
        "score_a": _score(y, a, n_classes, cls),
        "score_b": _score(y, b, n_classes, cls),
        "diff_b_minus_a": _score(y, b, n_classes, cls) - _score(y, a, n_classes, cls),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_b_better": float(np.mean(diffs > 0)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
