"""Explanation and sanity-check helpers for the demo app.

Only numpy and the standard library are used, so every function here can be tested without a GPU.

What the helpers measure (and what they do not):
- occlusion_attribution: remove one chunk of words, re-run the fused model, and record how much the
  probability of the target class moves. This is a sensitivity measure of THIS model. It is not a
  causal proof, and it does not tell whether the post is true.
- truncation_curve: predict from the first k% of the words. A large change means the prediction
  depends on the later part of the post.
- seed_agreement: do the 3 trained heads agree? Disagreement means the prediction is less stable.
- nearest_neighbours: the most similar TRAIN posts by cosine similarity of BanglaBERT embeddings.
  Only train rows are passed in by the app, so no test or validation post is shown as a neighbour.
- input_checks: warnings about the input (script, length, truncation, missing image).
"""
from __future__ import annotations

import html
import math
import re

import numpy as np

_WS = re.compile(r"\s+")
_BN_START, _BN_END = 0x0980, 0x09FF

LABEL_MEANING = {
    "Authentic": "Authentic: BTMD-এর 'Real' class। সঠিক/বিশ্বাসযোগ্য সংবাদ হিসেবে ধরা হয়েছে।",
    "Misleading": "Misleading: BTMD-এর 'Misinformation' class। fake বা বিভ্রান্তিকর হিসেবে ধরা হয়েছে।",
    "Satire": "Satire: ব্যঙ্গ-রচনা হিসেবে ধরা হয়েছে।",
}


def split_units(text: str) -> list[str]:
    """Whitespace tokens. Bangla punctuation stays attached to its word."""
    return [t for t in _WS.split((text or "").strip()) if t]


def chunk_spans(n_units: int, max_units: int) -> list[tuple[int, int]]:
    """Split n_units word positions into at most max_units contiguous spans (end is exclusive)."""
    if max_units < 1:
        raise ValueError("max_units must be at least 1")
    if n_units <= 0:
        return []
    size = max(1, math.ceil(n_units / max_units))
    return [(s, min(s + size, n_units)) for s in range(0, n_units, size)]


def occlusion_attribution(text: str, prob_fn, target_index: int, max_units: int = 30) -> dict:
    """Leave-one-span-out sensitivity.

    prob_fn(text) must return the class-probability vector for a text string.
    delta = p(target | full text) - p(target | span removed).
    delta > 0: the span pushes the prediction TOWARDS the target class.
    delta < 0: the span pushes it AWAY from the target class.
    """
    units = split_units(text)
    if not units:
        raise ValueError("occlusion needs at least one word")
    full = float(np.asarray(prob_fn(" ".join(units)), dtype=float)[target_index])
    spans = []
    for start, end in chunk_spans(len(units), max_units):
        kept = units[:start] + units[end:]
        p = float(np.asarray(prob_fn(" ".join(kept)), dtype=float)[target_index])
        spans.append({"text": " ".join(units[start:end]), "start": start, "end": end,
                      "delta": full - p})
    return {"full_prob": full, "spans": spans}


def top_drivers(spans: list[dict], k: int = 5) -> tuple[list[dict], list[dict]]:
    """The k spans that most support the target, and the k that most oppose it."""
    support = sorted([s for s in spans if s["delta"] > 0], key=lambda s: -s["delta"])[:k]
    oppose = sorted([s for s in spans if s["delta"] < 0], key=lambda s: s["delta"])[:k]
    return support, oppose


def highlight_html(spans: list[dict]) -> str:
    """Post text with each span shaded: green = supports the target, red = opposes it.
    Text is HTML-escaped, so a post cannot inject markup into the page."""
    if not spans:
        return ""
    scale = max(abs(s["delta"]) for s in spans) or 1.0
    parts = []
    for s in spans:
        d = s["delta"]
        if abs(d) < 1e-6:
            style = ""
        else:
            alpha = min(0.85, abs(d) / scale * 0.85)
            colour = f"rgba(22,163,74,{alpha:.2f})" if d > 0 else f"rgba(220,38,38,{alpha:.2f})"
            style = f' style="background:{colour};padding:1px 3px;border-radius:4px"'
        parts.append(f'<span title="change in target probability: {d:+.3f}"{style}>'
                     f"{html.escape(s['text'])}</span>")
    return '<div style="line-height:2.1;font-size:1.05em">' + " ".join(parts) + "</div>"


def truncation_curve(text: str, prob_fn, fractions=(0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """Probabilities when only the first fraction of the words is used."""
    units = split_units(text)
    if not units:
        raise ValueError("truncation check needs at least one word")
    rows = []
    for f in fractions:
        k = max(1, math.ceil(len(units) * f))
        probs = np.asarray(prob_fn(" ".join(units[:k])), dtype=float)
        rows.append({"fraction": f, "words": k, "probs": probs})
    return rows


def seed_agreement(per_seed_probs) -> dict:
    """Majority vote over the seeds and the spread of the majority class probability."""
    stacked = np.stack([np.asarray(p, dtype=float).reshape(-1) for p in per_seed_probs])
    preds = stacked.argmax(axis=1)
    majority = int(np.bincount(preds, minlength=stacked.shape[1]).argmax())
    agreement = float(np.mean(preds == majority))
    spread = float(stacked[:, majority].max() - stacked[:, majority].min())
    return {"preds": preds.tolist(), "majority": majority, "agreement": agreement,
            "spread_majority_prob": spread}


def nearest_neighbours(query, ref, k: int = 5) -> list[tuple[int, float]]:
    """Row indices of ref with the highest cosine similarity to query, most similar first."""
    q = np.asarray(query, dtype=float).reshape(-1)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    R = np.asarray(ref, dtype=float)
    Rn = R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    sims = Rn @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def bangla_ratio(text: str) -> float:
    """Share of letters that are in the Bengali Unicode block. Digits and punctuation are ignored."""
    letters = [c for c in (text or "") if c.isalpha() or _BN_START <= ord(c) <= _BN_END]
    if not letters:
        return 0.0
    bn = sum(1 for c in letters if _BN_START <= ord(c) <= _BN_END)
    return bn / len(letters)


def input_checks(text: str, has_image: bool, n_tokens: int | None = None,
                 max_tokens: int = 128) -> list[tuple[str, str]]:
    """Return (level, message) pairs. level is 'warn' or 'info'."""
    msgs = []
    ratio = bangla_ratio(text)
    if ratio < 0.5:
        msgs.append(("warn", f"Bangla script-এর অনুপাত {ratio:.0%}। model মূলত Bangla post-এ train হয়েছে, "
                             "তাই English-heavy বা মিশ্র input-এ ফল কম নির্ভরযোগ্য হতে পারে।"))
    n_chars = len((text or "").strip())
    if n_chars < 40:
        msgs.append(("warn", "Post খুব ছোট। training-এ Misleading post-এর median দৈর্ঘ্য মাত্র ১০০ character, "
                             "তাই ছোট post-এ length-এর প্রভাব বেশি হতে পারে।"))
    if n_tokens is not None and n_tokens > max_tokens:
        msgs.append(("info", f"Post-এ {n_tokens}টি token আছে, model শুধু প্রথম {max_tokens}টি দেখে। "
                             "শেষের অংশ prediction-এ নেই।"))
    if not has_image:
        msgs.append(("info", "কোনো ছবি দেওয়া হয়নি। Image block zero vector হিসেবে ধরা হয়েছে (training-এও "
                             "১৫১৩টি row-এ ছবি ছিল না)। Prediction তখন প্রায় শুধু text-এর উপর নির্ভর করে।"))
    return msgs


def input_mode(text: str, has_image: bool) -> str:
    """Which model path the app uses.
    'text_and_image' and 'text_only' -> fused model (text is always required there).
    'image_only' -> trained CLIP image head. The fused model never saw an empty text block.
    'empty' -> nothing to predict from."""
    has_text = bool((text or "").strip())
    if has_text and has_image:
        return "text_and_image"
    if has_text:
        return "text_only"
    if has_image:
        return "image_only"
    return "empty"


def confidence_band(p: float) -> str:
    """Display band only. These cut-offs are not calibrated probabilities."""
    if p >= 0.80:
        return "বেশি"
    if p >= 0.60:
        return "মাঝারি"
    return "কম"


def accuracy_macro_f1(y_true, y_pred, n_classes: int = 3) -> tuple[float, float]:
    """Accuracy and macro-F1. Macro-F1 averages only the classes that appear in y_true or y_pred,
    the same convention as sklearn. A class absent from the batch is not counted as F1 = 0."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    f1s = []
    for c in range(n_classes):
        if not (np.any(y_true == c) or np.any(y_pred == c)):
            continue
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return acc, float(np.mean(f1s)) if f1s else 0.0


def per_class_from_confusion(cm, labels) -> list[dict]:
    """Precision, recall and F1 per class, from a confusion matrix (rows = true, columns = predicted)."""
    cm = np.asarray(cm, dtype=float)
    rows = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({"class": label, "precision": round(float(precision), 3),
                     "recall": round(float(recall), 3), "F1": round(float(f1), 3),
                     "test support (mean over seeds)": round(float(cm[i].sum()), 1)})
    return rows
