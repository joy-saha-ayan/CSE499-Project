"""Model-selection grid: run list, aggregation over seeds, winner rule and printed tables.

Selection rule (fixed before any run, so the test set is never used to choose):
- The winner of each stage is the configuration with the highest validation macro-F1,
  averaged over the seeds. Ties go to the smaller seed standard deviation.
- Accuracy and recall are shown for every configuration, but they never pick the winner.
- Test metrics are reported for every configuration, and the winner is never chosen from them.

Pure Python and numpy: no torch import here, so it runs in the unit tests without a GPU.
"""
from itertools import product

import numpy as np


def unimodal_configs(text_names, image_names):
    """Text-only and image-only configurations, plus the two controls."""
    configs = [{"name": f"text:{t}", "kind": "text", "text": t, "image": None, "fusion": None}
               for t in text_names]
    configs += [{"name": f"image:{i}", "kind": "image", "text": None, "image": i, "fusion": None}
                for i in image_names]
    configs += [
        {"name": "control:length_only", "kind": "control", "text": None, "image": None, "fusion": "length"},
        {"name": "control:has_image_only", "kind": "control", "text": None, "image": None, "fusion": "presence"},
    ]
    return configs


def fusion_configs(text_names, image_names, fusion_kinds):
    """Every text x image x fusion combination."""
    return [
        {"name": f"fusion:{f}:{t}+{i}", "kind": "fusion", "text": t, "image": i, "fusion": f}
        for t, i, f in product(text_names, image_names, fusion_kinds)
    ]


def aggregate_over_seeds(records):
    """records: one dict per (config, seed) with metric values. Returns {name: summary dict}.

    Each summary holds the mean and the standard deviation (ddof=0) of every numeric metric.
    """
    by_name = {}
    for rec in records:
        by_name.setdefault(rec["name"], []).append(rec)
    summary = {}
    for name, recs in by_name.items():
        metric_keys = [k for k, v in recs[0]["metrics"].items() if isinstance(v, float)]
        row = {"name": name, "kind": recs[0]["kind"], "seeds": sorted(r["seed"] for r in recs)}
        for k in metric_keys:
            vals = np.array([r["metrics"][k] for r in recs], dtype=float)
            row[k + "_mean"] = float(vals.mean())
            row[k + "_std"] = float(vals.std())
        summary[name] = row
    return summary


def select_best(summary, kind, metric="val_macro_f1"):
    """Return the name of the best configuration of one kind, or None if there is none.

    Highest mean validation metric wins. A tie goes to the smaller standard deviation.
    """
    rows = [r for r in summary.values() if r["kind"] == kind]
    if not rows:
        return None
    rows.sort(key=lambda r: (-r[metric + "_mean"], r[metric + "_std"], r["name"]))
    return rows[0]["name"]


def format_table(summary, names, columns=None):
    """Markdown-like text table. Rows follow `names`. Values are mean +- std over seeds."""
    columns = columns or [
        ("val_macro_f1", "Val F1"),
        ("test_accuracy", "Test Acc"),
        ("test_macro_precision", "Test P"),
        ("test_macro_recall", "Test R"),
        ("test_macro_f1", "Test F1"),
    ]
    header = f"{'config':<42}" + "".join(f"{label:>18}" for _, label in columns)
    lines = [header, "-" * len(header)]
    for name in names:
        row = summary[name]
        cells = []
        for key, _ in columns:
            mean = row.get(key + "_mean")
            std = row.get(key + "_std")
            cells.append("n/a" if mean is None else f"{mean:.4f}+-{std:.4f}")
        lines.append(f"{name:<42}" + "".join(f"{c:>18}" for c in cells))
    return "\n".join(lines)


def confusion_text(cm, labels) -> str:
    """Plain-text confusion matrix: rows are true labels, columns are predicted labels."""
    cm = np.asarray(cm)
    width = max(12, max(len(l) for l in labels) + 2)
    header = "true \\ pred".ljust(width) + "".join(l.rjust(width) for l in labels)
    lines = [header]
    for i, lab in enumerate(labels):
        cells = [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in cm[i]]
        lines.append(lab.ljust(width) + "".join(c.rjust(width) for c in cells))
    return "\n".join(lines)


def mean_confusion(cms):
    """Average confusion-matrix counts over seeds (float, shown with one decimal)."""
    return np.mean(np.stack([np.asarray(c, dtype=float) for c in cms]), axis=0)
