"""Step 8: train every configuration of the selection grid, with 3 seeds, and pick the winners on validation.

Grid (frozen backbones, same head and training rule for all):
- text only:   banglabert, mbert, xlmr                       (3)
- image only:  clip, vit, swin                               (3)
- controls:    length only, has_image only                   (2, never selected)
- fusion:      text x image x {early, intermediate, late}    (27)

Winners are chosen by mean validation macro-F1 over the seeds, separately for text, image
and fusion. The test set is only reported, never used for a choice.

Outputs in RESULTS_DIR/selection:
  runs.json         one record per (configuration, seed): validation and test metrics,
                    confusion matrices, per-class scores, image-subset test scores
  summary.json      mean and std over seeds for every configuration
  summary.md        printed tables used in the notebook
  best_config.json  winners (text, image, fusion) and the rule that picked them
  probs/            test-set probabilities of every run (for later bootstrap tests)
Model heads are saved to CHECKPOINT_DIR/selection as float16 state dicts.

Run: python scripts/08_model_selection.py
"""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from configs.config import (  # noqa: E402
    FEATURE_DIR,
    IMAGE_BACKBONES,
    IMAGE_FEATURE_FILES,
    LABELS_3,
    SEEDS,
    SELECTION_CKPT_DIR,
    SELECTION_DIR,
    TEXT_BACKBONES,
    TEXT_FEATURE_FILES,
)
from src.fusion_heads import fit_and_predict  # noqa: E402
from src.image_presence import image_subset_metrics  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.selection import (  # noqa: E402
    aggregate_over_seeds,
    fusion_configs,
    format_table,
    select_best,
    unimodal_configs,
)
from src.utils import get_device, save_json, set_seed  # noqa: E402

FUSION_KINDS = ["early", "intermediate", "late"]


def build_run(cfg, blocks):
    """Return (fusion kind, blocks dict) for one configuration."""
    if cfg["kind"] == "text":
        return "early", {"text": blocks["text"][cfg["text"]]}
    if cfg["kind"] == "image":
        return "early", {"image": blocks["image"][cfg["image"]]}
    if cfg["kind"] == "control":
        return "early", {"extra": blocks["controls"][cfg["fusion"]]}
    return cfg["fusion"], {"text": blocks["text"][cfg["text"]], "image": blocks["image"][cfg["image"]]}


def metrics_for(y_true, y_pred):
    """y_true and y_pred hold class indices (0, 1, 2). They are mapped to label names first."""
    names = np.array(LABELS_3)
    m = classification_metrics(names[np.asarray(y_true)], names[np.asarray(y_pred)], LABELS_3)
    per = m["per_class"].values()
    m["macro_precision"] = float(np.mean([v["precision"] for v in per]))
    m["macro_recall"] = float(np.mean([v["recall"] for v in per]))
    return m


def main():
    set_seed(42)
    device = get_device()
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (SELECTION_DIR / "probs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    y = meta["y"].astype(int).to_numpy()
    split = {k: (meta["split"] == k).to_numpy() for k in ["train", "val", "test"]}
    has_image = meta["has_image"].astype(bool).to_numpy()
    length = np.log1p(meta["model_text"].fillna("").astype(str).str.len().to_numpy(dtype=float))

    blocks = {
        "text": {k: np.load(FEATURE_DIR / f) for k, f in TEXT_FEATURE_FILES.items()},
        "image": {k: np.load(FEATURE_DIR / f) for k, f in IMAGE_FEATURE_FILES.items()},
        "controls": {
            "length": length.reshape(-1, 1).astype(np.float32),
            "presence": has_image.astype(np.float32).reshape(-1, 1),
        },
    }
    for group in ["text", "image"]:
        for k, arr in blocks[group].items():
            if len(arr) != len(meta):
                raise ValueError(f"{group} '{k}' has {len(arr)} rows, meta has {len(meta)}")

    configs = unimodal_configs(list(TEXT_BACKBONES), list(IMAGE_BACKBONES))
    configs += fusion_configs(list(TEXT_BACKBONES), list(IMAGE_BACKBONES), FUSION_KINDS)
    print(f"Configurations: {len(configs)} x seeds {SEEDS} = {len(configs) * len(SEEDS)} runs")

    records = []
    for idx, (cfg, seed) in enumerate(product(configs, SEEDS), start=1):
        kind, run_blocks = build_run(cfg, blocks)
        out = fit_and_predict(kind, run_blocks, y, split, seed=seed, device=device)

        va, te = split["val"], split["test"]
        val_pred = out["val"].argmax(axis=1)
        test_pred = out["test"].argmax(axis=1)
        mv = metrics_for(y[va], val_pred)
        mt = metrics_for(y[te], test_pred)
        img_sub = image_subset_metrics(
            np.array(LABELS_3)[y[te]], np.array(LABELS_3)[test_pred], has_image[te], LABELS_3
        ) if has_image[te].any() else None

        flat = {
            "val_macro_f1": mv["macro_f1"], "val_accuracy": mv["accuracy"],
            "test_macro_f1": mt["macro_f1"], "test_accuracy": mt["accuracy"],
            "test_macro_precision": mt["macro_precision"], "test_macro_recall": mt["macro_recall"],
        }
        if img_sub is not None:
            flat["test_image_subset_macro_f1"] = img_sub["macro_f1"]
            flat["test_image_subset_accuracy"] = img_sub["accuracy"]
        records.append({
            "name": cfg["name"], "kind": cfg["kind"], "seed": seed, "fusion_kind": kind,
            "metrics": flat,
            "details": {"val": mv, "test": mt, "image_subset": img_sub},
        })

        np.save(SELECTION_DIR / "probs" / f"{cfg['name'].replace(':', '_').replace('+', '_')}_seed{seed}.npy",
                out["test"])
        heads = out["model"] if isinstance(out["model"], list) else [out["model"]]
        for j, head in enumerate(heads):
            state = {k: v.detach().cpu().half() for k, v in head.state_dict().items()}
            fname = f"{cfg['name'].replace(':', '_').replace('+', '_')}_seed{seed}_{j}.pt"
            torch.save(state, SELECTION_CKPT_DIR / fname)
        print(f"[{idx:3d}/{len(configs) * len(SEEDS)}] {cfg['name']:<40} seed {seed}: "
              f"val F1 {mv['macro_f1']:.4f} | test F1 {mt['macro_f1']:.4f} acc {mt['accuracy']:.4f}")

    save_json(records, SELECTION_DIR / "runs.json")

    summary = aggregate_over_seeds(records)
    # Mean confusion matrix on test, per configuration (counts averaged over seeds)
    by_name = {}
    for rec in records:
        by_name.setdefault(rec["name"], []).append(np.array(rec["details"]["test"]["confusion_matrix"], float))
    for name, cms in by_name.items():
        summary[name]["test_confusion_mean"] = np.mean(np.stack(cms), axis=0).tolist()
    save_json(summary, SELECTION_DIR / "summary.json")

    best = {
        "text": select_best(summary, "text"),
        "image": select_best(summary, "image"),
        "fusion": select_best(summary, "fusion"),
        "rule": "highest mean validation macro-F1 over seeds; ties go to smaller std; test never used",
    }
    save_json(best, SELECTION_DIR / "best_config.json")

    text_names = [c["name"] for c in configs if c["kind"] == "text"]
    image_names = [c["name"] for c in configs if c["kind"] == "image"]
    control_names = [c["name"] for c in configs if c["kind"] == "control"]
    fusion_names = sorted([c["name"] for c in configs if c["kind"] == "fusion"],
                          key=lambda n: -summary[n]["val_macro_f1_mean"])
    md = "\n\n".join([
        "## Text only\n```\n" + format_table(summary, text_names) + "\n```",
        "## Image only\n```\n" + format_table(summary, image_names) + "\n```",
        "## Controls (not eligible for selection)\n```\n" + format_table(summary, control_names) + "\n```",
        "## Fusion (sorted by validation macro-F1)\n```\n" + format_table(summary, fusion_names) + "\n```",
        f"## Winners\n{best}",
    ])
    (SELECTION_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("\nWinners (validation only):", best)


if __name__ == "__main__":
    main()
