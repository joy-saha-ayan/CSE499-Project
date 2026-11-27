"""Step 10: diagnostic controls for the two shortcuts in the progress audit.

Question 1 (image presence): does the image content help beyond "this post has an image"?
    Compare the early fusion head with the same head where every image vector is swapped with
    another post's image vector (presence kept, content broken), and with a head that only sees
    the presence flag next to the text.
Question 2 (text length): do the gains hold inside length quartiles, or only overall?
Question 3 (group split): are there near-copies of a train post in the validation or test split?

Decisions:
- Selection stays where it was (script 08, validation only). This script changes no choice.
- Validation numbers are used to read the controls. Test numbers are written to the output for
  completeness and are not used to pick anything.
- Same head, same training rule, same three seeds as script 08 (fit_and_predict in src/fusion_heads.py).

Input : FEATURE_DIR/meta.csv, the text and image feature files used by script 08.
Output: RESULTS_DIR/controls/controls.json and controls.md, and test probabilities in
        RESULTS_DIR/controls/probs/ (used by script 12 with --probs-dir)

Run: python scripts/10_confound_controls.py   (CPU is enough; the heads are small)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    FEATURE_DIR,
    IMAGE_FEATURE_FILES,
    LABELS_3,
    RESULTS_DIR,
    SEEDS,
    TEXT_FEATURE_FILES,
)
from src.confound_controls import (  # noqa: E402
    cross_split_near_duplicates,
    exact_text_overlap,
    length_quartile_labels,
    near_duplicate_pair_table,
    shuffle_images_within_present,
)
from src.fusion_heads import fit_and_predict  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.utils import get_device, save_json, set_seed  # noqa: E402

CONTROL_DIR = RESULTS_DIR / "controls"
NEAR_DUP_THRESHOLD = 0.8


def _metrics(y_true, y_pred):
    names = np.array(LABELS_3)
    m = classification_metrics(names[y_true], names[y_pred], LABELS_3)
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def near_duplicate_audit(meta, split, pair_rows=None):
    texts = meta["model_text"].fillna("").astype(str).to_numpy()
    train_texts = texts[split["train"]]
    report = {"threshold": NEAR_DUP_THRESHOLD, "splits": {}}
    train_global = np.flatnonzero(split["train"])
    sample_ids = meta["sample_id"].to_numpy()
    labels = meta["y"].astype(int).to_numpy()
    for name in ["val", "test"]:
        q = texts[split[name]]
        hits = cross_split_near_duplicates(train_texts, q, threshold=NEAR_DUP_THRESHOLD)
        if pair_rows is not None:
            pair_rows.extend(near_duplicate_pair_table(
                hits, name, np.flatnonzero(split[name]), train_global, sample_ids, labels))
        exact = exact_text_overlap(train_texts, q)
        report["splits"][name] = {
            "n": int(len(q)),
            "exact_match_with_train": int(len(exact)),
            "near_match_with_train": int(len({qi for qi, _, _ in hits})),
            "example_similarities": sorted([round(s, 3) for _, _, s in hits], reverse=True)[:10],
        }
    return report


def main():
    set_seed(42)
    device = get_device()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    y = meta["y"].astype(int).to_numpy()
    split = {k: (meta["split"] == k).to_numpy() for k in ["train", "val", "test"]}
    has_image = meta["has_image"].astype(bool).to_numpy()
    length = np.log1p(meta["model_text"].fillna("").astype(str).str.len().to_numpy(dtype=float))

    bert = np.load(FEATURE_DIR / TEXT_FEATURE_FILES["banglabert"]).astype(np.float32)
    clip = np.load(FEATURE_DIR / IMAGE_FEATURE_FILES["clip"]).astype(np.float32)
    for name, arr in [("bert", bert), ("clip", clip)]:
        if len(arr) != len(meta):
            raise ValueError(f"{name} has {len(arr)} rows, meta has {len(meta)}")

    pair_rows = []  # near-duplicate pairs with both labels, written to near_duplicate_pairs.csv
    presence = has_image.astype(np.float32).reshape(-1, 1)
    length_col = length.reshape(-1, 1).astype(np.float32)

    # Presence is kept, content is broken: shuffle inside each split, zero rows stay zero.
    clip_shuffled, _ = shuffle_images_within_present(clip, has_image, groups=meta["split"].to_numpy(), seed=42)

    configs = {
        "text_banglabert": {"text": bert},
        "presence_only": {"extra": presence},
        "length_only": {"extra": length_col},
        "text_plus_presence": {"text": bert, "extra": presence},
        "fusion_real": {"text": bert, "image": clip},
        "fusion_image_shuffled": {"text": bert, "image": clip_shuffled},
    }

    records = {name: [] for name in configs}
    val_preds = {name: [] for name in configs}
    for name, blocks in configs.items():
        for seed in SEEDS:
            out = fit_and_predict("early", blocks, y, split, seed=seed, device=device)
            va_pred = out["val"].argmax(axis=1)
            te_pred = out["test"].argmax(axis=1)
            records[name].append({
                "seed": seed,
                "val": _metrics(y[split["val"]], va_pred),
                "test": _metrics(y[split["test"]], te_pred),
            })
            val_preds[name].append(va_pred)
            (CONTROL_DIR / "probs").mkdir(parents=True, exist_ok=True)
            np.save(CONTROL_DIR / "probs" / f"{name}_seed{seed}.npy", out["test"])
            print(f"{name:24s} seed {seed}: val macro-F1 {records[name][-1]['val']['macro_f1']:.4f}")

    def mean_val(name):
        vals = [r["val"]["macro_f1"] for r in records[name]]
        return float(np.mean(vals)), float(np.std(vals))

    # Length quartiles on validation rows only.
    va_len = length_quartile_labels(np.expm1(length[split["val"]]), n_bins=4)
    by_quartile = {}
    for name in ["text_banglabert", "text_plus_presence", "fusion_real", "fusion_image_shuffled"]:
        rows = []
        for q in range(4):
            mask = va_len == q
            per_seed = [
                classification_metrics(
                    np.array(LABELS_3)[y[split["val"]][mask]],
                    np.array(LABELS_3)[p[mask]],
                    LABELS_3,
                )["macro_f1"]
                for p in val_preds[name]
            ]
            rows.append({"quartile": q + 1, "n": int(mask.sum()), "macro_f1_mean": float(np.mean(per_seed))})
        by_quartile[name] = rows

    summary = {name: {"val_macro_f1_mean": mean_val(name)[0], "val_macro_f1_std": mean_val(name)[1]}
               for name in configs}
    contrasts = {
        "image_content_beyond_presence (fusion_real - text_plus_presence)":
            summary["fusion_real"]["val_macro_f1_mean"] - summary["text_plus_presence"]["val_macro_f1_mean"],
        "shuffled_image_beyond_presence (fusion_image_shuffled - text_plus_presence)":
            summary["fusion_image_shuffled"]["val_macro_f1_mean"] - summary["text_plus_presence"]["val_macro_f1_mean"],
        "presence_gain_on_text (text_plus_presence - text_banglabert)":
            summary["text_plus_presence"]["val_macro_f1_mean"] - summary["text_banglabert"]["val_macro_f1_mean"],
    }

    import platform
    import torch
    environment = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "feature_meta_rows": int(len(meta)),
    }
    result = {
        "environment": environment,
        "note": "Validation numbers are used to read the controls. Test numbers are reported only.",
        "seeds": SEEDS,
        "summary": summary,
        "contrasts_val_macro_f1": contrasts,
        "length_quartiles_val": by_quartile,
        "runs": records,
        "near_duplicates": near_duplicate_audit(meta, split, pair_rows),
    }
    save_json(result, CONTROL_DIR / "controls.json")
    pd.DataFrame(pair_rows).to_csv(CONTROL_DIR / "near_duplicate_pairs.csv", index=False)

    lines = ["# Confound controls (diagnostic, validation used for reading)", ""]
    lines += ["| Configuration | Val macro-F1 (mean ± std) |", "|---|---|"]
    for name in configs:
        m, s = mean_val(name)
        lines.append(f"| {name} | {m:.4f} ± {s:.4f} |")
    lines += ["", "## Contrasts (validation macro-F1)"]
    lines += [f"- {k}: {v:+.4f}" for k, v in contrasts.items()]
    lines += ["", "## Near-duplicates across splits (threshold {})".format(NEAR_DUP_THRESHOLD)]
    for name, s in result["near_duplicates"]["splits"].items():
        lines.append(f"- {name}: {s['near_match_with_train']} of {s['n']} near-match a train post; "
                     f"{s['exact_match_with_train']} exact")
    (CONTROL_DIR / "controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
