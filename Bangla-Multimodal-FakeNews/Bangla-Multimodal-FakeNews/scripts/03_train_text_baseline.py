"""Step 3 (E1): fine-tune BanglaBERT on text only, 3-class. Each seed's result is saved separately.

Run: python scripts/03_train_text_baseline.py
"""
import json  # noqa: E402
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import CHECKPOINT_DIR, FEATURE_DIR, LABELS_3, RESULTS_DIR, SEEDS  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.text_encoder import finetune_text_classifier  # noqa: E402
from src.utils import save_json  # noqa: E402

E1_RUN_DIR = RESULTS_DIR / "e1_runs"


def main():
    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    train = meta[meta["split"] == "train"]
    val = meta[meta["split"] == "val"]
    test = meta[meta["split"] == "test"]
    E1_RUN_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.array(LABELS_3)
    y_test = test["y"].astype(int).to_numpy()
    runs = []
    for seed in SEEDS:
        # A finished seed is loaded from disk, so a disconnect never repeats it.
        run_path = E1_RUN_DIR / f"seed{seed}_metrics.json"
        if run_path.exists():
            m = json.loads(run_path.read_text(encoding="utf-8"))
            print(f"seed {seed}: already finished, loaded from disk")
        else:
            out_dir = CHECKPOINT_DIR / f"text_finetune_seed{seed}"
            pred, probs = finetune_text_classifier(train, val, test, out_dir, seed=seed)
            m = classification_metrics(labels[y_test], labels[pred], LABELS_3)
            # Per-sample predictions are kept for the confusion matrix and error analysis figures.
            pred_df = pd.DataFrame({"sample_id": test["sample_id"].to_numpy(), "y_true": y_test,
                                    "y_pred": pred})
            for i, name in enumerate(LABELS_3):
                pred_df[f"p_{name}"] = probs[:, i]
            pred_df.to_csv(E1_RUN_DIR / f"seed{seed}_predictions.csv", index=False)
            save_json(m, run_path)
        runs.append(m)
        print(f"seed {seed}: accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    acc = [r["accuracy"] for r in runs]
    f1 = [r["macro_f1"] for r in runs]
    summary = {
        "experiment": "E1_banglabert_finetuned_text_only",
        "seeds": SEEDS,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1)),
        "runs": runs,
    }
    save_json(summary, RESULTS_DIR / "e1_text_only.json")
    print("Saved:", RESULTS_DIR / "e1_text_only.json")


if __name__ == "__main__":
    main()
