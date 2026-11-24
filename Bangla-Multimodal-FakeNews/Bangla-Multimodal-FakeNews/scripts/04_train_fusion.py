"""Step 4: fusion experiments, decision layer (E6) and evidence verification check.

Experiments (same MLP, only the feature subset changes; each seed is a separate run):
  E0   image presence flag only (control: how much does "has an image" alone predict?)
  E1f  frozen BanglaBERT only
  E2   CLIP image only
  E3   BanglaBERT + CLIP image + CLIP text + similarity
  E4   E3 + cultural (only when USE_CULTURAL_FEATURES is True; off since decision D-2)
  E5   E3 + evidence (NLI), plus cultural only when USE_CULTURAL_FEATURES is True
  E6   the E5 classifier + four-class decision layer (rule-based, threshold chosen on validation)

Run: python scripts/04_train_fusion.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    ANNOTATION_DIR,
    CHECKPOINT_DIR,
    FEATURE_DIR,
    FUSION_EPOCHS,
    FUSION_LR,
    FUSION_PATIENCE,
    LABELS_3,
    NLI_MODEL,
    NLI_THRESHOLD,
    RESULTS_DIR,
    SEED,
    SEEDS,
    TARGET_COVERAGE,
)
from src.decision import choose_min_confidence, final_decision  # noqa: E402
from src.evidence_verification import NLIVerifier, verdict_from_scores  # noqa: E402
from src.fusion_model import predict_proba, standardize, train_ccef  # noqa: E402
from src.image_presence import image_subset_metrics  # noqa: E402
from src.metrics import classification_metrics, expected_calibration_error  # noqa: E402
from src.utils import get_device, save_json, set_seed  # noqa: E402
from src.experiments import experiment_definitions  # noqa: E402
from configs.config import USE_CULTURAL_FEATURES  # noqa: E402

EXPERIMENTS = experiment_definitions(USE_CULTURAL_FEATURES)


def load_block(name):
    path = FEATURE_DIR / f"{name}.npy"
    return np.load(path) if path.exists() else None


def build_X(blocks):
    arrays = []
    for b in blocks:
        arr = load_block(b)
        if arr is None:
            return None
        arrays.append(arr)
    return np.hstack(arrays)


def summarize(runs, blocks):
    acc = [r["accuracy"] for r in runs]
    f1 = [r["macro_f1"] for r in runs]
    ece = [r["ece"] for r in runs]
    sub_f1 = [r["image_subset"]["macro_f1"] for r in runs]
    sub_acc = [r["image_subset"]["accuracy"] for r in runs]
    return {
        "features": blocks,
        "seeds": SEEDS,
        "accuracy_mean": float(np.mean(acc)),
        "accuracy_std": float(np.std(acc)),
        "macro_f1_mean": float(np.mean(f1)),
        "macro_f1_std": float(np.std(f1)),
        "ece_mean": float(np.mean(ece)),
        "image_subset_n": runs[0]["image_subset"]["n"],
        "image_subset_accuracy_mean": float(np.mean(sub_acc)),
        "image_subset_macro_f1_mean": float(np.mean(sub_f1)),
        "image_subset_macro_f1_std": float(np.std(sub_f1)),
        "runs": runs,
    }


def decision_layer_report(cache, evid_all, y, te):
    """Choose min_confidence on validation, then run the four-class decision on the test set.

    'Insufficient' has no gold label, so its accuracy cannot be measured. Only coverage and
    selective accuracy (accuracy on the answered samples) are reported.
    """
    min_conf = choose_min_confidence(cache["val_proba"].max(axis=1), TARGET_COVERAGE)
    evid = evid_all[te]
    cls_pred = cache["test_proba"].argmax(axis=1)
    cls_conf = cache["test_proba"].max(axis=1)
    verdicts = [verdict_from_scores(e, c, NLI_THRESHOLD) for e, c in zip(evid[:, 0], evid[:, 1])]
    finals = [
        final_decision(LABELS_3[p], v, c, min_conf)
        for p, v, c in zip(cls_pred, verdicts, cls_conf)
    ]
    answered = np.array([f != "Insufficient" for f in finals])
    y_true = np.array(LABELS_3)[y[te]]
    correct = np.array(finals)[answered] == y_true[answered]
    return {
        "min_confidence_from_validation": float(min_conf),
        "target_coverage": TARGET_COVERAGE,
        "final_label_counts_test": {k: int(v) for k, v in pd.Series(finals).value_counts().items()},
        "evidence_verdict_counts_test": {k: int(v) for k, v in pd.Series(verdicts).value_counts().items()},
        "coverage_test": float(answered.mean()),
        "selective_accuracy_test": float(correct.mean()) if answered.any() else None,
        "note": "'Insufficient' has no gold label and is not counted in accuracy.",
    }


def evidence_verification_report(device):
    """If manually annotated claim-evidence pairs exist, measure NLI verification on them."""
    path = ANNOTATION_DIR / "claim_evidence_pairs.csv"
    if not path.exists():
        return {"status": "skipped", "reason": f"{path.name} not found; manually annotated pairs are needed"}
    from src.metrics import classification_metrics as cm

    pairs = pd.read_csv(path)
    labels = ["entailment", "contradiction", "neutral"]
    gold = pairs["relation"].astype(str).str.strip().str.lower()
    bad = set(gold) - set(labels)
    if bad:
        raise ValueError(f"Unknown values in 'relation': {bad}")
    verifier = NLIVerifier(NLI_MODEL, device)
    probs = verifier.probs(pairs["evidence"].astype(str).tolist(), pairs["claim"].astype(str).tolist())
    pred = np.array(labels)[probs.argmax(axis=1)]
    return {"status": "ok", "n_pairs": int(len(pairs)), **cm(gold.to_numpy(), pred, labels)}


def write_summary(results, path):
    lines = [
        "| Experiment | Features | Accuracy (mean ± std) | Macro-F1 (mean ± std) |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        if "accuracy_mean" not in r:
            continue
        feats = ", ".join(r.get("features", ["text (fine-tuned)"]))
        lines.append(
            f"| {name} | {feats} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
            f"| {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    set_seed(SEED)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    y = meta["y"].to_numpy().astype(int)
    tr = (meta["split"] == "train").to_numpy()
    va = (meta["split"] == "val").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    has_image = meta["has_image"].astype(bool).to_numpy()
    test_has_image = has_image[te]  # image-present test rows, for the image subset metric

    results = {}
    e1_path = RESULTS_DIR / "e1_text_only.json"
    if e1_path.exists():
        results["E1_banglabert_finetuned"] = json.loads(e1_path.read_text(encoding="utf-8"))

    cache, artifacts = {}, None
    for name, blocks in EXPERIMENTS:
        X = build_X(blocks)
        if X is None:
            print(f"[SKIP] {name}: missing features {blocks}")
            continue
        runs = []
        for seed in SEEDS:
            set_seed(seed)
            (Xtr, Xva, Xte), mu, sd = standardize(X[tr], X[va], X[te])
            model, val_f1 = train_ccef(
                Xtr, y[tr], Xva, y[va],
                epochs=FUSION_EPOCHS, lr=FUSION_LR, patience=FUSION_PATIENCE,
                seed=seed, device=device,
            )
            proba_te = predict_proba(model, Xte, device)
            pred = proba_te.argmax(axis=1)
            m = classification_metrics(
                np.array(LABELS_3)[y[te]], np.array(LABELS_3)[pred], LABELS_3
            )
            m["ece"] = expected_calibration_error(proba_te.max(axis=1), pred == y[te])
            m["val_macro_f1"] = val_f1
            # Same predictions, restricted to posts that have an image. Image models are
            # judged here, because a post without an image carries no image signal.
            m["image_subset"] = image_subset_metrics(
                np.array(LABELS_3)[y[te]], np.array(LABELS_3)[pred], test_has_image, LABELS_3
            )
            runs.append(m)
            print(f"{name} seed={seed}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}")

            if name == "E5_plus_evidence" and seed == SEEDS[0]:
                cache = {"val_proba": predict_proba(model, Xva, device), "test_proba": proba_te}
                artifacts = {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "input_dim": int(X.shape[1]),
                    "blocks": blocks,
                    "mu": mu,
                    "sd": sd,
                    "min_confidence": None,
                }
        results[name] = summarize(runs, blocks)

    evid_path = FEATURE_DIR / "evidence.npy"
    if cache and evid_path.exists():
        report = decision_layer_report(cache, np.load(evid_path), y, te)
        results["E6_full_decision_layer"] = report
        artifacts["min_confidence"] = report["min_confidence_from_validation"]
        joblib.dump(artifacts, CHECKPOINT_DIR / "e5_artifacts.joblib")
        print("E6:", report)

    results["evidence_verification"] = evidence_verification_report(device)

    save_json(results, RESULTS_DIR / "fusion_results.json")
    write_summary(results, RESULTS_DIR / "summary.md")
    print("\n" + (RESULTS_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
