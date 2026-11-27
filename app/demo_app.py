"""Capstone demo: Bangla multimodal fake-news detection (validation-selected model only).

Needs the outputs of: 01 (data), 02 (features), 08 (checkpoints/selection and results/selection).
Run in Colab after those steps, or locally:  python app/demo_app.py

Public link: off by default. Set BFND_PUBLIC_LINK=1 only when you have decided to share it.
The prediction uses the text and the image only. Wikipedia evidence and the cultural features
are not part of it (see the Model card tab for why).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    CLIP_IMAGE_MODEL,
    FEATURE_DIR,
    RESULTS_DIR,
    SEEDS,
    SELECTION_CKPT_DIR,
    TEXT_MODEL,
)
from src.serving import (  # noqa: E402
    LABELS,
    ImageEncoder,
    TextEncoder,
    leaderboard_rows,
    load_bundle,
    predict_post,
)


def pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = pick_device()
BUNDLE = load_bundle(FEATURE_DIR, SELECTION_CKPT_DIR, RESULTS_DIR, SEEDS)
TEXT_ENC = TextEncoder(TEXT_MODEL, DEVICE)
IMG_ENC = ImageEncoder(CLIP_IMAGE_MODEL, DEVICE)
MAX_BATCH_ROWS = 500


def per_class_from_confusion(cm) -> list[dict]:
    """Precision, recall and F1 per class, computed from the mean test confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    rows = []
    for i, label in enumerate(LABELS):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append({
            "class": label,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "F1": round(f1, 3),
            "test support (mean)": round(cm[i].sum(), 1),
        })
    return rows


def predict_tab(image_path, text):
    text = (text or "").strip()
    if not text:
        return None, None, None, "Bangla text likhun."

    from PIL import Image

    pil = Image.open(image_path) if image_path else None
    out = predict_post(BUNDLE, TEXT_ENC, IMG_ENC, text, pil)

    label_scores = {lab: float(p) for lab, p in zip(LABELS, out["fused"])}

    rows = [{"model": "Fused: BanglaBERT + CLIP (3-seed mean)",
             **{lab: round(float(p), 3) for lab, p in zip(LABELS, out["fused"])}},
            {"model": "Text only: BanglaBERT head",
             **{lab: round(float(p), 3) for lab, p in zip(LABELS, out["text_only"])}}]
    if out["image_only"] is not None:
        rows.append({"model": "Image only: CLIP head",
                     **{lab: round(float(p), 3) for lab, p in zip(LABELS, out["image_only"])}})
    comparison = pd.DataFrame(rows)

    seed_rows = []
    for seed, p in zip(BUNDLE.seeds, out["fused_per_seed"]):
        seed_rows.append({"seed": seed, **{lab: round(float(v), 3) for lab, v in zip(LABELS, p)}})
    seeds_df = pd.DataFrame(seed_rows)

    predicted = LABELS[int(np.argmax(out["fused"]))]
    per_seed_pred = [float(p[LABELS.index(predicted)]) for p in out["fused_per_seed"]]
    spread = max(per_seed_pred) - min(per_seed_pred)
    notes = [
        f"**Predicted (fused):** {predicted}, mean probability {out['fused'].max():.3f}.",
        f"**Seed spread for this class:** {spread:.3f} (large values mean the seeds disagree).",
        f"**Input used by the model:** `{out['cleaned_text'][:200]}`"
        + (" …" if len(out['cleaned_text']) > 200 else "") + f"  ({out['n_chars']} characters)",
        "**Image:** " + ("given, used in the fused model." if out["has_image"]
                         else "not given. The image block is a zero vector, as in training."),
        "**Length caution:** in training, Authentic posts are much longer than Misleading posts "
        "(median about 1136 vs 100 characters). A length-only model reaches accuracy 0.48 to 0.57 on test, "
        "so a long or short post can push the prediction for reasons unrelated to its content.",
        "**Not used here:** Wikipedia evidence (its relevance check failed) and cultural features "
        "(disabled, decision D-2). This is a classifier, not a fact-checker.",
    ]
    return label_scores, comparison, seeds_df, "\n\n".join(notes)


def leaderboard_tab():
    table = pd.DataFrame(leaderboard_rows(BUNDLE.summary, BUNDLE.winner))
    info = (
        f"**Winner (validation only):** `{BUNDLE.winner}`.  "
        "Rule: highest mean validation macro-F1 over seeds; ties go to the smaller std; the test set is "
        "reported but never used to choose. Std values use the population formula (ddof=0) over 3 seeds."
    )
    return info, table


def error_tab():
    cm = np.asarray(BUNDLE.confusion, dtype=float)
    cm_df = pd.DataFrame(
        np.round(cm, 1),
        index=[f"true {lab}" for lab in LABELS],
        columns=[f"pred {lab}" for lab in LABELS],
    )
    per_class = pd.DataFrame(per_class_from_confusion(cm))
    note = (
        "Confusion matrix is the mean over 3 seeds on the test set. F1 values here are computed from that "
        "mean matrix, so they can differ slightly from the mean of per-seed F1. Satire has only about 32 test rows: "
        "treat its numbers as unstable."
    )
    return note, cm_df, per_class


def batch_tab(file_path):
    if not file_path:
        return None, "CSV upload korun. `text` column lagbe; `label` column thakle metric ber hobe."
    df = pd.read_csv(file_path)
    if "text" not in df.columns:
        return None, "CSV-te `text` column nei."
    df = df.head(MAX_BATCH_ROWS)
    preds, conf = [], []
    for t in df["text"].fillna("").astype(str):
        out = predict_post(BUNDLE, TEXT_ENC, IMG_ENC, t, None)
        preds.append(LABELS[int(np.argmax(out["fused"]))])
        conf.append(round(float(out["fused"].max()), 3))
    result = df[["text"]].copy()
    result["text"] = result["text"].astype(str).str.slice(0, 80)
    result["prediction"] = preds
    result["confidence"] = conf

    note = (f"Predicted {len(result)} rows (limit {MAX_BATCH_ROWS}). Batch rows have no image, so the image "
            "block is a zero vector; this is a text-only setting for the fused model.")
    if "label" in df.columns:
        from sklearn.metrics import accuracy_score, f1_score

        gold = df["label"].astype(str).head(len(result)).tolist()
        if all(g in LABELS for g in gold):
            acc = accuracy_score(gold, preds)
            f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
            note += f"  **Accuracy {acc:.3f}, macro-F1 {f1:.3f}** on the uploaded labels."
        else:
            note += "  Label column found, but values must be exactly Authentic, Misleading or Satire for metrics."
    return result, note


MODEL_CARD = """
### What the model is
- Task: classify a Bangla post as **Authentic**, **Misleading** or **Satire**.
- Data: BTMD, 3910 rows after cleaning. Clickbait is excluded. Group-aware split: 2736 train, 586 validation, 588 test.
- Model: early fusion of frozen BanglaBERT text embeddings (768) and CLIP ViT-B/32 image embeddings (512), one CCEF MLP head, mean of 3 seeds.
- Selection: chosen by validation macro-F1 only, from the grid in the Leaderboard tab.

### Not part of the prediction
- **Wikipedia evidence:** the relevance check failed (13 of 40 rows judged relevant, pre-set threshold 0.6). It is reported as an ablation, not used here.
- **Cultural features:** disabled after a blind human check failed the agreement rule (decision D-2).
- **E6 decision layer:** exploratory. Its coverage and evidence verdicts have known problems.

### Limitations
- Three seeds and one split. Differences smaller than about 0.02 macro-F1 should not be read as real.
- Satire has about 32 test rows; its scores are unstable.
- Text length is a strong shortcut in this data.
- The app does not run OCR, so text inside an image is ignored.
- The image-only and text-only panels are single-modality heads trained the same way; they explain the fused score, they do not replace it.

### Decision log
See `docs/decisions/DECISIONS.md`: D-1 (evidence source), D-2 (cultural labels), D-3 (selection protocol), D-4 (evidence relevance gate FAIL).
"""


def build_ui():
    with gr.Blocks(title="Bangla Multimodal Fake-News Detection") as demo:
        gr.Markdown("# Bangla Multimodal Fake-News Detection\nFused BanglaBERT + CLIP model, validation-selected.")
        with gr.Tab("Predict"):
            with gr.Row():
                image_in = gr.Image(type="filepath", label="Post image (optional)")
                text_in = gr.Textbox(label="Bangla post text", lines=4, placeholder="এখানে পোস্টের লেখা দিন")
            run = gr.Button("Analyse", variant="primary")
            with gr.Row():
                label_out = gr.Label(label="Fused probabilities")
                with gr.Column():
                    comparison_out = gr.Dataframe(label="Fused vs single-modality")
                    seeds_out = gr.Dataframe(label="Per-seed probabilities")
            notes_out = gr.Markdown()
            run.click(predict_tab, inputs=[image_in, text_in],
                      outputs=[label_out, comparison_out, seeds_out, notes_out])

        with gr.Tab("Leaderboard"):
            lb_info, lb_table = leaderboard_tab()
            gr.Markdown(lb_info)
            gr.Dataframe(value=lb_table, label="All configurations (mean and std over 3 seeds)")

        with gr.Tab("Error analysis"):
            ea_note, cm_df, pc_df = error_tab()
            gr.Markdown(ea_note)
            gr.Dataframe(value=cm_df, label="Confusion matrix (test, mean over seeds)")
            gr.Dataframe(value=pc_df, label="Per-class precision, recall and F1")

        with gr.Tab("Batch CSV"):
            file_in = gr.File(type="filepath", label="CSV with a `text` column (optional `label`)")
            batch_run = gr.Button("Run batch")
            batch_out = gr.Dataframe(label="Predictions")
            batch_note = gr.Markdown()
            batch_run.click(batch_tab, inputs=[file_in], outputs=[batch_out, batch_note])

        with gr.Tab("Model card"):
            gr.Markdown(MODEL_CARD)
    return demo


if __name__ == "__main__":
    build_ui().launch(share=os.environ.get("BFND_PUBLIC_LINK") == "1")
