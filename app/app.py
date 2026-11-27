"""Gradio demo: meme image + Bangla text -> verdict and explanation.

Prerequisite: steps 1-4 have been run (checkpoints/e5_artifacts.joblib and
checkpoints/cultural_models.joblib must exist).
Run: python app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from configs.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLIP_IMAGE_MODEL,
    CLIP_TEXT_MODEL,
    EMBED_MODEL,
    FEATURE_DIR,
    HYBRID_ALPHA,
    LABELS_3,
    NLI_MODEL,
    NLI_THRESHOLD,
    TOP_K,
)
from src.decision import final_decision  # noqa: E402
from src.evidence_retrieval import EvidenceRetriever  # noqa: E402
from src.evidence_verification import NLIVerifier, aggregate_evidence, verdict_from_scores  # noqa: E402
from src.explainability import build_explanation  # noqa: E402
from src.fusion_model import CCEF, predict_proba  # noqa: E402
from src.image_encoder import encode_images, encode_texts, image_text_similarity, load_clip  # noqa: E402
from src.text_encoder import frozen_text_embeddings  # noqa: E402
from src.utils import get_device  # noqa: E402

DEVICE = get_device()
ART = joblib.load(CHECKPOINT_DIR / "e5_artifacts.joblib")
CULT_MODELS = joblib.load(CHECKPOINT_DIR / "cultural_models.joblib")
EV = pd.read_csv(FEATURE_DIR / "evidence_clean.csv")
EV_TEXTS = EV["evidence_text"].astype(str).tolist()
RETRIEVER = EvidenceRetriever(EV, EMBED_MODEL, DEVICE)
VERIFIER = NLIVerifier(NLI_MODEL, DEVICE)
CLIP_IMG = load_clip(CLIP_IMAGE_MODEL, DEVICE)
CLIP_TXT = load_clip(CLIP_TEXT_MODEL, DEVICE)
MODEL = CCEF(ART["input_dim"], len(LABELS_3)).to(DEVICE)
MODEL.load_state_dict(ART["state_dict"])
MODEL.eval()


def predict(image_path, text):
    text = (text or "").strip()
    if not text:
        return "Please enter Bangla text", ""

    bert = frozen_text_embeddings([text])
    img = encode_images(CLIP_IMG, [image_path]) if image_path else np.zeros((1, 512), dtype=np.float32)
    clip_txt = encode_texts(CLIP_TXT, [text])
    feats = {
        "bert": bert,
        "clip_img": img,
        "clip_txt": clip_txt,
        "sim": image_text_similarity(img, clip_txt).reshape(-1, 1),
        "cultural": np.column_stack([m.predict_proba(bert)[:, 1] for m in CULT_MODELS]),
    }

    hits = RETRIEVER.retrieve([text], k=TOP_K, alpha=HYBRID_ALPHA)[0]
    premises = [EV_TEXTS[j] for j, _ in hits]
    nli = VERIFIER.probs(premises, [text] * len(premises))
    feats["evidence"] = aggregate_evidence(nli).reshape(1, 3)

    X = np.hstack([feats[b] for b in ART["blocks"]])
    Xs = (X - ART["mu"]) / ART["sd"]
    proba = predict_proba(MODEL, Xs, DEVICE)[0]
    cls_label = LABELS_3[int(proba.argmax())]
    conf = float(proba.max())
    verdict = verdict_from_scores(feats["evidence"][0, 0], feats["evidence"][0, 1], NLI_THRESHOLD)
    decision = final_decision(cls_label, verdict, conf, ART["min_confidence"])

    evidences = [
        {
            "source": EV.iloc[j].get("source", "unknown"),
            "text": EV_TEXTS[j],
            "entail": float(p[0]),
            "contra": float(p[1]),
        }
        for (j, _), p in zip(hits, nli)
    ]
    report = build_explanation(text, "", cls_label, conf, verdict, decision, evidences)
    return decision, report


with gr.Blocks() as demo:
    gr.Markdown("# Bangla Multimodal Fake News Detection (prototype)")
    with gr.Row():
        image = gr.Image(type="filepath", label="Meme image (optional)")
        text = gr.Textbox(label="Bangla text / caption", lines=3)
    button = gr.Button("Analyze")
    verdict_box = gr.Textbox(label="Final verdict")
    explanation = gr.Markdown()
    button.click(predict, inputs=[image, text], outputs=[verdict_box, explanation])


if __name__ == "__main__":
    demo.launch(share=True)
