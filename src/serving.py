"""Inference helpers for the demo app: the validation-selected fusion model, rebuilt from saved files.

Training-time inputs, reproduced exactly:
- text:    BanglaBERT mean-pooled embedding (max length 128) of clean_bangla_text(text).
           USE_OCR is False, so training's model_text is the cleaned text alone.
- image:   CLIP ViT-B/32 image embedding, L2-normalised. A missing image gives a zero vector.
- scaling: mean and std of the TRAIN rows of each block (the rule in src.fusion_model.standardize).
           Checkpoints store head weights only, so these statistics are recomputed from the saved
           feature files at start-up.
- head:    early-fusion CCEF, one head per seed. Probabilities are averaged over the seeds.

numpy is imported at module level. torch, transformers and sentence-transformers are imported
inside functions, so the pure helpers can be tested without a GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocessing import clean_bangla_text

LABELS = ["Authentic", "Misleading", "Satire"]
TEXT_DIM = 768
IMAGE_DIM = 512
MAX_LEN = 128
EXPECTED_WINNER = "fusion:early:banglabert+clip"


def checkpoint_stem(config_name: str) -> str:
    """Same file-name rule as scripts/08_model_selection.py."""
    return config_name.replace(":", "_").replace("+", "_")


def fit_train_stats(block, train_mask):
    """Mean and std from the train rows only. A zero std becomes 1, as in standardize()."""
    train = np.asarray(block, dtype=np.float32)[np.asarray(train_mask, dtype=bool)]
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def apply_stats(block, mu, sd) -> np.ndarray:
    return ((np.asarray(block, dtype=np.float32) - mu) / sd).astype(np.float32)


def build_fused_input(text_vec, image_vec, text_stats, image_stats) -> np.ndarray:
    """Text block first, then image block, each standardised with its own train statistics."""
    t = apply_stats(np.asarray(text_vec, dtype=np.float32).reshape(1, -1), *text_stats)
    i = apply_stats(np.asarray(image_vec, dtype=np.float32).reshape(1, -1), *image_stats)
    return np.concatenate([t, i], axis=1)


def average_probs(prob_list) -> np.ndarray:
    """Equal-weight mean of the per-seed probability vectors."""
    stacked = np.stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in prob_list])
    return stacked.mean(axis=0)


def leaderboard_rows(summary: dict, winner: str | None = None) -> list[dict]:
    """Flatten summary.json (mean and std over seeds) into table rows, grouped by kind."""
    rows = []
    for name, r in summary.items():
        rows.append({
            "config": name,
            "kind": r["kind"],
            "val_F1": round(r["val_macro_f1_mean"], 4),
            "val_F1_std": round(r["val_macro_f1_std"], 4),
            "test_acc": round(r["test_accuracy_mean"], 4),
            "test_F1": round(r["test_macro_f1_mean"], 4),
            "test_F1_std": round(r["test_macro_f1_std"], 4),
            "winner": name == winner,
        })
    return sorted(rows, key=lambda x: (x["kind"], -x["val_F1"]))


def load_heads(ckpt_dir, config_name: str, seeds, input_dim: int) -> list:
    """Load one CCEF head per seed. The checkpoints are float16, so they are cast back to float32."""
    import torch

    from src.fusion_model import CCEF

    stem = checkpoint_stem(config_name)
    heads = []
    for seed in seeds:
        path = Path(ckpt_dir) / f"{stem}_seed{seed}_0.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}. Run scripts/08_model_selection.py first.")
        state = torch.load(path, map_location="cpu")
        model = CCEF(input_dim, len(LABELS))
        model.load_state_dict({k: v.float() for k, v in state.items()})
        model.eval()
        heads.append(model)
    return heads


def head_probs(heads, x: np.ndarray) -> list:
    """Per-head class probabilities for one input row."""
    import torch

    x_t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        return [torch.softmax(h(x_t), dim=1).numpy()[0] for h in heads]


@dataclass
class DemoBundle:
    winner: str
    seeds: list
    text_stats: tuple
    image_stats: tuple
    fused_heads: list
    text_heads: list
    image_heads: list
    summary: dict
    confusion: list


def load_bundle(feature_dir, ckpt_dir, results_dir, seeds) -> DemoBundle:
    """Build everything the app needs. Reads the features only to recompute the train statistics."""
    import pandas as pd

    feature_dir, ckpt_dir, results_dir = Path(feature_dir), Path(ckpt_dir), Path(results_dir)
    selection = results_dir / "selection"
    best = json.loads((selection / "best_config.json").read_text(encoding="utf-8"))
    winner = best["fusion"]
    if winner != EXPECTED_WINNER:
        raise ValueError(
            f"The app is written for {EXPECTED_WINNER!r}, but the validation rule picked {winner!r}. "
            "Update the app before serving a different model."
        )

    meta = pd.read_csv(feature_dir / "meta.csv")
    mask = meta["split"].eq("train").to_numpy()
    bert = np.load(feature_dir / "bert.npy")
    clip = np.load(feature_dir / "clip_img.npy")
    if len(bert) != len(meta) or len(clip) != len(meta):
        raise ValueError("Feature rows do not match meta.csv. Re-run scripts/02_extract_features.py.")

    summary = json.loads((selection / "summary.json").read_text(encoding="utf-8"))
    confusion = summary[winner]["test_confusion_mean"]

    return DemoBundle(
        winner=winner,
        seeds=list(seeds),
        text_stats=fit_train_stats(bert, mask),
        image_stats=fit_train_stats(clip, mask),
        fused_heads=load_heads(ckpt_dir, winner, seeds, TEXT_DIM + IMAGE_DIM),
        text_heads=load_heads(ckpt_dir, "text:banglabert", seeds, TEXT_DIM),
        image_heads=load_heads(ckpt_dir, "image:clip", seeds, IMAGE_DIM),
        summary=summary,
        confusion=confusion,
    )


class TextEncoder:
    """BanglaBERT mean-pooled embeddings, the same pooling as frozen_text_embeddings()."""

    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def count_tokens(self, text: str) -> int:
        """Token count before truncation, so the app can tell the user what the model did not see."""
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    def embed(self, text: str) -> np.ndarray:
        import torch

        enc = self.tok([text], truncation=True, max_length=MAX_LEN, padding=True,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.cpu().numpy()[0].astype(np.float32)


class ImageEncoder:
    """CLIP ViT-B/32 image embeddings, normalised as in src.image_encoder.encode_images()."""

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, pil_image) -> np.ndarray:
        if pil_image is None:
            return np.zeros(IMAGE_DIM, dtype=np.float32)
        emb = self.model.encode([pil_image.convert("RGB")], batch_size=1,
                                normalize_embeddings=True, convert_to_numpy=True)
        return emb[0].astype(np.float32)


def fused_probs_for_text(bundle: DemoBundle, text_enc: TextEncoder, text: str, image_vec) -> tuple:
    """Fused probabilities for raw post text and a FIXED image vector (zeros = no image).
    Used by the explanation tools, which vary the text and keep the image unchanged.
    Returns (mean probabilities, per-seed probabilities)."""
    cleaned = clean_bangla_text(text)
    t_raw = text_enc.embed(cleaned)
    fused_x = build_fused_input(t_raw, np.asarray(image_vec, dtype=np.float32),
                                bundle.text_stats, bundle.image_stats)
    per_seed = [np.asarray(p, dtype=np.float64) for p in head_probs(bundle.fused_heads, fused_x)]
    return average_probs(per_seed), per_seed


def image_only_probs(bundle: DemoBundle, image_vec) -> np.ndarray:
    """Mean probabilities of the trained image-only heads (image:clip), for a post with no text.
    The image block is standardised with the train statistics, as in training."""
    x = apply_stats(np.asarray(image_vec, dtype=np.float32).reshape(1, -1), *bundle.image_stats)
    return average_probs(head_probs(bundle.image_heads, x))


def load_reference(feature_dir) -> dict:
    """TRAIN rows only: normalised BanglaBERT embeddings, labels (0..2) and cleaned post text.
    Used for 'similar known posts'. Validation and test rows are never returned here."""
    import pandas as pd

    feature_dir = Path(feature_dir)
    meta = pd.read_csv(feature_dir / "meta.csv")
    mask = meta["split"].eq("train").to_numpy()
    bert = np.load(feature_dir / "bert.npy").astype(np.float32)
    ref = bert[mask]
    ref = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    sub = meta.loc[mask]
    return {
        "embeddings": ref,
        "labels": sub["y"].astype(int).to_numpy(),
        "texts": sub["model_text"].fillna("").astype(str).tolist(),
    }


def predict_post(bundle: DemoBundle, text_enc: TextEncoder, img_enc: ImageEncoder,
                 text: str, pil_image=None) -> dict:
    """Run the full path for one post. Returns probabilities for the fused model and for each
    modality alone, plus the cleaned text that was actually fed to the model."""
    cleaned = clean_bangla_text(text)
    t_raw = text_enc.embed(cleaned)
    i_raw = img_enc.embed(pil_image)
    has_image = pil_image is not None

    fused_x = build_fused_input(t_raw, i_raw, bundle.text_stats, bundle.image_stats)
    per_seed = head_probs(bundle.fused_heads, fused_x)

    text_only = average_probs(head_probs(
        bundle.text_heads, apply_stats(t_raw.reshape(1, -1), *bundle.text_stats)))
    image_only = None
    if has_image:
        image_only = average_probs(head_probs(
            bundle.image_heads, apply_stats(i_raw.reshape(1, -1), *bundle.image_stats)))

    return {
        "cleaned_text": cleaned,
        "n_chars": len(cleaned),
        "n_tokens": text_enc.count_tokens(cleaned),
        "text_embedding": t_raw,
        "image_embedding": i_raw,
        "has_image": has_image,
        "fused": average_probs(per_seed),
        "fused_per_seed": [np.asarray(p, dtype=np.float64) for p in per_seed],
        "text_only": text_only,
        "image_only": image_only,
    }
