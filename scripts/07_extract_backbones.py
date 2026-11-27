"""Step 7: frozen embeddings for every backbone in the selection grid.

Existing files are reused: bert.npy (BanglaBERT text) and clip_img.npy (CLIP image).
Missing ones are computed and saved to FEATURE_DIR: mBERT and XLM-R text, ViT and Swin image.
Rows without an image get zero vectors, and the script checks that the zero rows are
exactly the rows without an image (same check as script 02).

Run: python scripts/07_extract_backbones.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    FEATURE_DIR,
    IMAGE_BACKBONES,
    IMAGE_FEATURE_FILES,
    MAX_LEN,
    PROCESSED_DIR,
    TEXT_BACKBONES,
    TEXT_FEATURE_FILES,
)
from src.utils import get_device  # noqa: E402


def main():
    device = get_device()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    # meta.csv is written by script 02 in the same row order as the saved feature files.
    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    texts = meta["model_text"].fillna("").astype(str).tolist()
    has_image = meta["has_image"].astype(bool).to_numpy()
    unified = pd.read_csv(PROCESSED_DIR / "unified_split.csv")
    if len(unified) != len(meta) or not (unified["sample_id"].astype(str).values == meta["sample_id"].astype(str).values).all():
        raise ValueError("meta.csv and unified_split.csv are not in the same row order; rerun script 02")
    img_paths = [p if isinstance(p, str) else None for p in unified["image_path"]]

    from src.backbones import image_embeddings, text_embeddings

    for key, model_id in TEXT_BACKBONES.items():
        path = FEATURE_DIR / TEXT_FEATURE_FILES[key]
        if path.exists():
            print(f"[REUSE] text {key}: {path.name} {np.load(path).shape}")
            continue
        emb = text_embeddings(model_id, texts, max_len=MAX_LEN, device=device)
        np.save(path, emb)
        print(f"[NEW]   text {key} ({model_id}): {emb.shape}")

    for key, model_id in IMAGE_BACKBONES.items():
        path = FEATURE_DIR / IMAGE_FEATURE_FILES[key]
        if path.exists():
            emb = np.load(path)
            print(f"[REUSE] image {key}: {path.name} {emb.shape}")
        else:
            emb = image_embeddings(model_id, img_paths, device=device)
            np.save(path, emb)
            print(f"[NEW]   image {key} ({model_id}): {emb.shape}")
        zero = np.all(emb == 0, axis=1)
        ok = np.array_equal(zero, ~has_image)
        print(f"        zero rows == rows without image: {ok} ({int(zero.sum())} zero rows)")
        if not ok:
            raise ValueError(f"image backbone '{key}': zero rows do not match the metadata")


if __name__ == "__main__":
    main()
