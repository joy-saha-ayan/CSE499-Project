"""Consistency check: does the app path reproduce the training-time outputs?

Run on Colab after notebooks 02 and 08, with a GPU:
    !python app/check_app_consistency.py

It checks the first N test rows in three steps, so a failure points to the stage that broke:
  1. Text:   BanglaBERT embedding of model_text (app path)   vs  features/bert.npy
  2. Image:  CLIP embedding of the image (app path)           vs  features/clip_img.npy
  3. Output: fused probabilities of each seed (app path)      vs  results/selection/probs/<winner>_seed<s>.npy

Pass rule: text and image max |diff| < 1e-3, probability max |diff| < 1e-2, and no predicted-label
mismatch. Probability differences are expected to be small but not zero, because the app loads the
heads from float16 checkpoints, while the saved probabilities come from the float32 models.

This script does not change any file. It only reads the features, the checkpoints and the probabilities.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from configs.config import (  # noqa: E402
    CLIP_IMAGE_MODEL,
    FEATURE_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEEDS,
    SELECTION_CKPT_DIR,
    SELECTION_DIR,
    TEXT_MODEL,
)
from src.serving import (  # noqa: E402
    ImageEncoder,
    TextEncoder,
    checkpoint_stem,
    fused_probs_for_text,
    load_bundle,
)

N_ROWS = 20
TEXT_TOL = 1e-3
IMAGE_TOL = 1e-3
PROB_TOL = 1e-2


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_bundle(FEATURE_DIR, SELECTION_CKPT_DIR, RESULTS_DIR, SEEDS)

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    unified = pd.read_csv(PROCESSED_DIR / "unified.csv", usecols=["sample_id", "image_path"])
    img_path_by_id = dict(zip(unified["sample_id"].astype(str), unified["image_path"]))

    test_rows = np.where(meta["split"].eq("test").to_numpy())[0][:N_ROWS]
    bert = np.load(FEATURE_DIR / "bert.npy")
    clip = np.load(FEATURE_DIR / "clip_img.npy")
    saved = {
        seed: np.load(SELECTION_DIR / "probs" / f"{checkpoint_stem(bundle.winner)}_seed{seed}.npy")
        for seed in SEEDS
    }

    text_enc = TextEncoder(TEXT_MODEL, device)
    img_enc = ImageEncoder(CLIP_IMAGE_MODEL, device)

    text_diffs, img_diffs, prob_diffs = [], [], []
    label_mismatch = 0
    for k, r in enumerate(test_rows):
        row = meta.iloc[r]
        model_text = "" if pd.isna(row["model_text"]) else str(row["model_text"])

        t_vec = text_enc.embed(model_text)
        text_diffs.append(float(np.abs(t_vec - bert[r]).max()))

        pil = None
        if bool(row["has_image"]):
            path = img_path_by_id.get(str(row["sample_id"]))
            if not isinstance(path, str) or not Path(path).exists():
                raise FileNotFoundError(f"Image for sample {row['sample_id']} not found: {path}")
            pil = Image.open(path)
        i_vec = img_enc.embed(pil)
        img_diffs.append(float(np.abs(i_vec - clip[r]).max()))

        _, per_seed = fused_probs_for_text(bundle, text_enc, model_text, i_vec)
        for s_idx, seed in enumerate(SEEDS):
            ref = saved[seed][k]
            prob_diffs.append(float(np.abs(per_seed[s_idx] - ref).max()))
            if int(np.argmax(per_seed[s_idx])) != int(np.argmax(ref)):
                label_mismatch += 1

    t_max, i_max, p_max = max(text_diffs), max(img_diffs), max(prob_diffs)
    print(f"Rows checked: {len(test_rows)} (first test rows), heads per row: {len(SEEDS)}")
    print(f"1. Text embedding   max |diff| vs bert.npy:        {t_max:.2e}  (tol {TEXT_TOL})")
    print(f"2. Image embedding  max |diff| vs clip_img.npy:    {i_max:.2e}  (tol {IMAGE_TOL})")
    print(f"3. Fused probability max |diff| vs saved probs:    {p_max:.2e}  (tol {PROB_TOL})")
    print(f"   Predicted-label mismatches:                     {label_mismatch}")

    ok = t_max < TEXT_TOL and i_max < IMAGE_TOL and p_max < PROB_TOL and label_mismatch == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        print("Where to look: step 1 or 2 failed -> the features or the encoder differ from training.")
        print("               only step 3 failed -> the head, the train statistics or the checkpoint differ.")


if __name__ == "__main__":
    main()
