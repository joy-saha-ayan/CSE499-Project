"""Step 13: OCR character error rate on hand-typed references (limitation 13).

Input : ANNOTATION_DIR/ocr_gold.csv with columns
        image_id, reference_text, reference_source, ocr_text
        reference_source must be "human" for every row. The script refuses otherwise.
Output: RESULTS_DIR/ocr/ocr_cer.json and ocr_cer.md

Rules:
- The reference is typed by a person who looked at the image. The OCR output is never copied into it.
- No OCR model is run here. ocr_text is whatever the OCR step produced, saved to the CSV beforehand.
- Nothing is tuned on these rows. The OCR filter thresholds are not changed after reading the result.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.config import ANNOTATION_DIR, RESULTS_DIR  # noqa: E402
from src.ocr_metrics import cer, corpus_cer  # noqa: E402
from src.utils import save_json  # noqa: E402

INPUT = ANNOTATION_DIR / "ocr_gold.csv"
OUT_DIR = RESULTS_DIR / "ocr"
REQUIRED = ["image_id", "reference_text", "reference_source", "ocr_text"]


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing {INPUT}. Create it with a person typing reference_text for about 20 images.")
    df = pd.read_csv(INPUT, dtype=str).fillna("")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"ocr_gold.csv is missing columns: {missing}")
    bad_source = df.loc[df["reference_source"].str.strip().str.lower() != "human", "image_id"].tolist()
    if bad_source:
        raise SystemExit(f"reference_source must be 'human' for every row. Not human: {bad_source[:10]}")
    if (df["reference_text"].str.strip() == "").any():
        raise SystemExit("Some reference_text cells are empty. Type the text for every image.")
    if len(df) < 20:
        print(f"Warning: only {len(df)} images. Report this as a pilot, not as an accuracy estimate.")

    pairs = list(zip(df["reference_text"], df["ocr_text"]))
    summary = corpus_cer(pairs)
    per_image = [{"image_id": i, "cer": round(cer(r, h), 4)} for i, r, h in
                 zip(df["image_id"], df["reference_text"], df["ocr_text"])]
    result = {"input": INPUT.name, "summary": summary, "per_image": per_image}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(result, OUT_DIR / "ocr_cer.json")
    lines = ["# OCR character error rate (pilot)", "",
             f"- images: {summary['images']}",
             f"- reference characters: {summary['reference_chars']}",
             f"- pooled CER: {summary['pooled_cer']:.4f}",
             f"- mean per-image CER: {summary['mean_per_image_cer']:.4f}", "",
             "| image_id | CER |", "|---|---|"]
    lines += [f"| {x['image_id']} | {x['cer']:.4f} |" for x in per_image]
    (OUT_DIR / "ocr_cer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main()
