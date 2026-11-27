"""Step 2: build every feature block and save it to FEATURE_DIR.

Order: (OCR, optional) -> frozen BanglaBERT -> CLIP image and text -> cultural -> evidence
(retrieval + NLI). A feature whose input file is missing (annotation or evidence) is skipped,
and the experiments that need it are skipped too.

Run: python scripts/02_extract_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLIP_IMAGE_MODEL,
    CLIP_TEXT_MODEL,
    CULTURAL_COLS,
    USE_CULTURAL_FEATURES,
    CULTURAL_HUMAN_CSV,
    CULTURAL_SILVER_CSV,
    EMBED_MODEL,
    EVIDENCE_CSV,
    EVIDENCE_SOURCE,
    FEATURE_DIR,
    HYBRID_ALPHA,
    LABEL2ID,
    NLI_MODEL,
    OCR_LANGS,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    TOP_K,
    USE_OCR,
)
from src.utils import get_device, save_json, set_seed  # noqa: E402


def main():
    set_seed(SEED)
    device = get_device()
    for d in (FEATURE_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "unified_split.csv")
    df["sample_id"] = df["sample_id"].astype(str)
    df["text"] = df["text"].fillna("")

    # ---- OCR (optional) ----------------------------------------------------
    if USE_OCR:
        from src.ocr import load_ocr_reader, ocr_text

        reader = load_ocr_reader(OCR_LANGS, use_gpu=(device == "cuda"))
        df["ocr_text"] = [ocr_text(reader, p) if isinstance(p, str) else "" for p in df["image_path"]]
    else:
        df["ocr_text"] = ""
    df["ocr_text"] = df["ocr_text"].fillna("")
    df["model_text"] = (df["text"] + " " + df["ocr_text"]).str.strip()
    df["y"] = df["label"].map(LABEL2ID)

    df[["sample_id", "split", "label", "y", "group_id", "has_image", "model_text"]].to_csv(
        FEATURE_DIR / "meta.csv", index=False
    )

    # ---- Frozen BanglaBERT (text pathway) ----------------------------------
    from src.text_encoder import frozen_text_embeddings

    bert = frozen_text_embeddings(df["model_text"].tolist())
    np.save(FEATURE_DIR / "bert.npy", bert)
    print("bert:", bert.shape)

    # ---- CLIP (image pathway + image-text consistency) ---------------------
    from src.image_encoder import encode_images, encode_texts, image_text_similarity, load_clip
    from src.image_presence import has_image_feature, verify_missing_image_encoding

    # Images and text need different models: the multilingual CLIP encodes text only.
    clip_img_model = load_clip(CLIP_IMAGE_MODEL, device)
    clip_txt_model = load_clip(CLIP_TEXT_MODEL, device)
    img_paths = [p if isinstance(p, str) else None for p in df["image_path"]]
    img = encode_images(clip_img_model, img_paths)
    clip_txt = encode_texts(clip_txt_model, df["model_text"].tolist())
    sim = image_text_similarity(img, clip_txt).reshape(-1, 1)
    np.save(FEATURE_DIR / "clip_img.npy", img)
    np.save(FEATURE_DIR / "clip_txt.npy", clip_txt)
    np.save(FEATURE_DIR / "sim.npy", sim)

    # Missing images are zero vectors. Check that they match the metadata, then save the
    # presence flag as its own block so a presence-only control experiment (E0) can run.
    has_image = df["has_image"].astype(bool).to_numpy()
    presence = verify_missing_image_encoding(img, has_image)
    np.save(FEATURE_DIR / "has_image.npy", has_image_feature(has_image))
    print("clip_img:", img.shape, "| image presence:", presence)

    # ---- Cultural context (manual labels, or LLM silver labels if no manual file) ----
    # Manual labels take precedence. Silver labels come from scripts/05 and are marked as such.
    if CULTURAL_HUMAN_CSV.exists():
        cultural_file, cultural_source = CULTURAL_HUMAN_CSV, "manual"
    elif CULTURAL_SILVER_CSV.exists():
        cultural_file, cultural_source = CULTURAL_SILVER_CSV, "llm_silver"
    else:
        cultural_file, cultural_source = None, None

    if cultural_file is not None and not USE_CULTURAL_FEATURES:
        print("[SKIP] cultural: USE_CULTURAL_FEATURES is False (decision D-2); cultural.npy not built.")
    elif cultural_file is not None:
        from src.cultural_context import fit_cultural_classifiers

        annot = pd.read_csv(cultural_file)
        cult, cult_models = fit_cultural_classifiers(bert, df, annot, CULTURAL_COLS, seed=SEED)
        np.save(FEATURE_DIR / "cultural.npy", cult)
        joblib.dump(cult_models, CHECKPOINT_DIR / "cultural_models.joblib")
        save_json({"source": cultural_source, "file": cultural_file.name, "features": CULTURAL_COLS},
                  RESULTS_DIR / "cultural_source.json")
        print(f"cultural ({cultural_source}):", cult.shape)
    else:
        print("[SKIP] no cultural labels found (manual or silver); E4, E5 and E6 will not run.")

    # ---- Evidence retrieval + NLI verification -----------------------------
    if EVIDENCE_CSV.exists():
        from src.evidence_retrieval import EvidenceRetriever, drop_label_source_evidence, recall_at_k
        from src.evidence_verification import NLIVerifier, evidence_features

        ev = pd.read_csv(EVIDENCE_CSV)
        if df["source_url"].isna().all():
            # BTMD has no source URL, so the label-source guard cannot run. That is acceptable only
            # for Wikipedia evidence, which is retrieved from claim text and never from the label (D-1).
            if EVIDENCE_SOURCE != "wikipedia":
                raise ValueError(
                    "source_url is missing; the label-source leakage guard cannot run for "
                    f"EVIDENCE_SOURCE={EVIDENCE_SOURCE!r}. Set source_url_col in configs/btmd_schema.py."
                )
            guard_status = "not_applicable: BTMD has no source URL; evidence is Wikipedia built from claim text only"
            ev_clean, dropped = ev, 0
        else:
            guard_status = "applied"
            ev_clean, dropped = drop_label_source_evidence(ev, df["source_url"].dropna().tolist())
        print(f"Evidence: total {len(ev)}, dropped as label-source {dropped}, used {len(ev_clean)}")
        ev_clean.to_csv(FEATURE_DIR / "evidence_clean.csv", index=False)

        claims = df["model_text"].tolist()
        retriever = EvidenceRetriever(ev_clean, EMBED_MODEL, device)
        hits = retriever.retrieve(claims, k=TOP_K, alpha=HYBRID_ALPHA)
        verifier = NLIVerifier(NLI_MODEL, device)
        evid = evidence_features(
            verifier, claims, hits, ev_clean["evidence_text"].astype(str).tolist()
        )
        np.save(FEATURE_DIR / "evidence.npy", evid)

        report = {"label_source_guard": guard_status, "dropped_label_source": dropped,
                  "evidence_used": len(ev_clean)}
        if "group_id" in ev_clean.columns:
            gold = set(ev_clean["group_id"].astype(str))
            vals = []
            for i in np.where(df["split"].to_numpy() == "test")[0]:
                g = str(df.at[i, "group_id"])
                if g not in gold:
                    continue
                hit_groups = [str(ev_clean.at[j, "group_id"]) for j, _ in hits[i]]
                vals.append(recall_at_k(hit_groups, g, TOP_K))
            report["recall_at_k_test"] = float(np.mean(vals)) if vals else None
            report["n_test_with_gold"] = len(vals)
        save_json(report, RESULTS_DIR / "retrieval_report.json")
        print("evidence:", evid.shape, report)
    else:
        print(f"[SKIP] evidence database not found ({EVIDENCE_CSV}); E5 and E6 will not run.")


if __name__ == "__main__":
    main()
