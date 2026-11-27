"""Step 0: preflight check. Verifies model IDs, data files, schema and GPU before running anything.

Run it before every stage. If any item is FAIL, do not continue to the next stage.
Run: python scripts/00_preflight.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.btmd_schema import BTMD_SCHEMA  # noqa: E402
from configs.config import (  # noqa: E402
    ANNOTATION_DIR,
    CLIP_IMAGE_MODEL,
    CULTURAL_HUMAN_CSV,
    CULTURAL_SILVER_CSV,
    CLIP_TEXT_MODEL,
    EMBED_MODEL,
    EVIDENCE_CSV,
    NLI_MODEL,
    RAW_DIR,
    TEXT_MODEL,
)

REQUIRED_SCHEMA = ["table_file", "id_col", "text_col", "label_col"]
OPTIONAL_SCHEMA = ["image_col", "group_col", "source_url_col"]


def check_gpu():
    import torch

    if torch.cuda.is_available():
        print(f"[OK]   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] No GPU. BanglaBERT and CLIP will be very slow on CPU. "
              "In Colab, set Runtime > Change runtime type > T4 GPU.")


def check_models():
    from huggingface_hub import model_info

    ok = True
    for repo in [TEXT_MODEL, CLIP_IMAGE_MODEL, CLIP_TEXT_MODEL, EMBED_MODEL, NLI_MODEL]:
        try:
            model_info(repo)
            print(f"[OK]   Hugging Face model: {repo}")
        except Exception as e:  # diagnostic: show which model failed and why
            ok = False
            print(f"[FAIL] Hugging Face model: {repo} -> {type(e).__name__}: {e}")
    return ok


def check_schema():
    ok = True
    for key in REQUIRED_SCHEMA:
        if BTMD_SCHEMA.get(key) is None:
            ok = False
            print(f"[FAIL] btmd_schema.py: '{key}' is not filled in")
    for key in OPTIONAL_SCHEMA:
        if BTMD_SCHEMA.get(key) is None:
            print(f"[WARN] btmd_schema.py: '{key}' is not set; related features/guards will be skipped")
    if ok:
        print("[OK]   btmd_schema.py: required fields are filled in")
    return ok


def check_files():
    has_raw = RAW_DIR.exists() and any(RAW_DIR.iterdir())
    pairs = ANNOTATION_DIR / "claim_evidence_pairs.csv"
    print(f"[{'OK' if has_raw else 'FAIL'}]   BTMD slim data: {RAW_DIR}")
    print(f"[{'OK' if EVIDENCE_CSV.exists() else 'MISS'}] evidence.csv: {EVIDENCE_CSV}")
    print(f"[{'OK' if CULTURAL_HUMAN_CSV.exists() else 'MISS'}] cultural_annotations.csv (manual): {CULTURAL_HUMAN_CSV}")
    print(f"[{'OK' if CULTURAL_SILVER_CSV.exists() else 'MISS'}] cultural_silver_labels.csv (LLM): {CULTURAL_SILVER_CSV}")
    print(f"[{'OK' if pairs.exists() else 'MISS'}] claim_evidence_pairs.csv: {pairs}")
    return has_raw


def print_plan():
    have_evidence = EVIDENCE_CSV.exists()  # Wikipedia evidence needs no source URL (D-1)
    have_cultural = CULTURAL_HUMAN_CSV.exists() or CULTURAL_SILVER_CSV.exists()
    print("\nExperiments that can run:")
    print("  E1  BanglaBERT fine-tune (text)         : runs on BTMD alone")
    print("  E1f/E2/E3 frozen features + MLP          : runs on BTMD alone")
    print(f"  E4  + cultural                           : {'yes' if have_cultural else 'no, run scripts/05 first'}")
    print(f"  E5/E6 + evidence + decision layer        : "
          f"{'yes' if have_evidence and have_cultural else 'no, needs evidence (script 06) and cultural labels (script 05)'}")


def main():
    results = {
        "gpu": True,
        "models": check_models(),
        "schema": check_schema(),
        "data": check_files(),
    }
    check_gpu()
    print_plan()
    if all(results.values()):
        print("\nPreflight passed. Next step: python scripts/01_prepare_data.py")
    else:
        print("\nSome items FAILED. Fix them first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
