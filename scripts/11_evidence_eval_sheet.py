"""Step 11: build a NEW relevance sheet with a frozen dev / eval split (decision D-7).

Why a new sheet: the 40-row sheet from D-4 was used to decide the gate. Retriever changes must be
chosen on dev rows only, and the final gate is read once on eval rows.

Input : FEATURE_DIR/meta.csv (model_text, split), EVIDENCE_CSV (Wikipedia passages).
Output: ANNOTATION_DIR/evidence_eval_sheet.csv   (human fills relevant_0_or_1 with 0 or 1)
        ANNOTATION_DIR/evidence_eval_roles.json  (the frozen dev / eval post ids)

Rules:
- Posts come from the validation split only. The test split is never read here.
- The query is the post text only (model_text). No label is used.
- The script refuses to overwrite an existing sheet, so labelled rows are never lost.
- Retriever: BM25 from src/evidence_app.py (pure Python). Dense hybrid is a later comparison on dev.

Run: python scripts/11_evidence_eval_sheet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.config import ANNOTATION_DIR, EVIDENCE_CSV, FEATURE_DIR  # noqa: E402
from src.evidence_app import BM25Index  # noqa: E402
from src.evidence_eval import assign_posts, build_pairs  # noqa: E402
from src.utils import save_json  # noqa: E402

N_DEV = 40
N_EVAL = 40
TOP_K = 3
SEED = 42
SHEET = ANNOTATION_DIR / "evidence_eval_sheet.csv"
ROLES_JSON = ANNOTATION_DIR / "evidence_eval_roles.json"


def main():
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists. Not overwritten: labelled rows would be lost.")
    if not EVIDENCE_CSV.exists():
        raise SystemExit(f"Evidence table not found: {EVIDENCE_CSV}")

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    val = meta[meta["split"] == "val"].copy()
    val["post_id"] = val["sample_id"].astype(str)
    val["query_text"] = val["model_text"].fillna("").astype(str)
    val = val[val["query_text"].str.strip() != ""]

    roles = assign_posts(val["post_id"], N_DEV, N_EVAL, seed=SEED)
    evidence = pd.read_csv(EVIDENCE_CSV).fillna("")
    index = BM25Index(evidence["evidence_text"].astype(str).tolist())

    pairs = build_pairs(val[["post_id", "query_text"]], roles, index, evidence, k=TOP_K)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(SHEET, index=False, encoding="utf-8-sig")
    save_json({"seed": SEED, "n_dev": N_DEV, "n_eval": N_EVAL, "top_k": TOP_K, **roles}, ROLES_JSON)

    per_role = pairs.groupby("role")["post_id"].nunique().to_dict() if len(pairs) else {}
    print(f"Wrote {len(pairs)} pairs to {SHEET}")
    print(f"Posts with passages per role: {per_role}")
    print(f"Assigned posts: dev {len(roles['dev'])}, eval {len(roles['eval'])}")


if __name__ == "__main__":
    main()
