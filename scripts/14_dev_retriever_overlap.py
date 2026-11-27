"""Step 14: do the hybrid retriever's dev passages match the labelled BM25 passages? (D-11)

Why: the dev labels cover only the BM25 top-3 passages. A hybrid (BM25 + dense) retriever may return
passages nobody has labelled. Those need dev labels before the retrievers can be compared.

Input : ANNOTATION_DIR/evidence_eval_sheet.csv (dev rows labelled, eval rows blank)
        EVIDENCE_CSV, FEATURE_DIR/meta.csv (model_text is the query, as in script 11)
Output: RESULTS_DIR/evidence/dev_overlap.json
        RESULTS_DIR/evidence/dev_hybrid_top3.csv  (hybrid top-3 per dev post, with passage; used by scripts/15)
        RESULTS_DIR/evidence/dev_new_passages.csv  (unlabelled (post, passage) pairs, same columns as the sheet, label blank)

Rules:
- Only dev rows are read. The script refuses if any eval row has a label.
- No label is used to choose passages. Labels are only used to say which passages are already labelled.
- The hybrid alpha is 0.5, the existing default. It is not tuned here.
- Needs the GPU/CPU embedding model (EMBED_MODEL). It cannot run without sentence-transformers.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.config import ANNOTATION_DIR, EMBED_MODEL, EVIDENCE_CSV, FEATURE_DIR, RESULTS_DIR  # noqa: E402
from src.evidence_app import BM25Index, top_k_passages  # noqa: E402
from src.evidence_retrieval import EvidenceRetriever  # noqa: E402
from src.utils import get_device, save_json  # noqa: E402

SHEET = ANNOTATION_DIR / "evidence_eval_sheet.csv"
OUT_DIR = RESULTS_DIR / "evidence"
TOP_K = 3
ALPHA = 0.5
PASSAGE_CHARS = 400  # same cut as src/evidence_eval.py


def main():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    eval_labels = sheet.loc[sheet["role"] == "eval", "relevant_0_or_1"].str.strip()
    if (eval_labels != "").any():
        raise SystemExit("Eval rows have labels. This check must run before the eval labels are used. Stop.")
    dev = sheet[sheet["role"] == "dev"].copy()
    if (dev["relevant_0_or_1"].str.strip() == "").any():
        raise SystemExit("Some dev rows are unlabelled. Finish dev labelling first.")

    evidence = pd.read_csv(EVIDENCE_CSV)
    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    query_by_id = dict(zip(meta["sample_id"].astype(str), meta["model_text"].fillna("").astype(str)))
    missing = [p for p in dev["post_id"].unique() if p not in query_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} dev post_ids are not in meta.csv. The sheet and features disagree.")

    # Pair-level key: a passage labelled for one post is not labelled for another post.
    labelled_pairs = {(p, str(t)[:PASSAGE_CHARS]) for p, t in zip(dev["post_id"], dev["passage"])}
    post_text_by_id = dict(zip(dev["post_id"], dev["post_text"]))
    index = BM25Index(evidence["evidence_text"].astype(str).tolist())
    device = get_device()
    retriever = EvidenceRetriever(evidence, EMBED_MODEL, device)

    rows, new_rows = [], []
    for post_id in dev["post_id"].unique():
        query = query_by_id[post_id]
        bm25_hits = top_k_passages(index, evidence, query, TOP_K)
        bm25_keys = {(post_id, str(h["passage"])[:PASSAGE_CHARS]) for h in bm25_hits}
        hybrid_hits = retriever.retrieve([query], k=TOP_K, alpha=ALPHA)[0]
        for rank, (j, score) in enumerate(hybrid_hits, start=1):
            text = str(evidence.iloc[j]["evidence_text"])
            key = (post_id, text[:PASSAGE_CHARS])
            labelled = key in labelled_pairs
            rows.append({"post_id": post_id, "hybrid_rank": rank, "evidence_row": j, "passage": text[:PASSAGE_CHARS],
                         "hybrid_score": round(score, 4), "in_labelled_set": labelled,
                         "in_bm25_top3": key in bm25_keys})
            if not labelled:
                new_rows.append({
                    "pair_id": f"{post_id}_h{rank}",
                    "role": "dev",
                    "post_id": post_id,
                    "post_text": post_text_by_id.get(post_id, ""),
                    "rank": rank,
                    "score": round(score, 4),
                    "passage": text[:PASSAGE_CHARS],
                    "source": str(evidence.iloc[j].get("source", "")),
                    "url": str(evidence.iloc[j].get("url", "")),
                    "relevant_0_or_1": "",
                    "in_bm25_top3": key in bm25_keys,
                })

    hits = pd.DataFrame(rows)
    new_df = pd.DataFrame(new_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hits.to_csv(OUT_DIR / "dev_hybrid_top3.csv", index=False, encoding="utf-8-sig")
    summary = {
        "posts": int(dev["post_id"].nunique()),
        "hybrid_pairs": int(len(hits)),
        "hybrid_pairs_already_labelled": int(hits["in_labelled_set"].sum()) if len(hits) else 0,
        "unlabelled_hybrid_pairs": int(len(new_df)),
        "posts_with_new_pair": int(new_df["post_id"].nunique()) if len(new_df) else 0,
        "alpha": ALPHA, "top_k": TOP_K, "embed_model": EMBED_MODEL, "device": device,
        "note": "Pair-level: a (post, passage) pair counts as labelled only if that post was labelled with that passage. Passages are matched by their first 400 characters. Dev labels only.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(summary, OUT_DIR / "dev_overlap.json")
    new_df.to_csv(OUT_DIR / "dev_new_passages.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"New (post, passage) pairs to label: {len(new_df)} (see {OUT_DIR / 'dev_new_passages.csv'})")


if __name__ == "__main__":
    main()
