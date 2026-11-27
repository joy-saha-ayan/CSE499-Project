"""Step 16: build a FRESH evidence sheet for the pre-registered test (D-14). Labels are left blank.

Input : FEATURE_DIR/meta.csv (model_text, split), EVIDENCE_CSV (passages),
        ANNOTATION_DIR/evidence_eval_roles.json (the D-7 dev and eval post ids, which are excluded)
Output: ANNOTATION_DIR/evidence_fresh_sheet.csv        (labeller fills relevant_0_or_1; no method column)
        RESULTS_DIR/evidence/fresh_method_map.csv      (pair_id -> method and rank; NOT for the labeller)
        RESULTS_DIR/evidence/fresh_sample.json         (post ids, keyword queries, counts)

Rules (fixed in D-14 before any label):
- Posts: validation split only, 40 posts, not in the D-7 dev or eval roles, random.Random(43).
- Methods: baseline = post text as query; keyword = top 8 idf tokens of the post (stopwords removed).
- Each method gives BM25 top 3. The sheet holds the deduplicated union by (post_id, passage[:400]).
- Labels are never written here. The script refuses to overwrite an existing sheet.
"""
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from configs.config import ANNOTATION_DIR, EVIDENCE_CSV, FEATURE_DIR, RESULTS_DIR  # noqa: E402
from src.evidence_app import BM25Index, tokenize, top_k_passages  # noqa: E402
from src.utils import save_json  # noqa: E402

N_POSTS = 40
SEED = 43
TOP_K = 3
KEYWORDS = 8
PASSAGE_CHARS = 400
POST_CHARS = 300
ROLES_JSON = ANNOTATION_DIR / "evidence_eval_roles.json"
SHEET = ANNOTATION_DIR / "evidence_fresh_sheet.csv"
OUT_DIR = RESULTS_DIR / "evidence"

# Fixed list (D-14). Common Bangla and English function words that carry no claim content.
STOPWORDS = {
    "এবং", "ও", "এর", "এই", "সে", "তার", "তারা", "তিনি", "আমি", "আমরা", "যে", "যা", "কি", "কে", "না",
    "হয়", "হবে", "হয়েছে", "থেকে", "জন্য", "করে", "করা", "একটি", "একটা", "নিয়ে", "কিন্তু", "তো", "আর",
    "সব", "এখন", "আজ", "হলো", "হল", "দিয়ে", "মধ্যে", "উপর", "যদি", "কেন", "কী", "এ", "the", "and",
    "of", "to", "in", "is", "a", "for", "on", "with", "that", "this", "by", "at", "as", "it", "be",
}


def keyword_query(post_text, idf, n=KEYWORDS):
    tokens = {t.lower() for t in tokenize(str(post_text)) if len(t) >= 2 and t.lower() not in STOPWORDS}
    scored = [(idf[t], t) for t in tokens if t in idf]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return " ".join(t for _, t in scored[:n])


def build_idf(passages):
    n_docs = len(passages)
    df = Counter()
    for text in passages:
        df.update(set(t.lower() for t in tokenize(text)))
    return {t: math.log((n_docs + 1) / (c + 1)) for t, c in df.items()}


def main():
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists. Not overwritten: labels would be lost.")
    if not ROLES_JSON.exists():
        raise SystemExit(f"Roles file not found: {ROLES_JSON}")

    roles = json.loads(ROLES_JSON.read_text(encoding="utf-8"))
    used = {str(p) for p in roles.get("dev", [])} | {str(p) for p in roles.get("eval", [])}

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    val = meta[meta["split"] == "val"].copy()
    val["post_id"] = val["sample_id"].astype(str)
    val["query_text"] = val["model_text"].fillna("").astype(str)
    val = val[val["query_text"].str.strip() != ""]
    candidates = sorted({p for p in val["post_id"] if p not in used})
    if len(candidates) < N_POSTS:
        raise SystemExit(f"Only {len(candidates)} unused validation posts; need {N_POSTS}.")
    chosen = random.Random(SEED).sample(candidates, N_POSTS)
    text_by_id = dict(zip(val["post_id"], val["query_text"]))

    evidence = pd.read_csv(EVIDENCE_CSV).fillna("")
    passages = evidence["evidence_text"].astype(str).tolist()
    index = BM25Index(passages)
    idf = build_idf(passages)

    rows, method_rows, queries = [], [], {}
    seen = {}
    for pid in chosen:
        post_text = text_by_id[pid]
        kw = keyword_query(post_text, idf)
        queries[pid] = kw
        for method, query in [("baseline", post_text), ("keyword", kw)]:
            for hit in top_k_passages(index, evidence, query, TOP_K):
                key = (pid, hit["passage"][:PASSAGE_CHARS])
                if key in seen:
                    seen[key]["methods"].add(method)
                    seen[key]["rank"][method] = hit["rank"]
                    continue
                seen[key] = {"methods": {method}, "rank": {method: hit["rank"]}, "hit": hit, "post": pid,
                             "post_text": post_text}

    for n, (key, info) in enumerate(seen.items(), start=1):
        hit = info["hit"]
        pair_id = f"fresh_{n:03d}"
        rows.append({
            "pair_id": pair_id, "role": "fresh", "post_id": info["post"],
            "post_text": info["post_text"][:POST_CHARS], "rank": min(info["rank"].values()),
            "score": hit["score"], "passage": hit["passage"][:PASSAGE_CHARS],
            "source": hit["source"], "url": hit["url"], "relevant_0_or_1": "",
        })
        method_rows.append({
            "pair_id": pair_id, "post_id": info["post"],
            "methods": "+".join(sorted(info["methods"])),
            "rank_baseline": info["rank"].get("baseline", ""),
            "rank_keyword": info["rank"].get("keyword", ""),
        })

    sheet = pd.DataFrame(rows)
    method_df = pd.DataFrame(method_rows)
    n_keyword = int(method_df["rank_keyword"].astype(str).str.strip().ne("").sum())
    n_baseline = int(method_df["rank_baseline"].astype(str).str.strip().ne("").sum())

    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method_df.to_csv(OUT_DIR / "fresh_method_map.csv", index=False, encoding="utf-8-sig")
    save_json({"seed": SEED, "posts": chosen, "keyword_queries": queries, "top_k": TOP_K,
               "keywords": KEYWORDS, "unique_pairs": len(sheet), "keyword_pairs": n_keyword,
               "baseline_pairs": n_baseline,
               "note": "D-14. Labels are blank. The gate uses keyword_pairs only."},
              OUT_DIR / "fresh_sample.json")
    print(f"Wrote {len(sheet)} unique pairs to {SHEET}")
    print(f"Keyword top-3 pairs (gate set): {n_keyword} | baseline top-3 pairs: {n_baseline}")


if __name__ == "__main__":
    main()
