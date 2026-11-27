"""Frozen dev / eval split for judging evidence relevance (pure Python, no torch, no network).

Why: the first relevance check (40 pairs, 13 relevant) was used to decide the gate. Any retriever
change made after seeing that sheet would be tuned on the gate data. So the new check is split once
into two disjoint sets of posts:
- dev:  used to compare retrievers and fix retriever settings
- eval: used only once, for the final gate, after the retriever is frozen

Posts come from the validation split only. The test split is never used here.
"""
import random

import pandas as pd

from src.evidence_app import top_k_passages

ROLES = ("dev", "eval")
PASSAGE_CHARS = 400
POST_CHARS = 300


def assign_posts(post_ids, n_dev: int, n_eval: int, seed: int = 42) -> dict:
    """Return {"dev": [...], "eval": [...]}: disjoint, reproducible for the same seed and input."""
    ids = sorted({str(p) for p in post_ids})
    if n_dev + n_eval > len(ids):
        raise ValueError(f"need {n_dev + n_eval} posts, only {len(ids)} available")
    rng = random.Random(seed)
    chosen = rng.sample(ids, n_dev + n_eval)
    return {"dev": chosen[:n_dev], "eval": chosen[n_dev:]}


def build_pairs(posts: pd.DataFrame, roles: dict, index, evidence_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """One row per (post, retrieved passage). posts needs columns post_id and query_text.

    relevant_0_or_1 is left empty for a human to fill. Posts with no passage scoring above zero
    produce no rows; their count is reported by the caller, not hidden here.
    """
    role_of = {pid: role for role in ROLES for pid in roles.get(role, [])}
    rows = []
    for _, post in posts.iterrows():
        pid = str(post["post_id"])
        if pid not in role_of:
            continue
        query = str(post["query_text"])
        for hit in top_k_passages(index, evidence_df, query, k):
            rows.append({
                "pair_id": f"{pid}-{hit['rank']}",
                "role": role_of[pid],
                "post_id": pid,
                "post_text": query[:POST_CHARS],
                "rank": hit["rank"],
                "score": hit["score"],
                "passage": hit["passage"][:PASSAGE_CHARS],
                "source": hit["source"],
                "url": hit["url"],
                "relevant_0_or_1": "",
            })
    return pd.DataFrame(rows, columns=[
        "pair_id", "role", "post_id", "post_text", "rank", "score",
        "passage", "source", "url", "relevant_0_or_1",
    ])
