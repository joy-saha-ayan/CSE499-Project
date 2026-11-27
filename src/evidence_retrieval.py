"""Evidence retrieval: hybrid of BM25 and multilingual dense embeddings.

The pure helpers (drop_label_source_evidence, hybrid_scores, recall_at_k) run without torch.
EvidenceRetriever loads sentence-transformers only when it is created.
"""
import numpy as np
import pandas as pd

from src.preprocessing import normalize_url


def drop_label_source_evidence(evidence_df: pd.DataFrame, label_source_urls):
    """Drop evidence that comes from the same fact-check article as a BTMD label.

    Otherwise retrieval would simply return the label itself (label leakage).
    """
    if "url" not in evidence_df.columns:
        raise KeyError("evidence.csv must contain a 'url' column (needed for the leakage guard)")
    banned = {normalize_url(u) for u in label_source_urls} - {None}
    urls = evidence_df["url"].map(normalize_url)
    drop = urls.isin(banned)
    return evidence_df[~drop].reset_index(drop=True), int(drop.sum())


def normalize_scores(scores):
    s = np.asarray(scores, dtype=float)
    rng = s.max() - s.min()
    if rng == 0:
        return np.zeros_like(s)
    return (s - s.min()) / rng


def hybrid_scores(bm25_scores, dense_scores, alpha=0.5):
    return alpha * normalize_scores(bm25_scores) + (1 - alpha) * normalize_scores(dense_scores)


def recall_at_k(retrieved_groups, gold_group, k=5) -> float:
    return float(str(gold_group) in [str(g) for g in list(retrieved_groups)[:k]])


class EvidenceRetriever:
    def __init__(self, evidence_df: pd.DataFrame, embed_model_name: str, device: str):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        if len(evidence_df) == 0:
            raise ValueError("Evidence database is empty")
        self.df = evidence_df.reset_index(drop=True)
        self.texts = self.df["evidence_text"].astype(str).tolist()
        self.bm25 = BM25Okapi([t.split() for t in self.texts])
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        # e5 rule: passages and queries need different prefixes
        self.doc_emb = self.embedder.encode(
            ["passage: " + t for t in self.texts],
            batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
        )

    def retrieve(self, claims, k=5, alpha=0.5):
        """Return the top-k [(evidence_row_index, score), ...] for each claim."""
        q = self.embedder.encode(
            ["query: " + str(c) for c in claims],
            batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
        )
        dense = q @ self.doc_emb.T
        results = []
        for i, claim in enumerate(claims):
            bm = self.bm25.get_scores(str(claim).split())
            score = hybrid_scores(bm, dense[i], alpha)
            top = np.argsort(-score)[: min(k, len(score))]
            results.append([(int(j), float(score[j])) for j in top])
        return results
