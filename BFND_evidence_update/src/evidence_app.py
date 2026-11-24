"""Evidence for the app: BM25 search over Wikipedia passages, leakage guard, and the relevance gate.

Decision D-5: the app shows retrieved passages and the gate status. The prediction never uses them.
Evidence enters a prediction only after a validated gate (D-4) and a validation gain (E5), and
that change needs a new decision entry.

Pure Python, no torch and no rank_bm25, so the logic runs and is tested without a GPU.
The dense retriever (EvidenceRetriever in src/evidence_retrieval.py) is not used here yet:
the choice between BM25, dense and hybrid must be made on a labelled dev set first.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_MISSING = "NOT VALIDATED"
GATE_INVALID = "INVALID"

# Split on spaces and common punctuation, including the Bangla danda. Bangla vowel signs are not
# split off, which \w would do, so the tokenizer does not use \w.
_TOKEN_SPLIT = re.compile(r"[\s.,;:!?\"'()\[\]{}<>।॥\-–—/|]+")


def tokenize(text) -> list:
    text = str(text if text is not None else "").lower()
    return [t for t in _TOKEN_SPLIT.split(text) if t]


class BM25Index:
    """Okapi BM25 with the Lucene idf, log(1 + (N - df + 0.5) / (df + 0.5)), which is never negative."""

    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.tf]
        n = len(self.tf)
        self.avgdl = (sum(self.doc_len) / n) if n else 1.0
        self.avgdl = self.avgdl or 1.0  # all documents empty: avoid division by zero
        self.postings = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for word, freq in counts.items():
                self.postings[word].append((i, freq))
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

    def __len__(self) -> int:
        return len(self.tf)

    def scores(self, query) -> list:
        out = [0.0] * len(self.tf)
        for word in set(tokenize(query)):
            if word not in self.postings:
                continue
            idf = self.idf[word]
            for i, freq in self.postings[word]:
                norm = 1.0 - self.b + self.b * self.doc_len[i] / self.avgdl
                out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return out


def drop_fact_check_evidence(evidence_df: pd.DataFrame, domains):
    """D-5 leakage guard: drop rows whose URL host is, or is under, one of the given domains.

    Returns (kept_df, number_dropped). An empty domain list drops nothing.
    """
    if not domains:
        return evidence_df.reset_index(drop=True), 0
    if "url" not in evidence_df.columns:
        raise KeyError("evidence table must contain a 'url' column for the leakage guard")
    banned = [str(d).lower().strip().lstrip(".") for d in domains]

    def is_banned(url) -> bool:
        host = (urlparse(str(url)).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in banned)

    drop = evidence_df["url"].map(is_banned)
    return evidence_df[~drop].reset_index(drop=True), int(drop.sum())


def top_k_passages(index: BM25Index, evidence_df: pd.DataFrame, query: str, k: int) -> list:
    """Return up to k passages with a positive score, best first. The query is the post text only."""
    if len(index) != len(evidence_df):
        raise ValueError("index and evidence table have different lengths")
    if not str(query).strip() or k <= 0:
        return []
    scores = index.scores(query)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    rows = []
    for rank, i in enumerate(order, start=1):
        if scores[i] <= 0:
            break
        row = evidence_df.iloc[i]
        rows.append({
            "rank": rank,
            "score": round(float(scores[i]), 3),
            "passage": str(row["evidence_text"]),
            "source": str(row.get("source", "")),
            "url": str(row.get("url", "")),
        })
    return rows


# ---- relevance gate (D-4) -------------------------------------------------------------------

def relevance_share(labels) -> tuple:
    """Return (share, n, n_relevant) from 0/1 labels. Every row must be labelled; a blank raises."""
    values = []
    for k, raw in enumerate(labels):
        text = "" if raw is None or (isinstance(raw, float) and math.isnan(raw)) else str(raw).strip()
        if text not in ("0", "1", "0.0", "1.0"):
            raise ValueError(f"row {k}: relevance label must be 0 or 1, got {raw!r}")
        values.append(int(float(text)))
    n = len(values)
    if n == 0:
        raise ValueError("relevance sheet has no labelled rows")
    n_rel = sum(values)
    return n_rel / n, n, n_rel


def load_gate(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gate_status(gate: dict | None, threshold: float) -> tuple:
    """Return (status, detail). The threshold recorded in the gate file must equal the configured one.

    A different recorded threshold gives GATE_INVALID, so a changed threshold cannot pass silently.
    """
    if gate is None:
        return GATE_MISSING, "relevance gate file not found; run scripts/09_evidence_gate.py after labelling the sheet"
    recorded = gate.get("threshold")
    if recorded is None or abs(float(recorded) - float(threshold)) > 1e-9:
        return GATE_INVALID, (f"gate file threshold {recorded} differs from the configured {threshold} (D-4); "
                              "the file must be regenerated with the pre-set threshold")
    share = float(gate["relevance_share"])
    detail = f"relevance {share:.3f} ({gate.get('relevant')} of {gate.get('n')} rows), threshold {threshold}"
    return (GATE_PASS if share >= threshold else GATE_FAIL), detail
