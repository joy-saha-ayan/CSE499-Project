"""Tests for src/evidence_eval.py. Synthetic data only; these check the split and sheet logic."""
import pandas as pd
import pytest

from src.evidence_app import BM25Index
from src.evidence_eval import assign_posts, build_pairs


def test_assign_posts_is_disjoint_and_reproducible():
    ids = [f"p{i}" for i in range(100)]
    a = assign_posts(ids, n_dev=10, n_eval=10, seed=42)
    b = assign_posts(ids, n_dev=10, n_eval=10, seed=42)
    assert a == b
    assert len(a["dev"]) == 10 and len(a["eval"]) == 10
    assert set(a["dev"]).isdisjoint(a["eval"])


def test_assign_posts_changes_with_seed():
    ids = [f"p{i}" for i in range(100)]
    assert assign_posts(ids, 10, 10, seed=1) != assign_posts(ids, 10, 10, seed=2)


def test_assign_posts_rejects_too_few_posts():
    with pytest.raises(ValueError):
        assign_posts(["a", "b"], n_dev=2, n_eval=2)


def _evidence():
    return pd.DataFrame({
        "evidence_text": [
            "বাংলাদেশের অর্থনীতি নিয়ে নতুন প্রতিবেদন প্রকাশিত হয়েছে",
            "ফুটবল বিশ্বকাপের ইতিহাস ও প্রধান দলসমূহ",
            "ঢাকা শহরের যানজট কমাতে নতুন মেট্রোরেল চালু হচ্ছে",
        ],
        "source": ["wiki", "wiki", "wiki"],
        "url": ["u1", "u2", "u3"],
    })


def test_build_pairs_writes_rows_per_post_and_leaves_label_blank():
    ev = _evidence()
    index = BM25Index(ev["evidence_text"].tolist())
    posts = pd.DataFrame({
        "post_id": ["p1", "p2", "p9"],
        "query_text": ["ঢাকা শহরের মেট্রোরেল নিয়ে খবর", "ফুটবল বিশ্বকাপ জয়", "অন্য বিষয়"],
    })
    roles = {"dev": ["p1"], "eval": ["p2"]}  # p9 is in no role, so it must be skipped
    pairs = build_pairs(posts, roles, index, ev, k=2)
    assert set(pairs["post_id"]) <= {"p1", "p2"}
    assert set(pairs["role"]) <= {"dev", "eval"}
    assert (pairs["relevant_0_or_1"] == "").all()
    assert pairs.groupby("post_id")["rank"].max().max() <= 2


def test_build_pairs_skips_posts_with_no_matching_terms():
    ev = _evidence()
    index = BM25Index(ev["evidence_text"].tolist())
    posts = pd.DataFrame({"post_id": ["p1"], "query_text": ["xyz qqq"]})
    pairs = build_pairs(posts, {"dev": ["p1"], "eval": []}, index, ev, k=3)
    assert len(pairs) == 0
    assert list(pairs.columns)[-1] == "relevant_0_or_1"
