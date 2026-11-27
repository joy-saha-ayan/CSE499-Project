"""Tests for src/evidence_app.py: tokenizer, BM25, leakage guard, relevance gate (D-4, D-5)."""
import math

import pandas as pd
import pytest

from src.evidence_app import (
    GATE_FAIL,
    GATE_INVALID,
    GATE_MISSING,
    GATE_PASS,
    BM25Index,
    drop_fact_check_evidence,
    gate_status,
    relevance_share,
    top_k_passages,
    tokenize,
)


def test_tokenize_keeps_bangla_vowel_signs_inside_words():
    # \w would split at the vowel sign; the tokenizer must keep the word whole
    assert tokenize("বাংলাদেশের রাজধানী") == ["বাংলাদেশের", "রাজধানী"]


def test_tokenize_splits_on_danda_and_common_punctuation():
    assert tokenize("খবরটি ভুয়া।খবর, সত্য!") == ["খবরটি", "ভুয়া", "খবর", "সত্য"]


def test_tokenize_lowercases_latin_and_handles_none():
    assert tokenize("Dhaka DHAKA") == ["dhaka", "dhaka"]
    assert tokenize(None) == []


def test_bm25_ranks_the_document_with_the_query_terms_first():
    docs = ["ঢাকা বাংলাদেশের রাজধানী", "চট্টগ্রাম একটি বন্দর শহর", "ঢাকা একটি বড় শহর"]
    idx = BM25Index(docs)
    scores = idx.scores("ঢাকা রাজধানী")
    assert scores[0] > scores[2] > scores[1] == 0.0


def test_bm25_idf_is_never_negative_for_common_words():
    # a word in every document must not get a negative weight (Lucene idf)
    idx = BM25Index(["ক খ", "ক গ", "ক ঘ"])
    assert all(v > 0 for v in idx.idf.values())
    assert math.isclose(idx.idf["ক"], math.log(1 + (3 - 3 + 0.5) / (3 + 0.5)))


def test_bm25_empty_query_and_no_match_give_zero_scores():
    idx = BM25Index(["একটি বাক্য", "আরেকটি বাক্য"])
    assert idx.scores("") == [0.0, 0.0]
    assert idx.scores("অজানা শব্দ") == [0.0, 0.0]


def test_bm25_handles_all_empty_documents_without_division_by_zero():
    idx = BM25Index(["", " "])
    assert idx.scores("কিছু") == [0.0, 0.0]


def _evidence_table():
    return pd.DataFrame({
        "evidence_id": ["W1_0", "W2_0", "W3_0"],
        "group_id": ["wiki_1", "wiki_2", "wiki_3"],
        "evidence_text": ["ঢাকা. ঢাকা বাংলাদেশের রাজধানী", "পদ্মা. পদ্মা একটি নদী", "মেঘনা. মেঘনা একটি নদী"],
        "source": ["Bangla Wikipedia"] * 3,
        "url": ["https://bn.wikipedia.org/?curid=1", "https://bn.wikipedia.org/?curid=2",
                "https://bn.wikipedia.org/?curid=3"],
    })


def test_top_k_returns_ranked_passages_and_skips_zero_scores():
    ev = _evidence_table()
    idx = BM25Index(ev["evidence_text"].tolist())
    rows = top_k_passages(idx, ev, "ঢাকা রাজধানী", k=5)
    assert [r["rank"] for r in rows] == [1]          # only one passage matches; zero scores are dropped
    assert rows[0]["url"] == "https://bn.wikipedia.org/?curid=1"
    assert rows[0]["score"] > 0


def test_top_k_empty_query_returns_nothing():
    ev = _evidence_table()
    idx = BM25Index(ev["evidence_text"].tolist())
    assert top_k_passages(idx, ev, "   ", k=5) == []


def test_top_k_rejects_index_and_table_of_different_length():
    ev = _evidence_table()
    idx = BM25Index(ev["evidence_text"].tolist()[:2])
    with pytest.raises(ValueError):
        top_k_passages(idx, ev, "ঢাকা", k=3)


def test_guard_with_empty_domain_list_drops_nothing():
    ev = _evidence_table()
    kept, dropped = drop_fact_check_evidence(ev, [])
    assert dropped == 0 and len(kept) == 3


def test_guard_drops_exact_host_and_subdomain_but_not_lookalikes():
    ev = pd.DataFrame({
        "evidence_text": ["a", "b", "c", "d"],
        "url": [
            "https://factcheck.example/post/1",          # exact host: dropped
            "https://www.factcheck.example/post/2",      # subdomain: dropped
            "https://notfactcheck.example/post/3",       # lookalike suffix: kept
            "https://bn.wikipedia.org/?curid=4",         # other site: kept
        ],
    })
    kept, dropped = drop_fact_check_evidence(ev, [".factcheck.example"])
    assert dropped == 2
    assert kept["url"].tolist() == ["https://notfactcheck.example/post/3", "https://bn.wikipedia.org/?curid=4"]


def test_guard_requires_url_column_when_domains_are_given():
    with pytest.raises(KeyError):
        drop_fact_check_evidence(pd.DataFrame({"evidence_text": ["a"]}), ["example.org"])


def test_relevance_share_counts_zeros_and_ones():
    share, n, n_rel = relevance_share([1, 0, 1, 0])
    assert (share, n, n_rel) == (0.5, 4, 2)


def test_relevance_share_accepts_string_and_float_labels():
    assert relevance_share(["1", "0", "1.0", "0.0"])[0] == 0.5


@pytest.mark.parametrize("bad", [[None], [""], ["2"], ["yes"], [float("nan")]])
def test_relevance_share_rejects_blank_or_invalid_labels(bad):
    with pytest.raises(ValueError):
        relevance_share(bad)


def test_relevance_share_rejects_empty_sheet():
    with pytest.raises(ValueError):
        relevance_share([])


def test_gate_missing_file_is_not_validated():
    status, _ = gate_status(None, 0.6)
    assert status == GATE_MISSING


def test_gate_fails_below_threshold_and_passes_at_threshold():
    fail_gate = {"threshold": 0.6, "relevance_share": 13 / 40, "relevant": 13, "n": 40}
    assert gate_status(fail_gate, 0.6)[0] == GATE_FAIL      # 0.325 is the value recorded in D-4
    pass_gate = {"threshold": 0.6, "relevance_share": 24 / 40, "relevant": 24, "n": 40}
    assert gate_status(pass_gate, 0.6)[0] == GATE_PASS      # the threshold itself counts as a pass


def test_gate_with_a_different_recorded_threshold_is_invalid():
    # a file written with a looser threshold must not pass silently
    gate = {"threshold": 0.3, "relevance_share": 0.325, "relevant": 13, "n": 40}
    status, detail = gate_status(gate, 0.6)
    assert status == GATE_INVALID
    assert "0.6" in detail
