"""Tests for src/llm_labeling.py and src/wiki_evidence.py. Pure Python and pandas: no API key,
no network, no anthropic package. The real API and Wikipedia runs happen in Colab."""
import numpy as np
import pandas as pd
import pytest

from src.llm_labeling import (
    agreement_report,
    append_cache,
    call_with_retries,
    cohen_kappa_binary,
    cultural_prompt,
    load_cache,
    parse_cultural_answer,
    parse_json_object,
    parse_query_answer,
    query_prompt,
    run_cached,
    stratified_check_sample,
)
from src.wiki_evidence import (
    build_extract_params,
    build_search_params,
    chunk_text,
    evidence_rows,
    parse_extracts,
    parse_search_ids,
)

LABELS = ["Authentic", "Misleading", "Satire"]
COLS = ["sarcasm", "humor", "local_reference", "historical_reference"]


# ---- prompts never contain the class label ---------------------------------------

def test_cultural_prompt_has_no_class_label():
    prompt = cultural_prompt("আজকের খবর দেখুন", COLS)
    for name in LABELS:
        assert name not in prompt


def test_query_prompt_has_no_class_label():
    prompt = query_prompt("আজকের খবর দেখুন")
    for name in LABELS:
        assert name not in prompt


def test_cultural_prompt_truncates_long_text():
    prompt = cultural_prompt("ক" * 5000, COLS, max_chars=100)
    assert prompt.count("ক") == 100


def test_cultural_prompt_rejects_unknown_feature():
    with pytest.raises(KeyError):
        cultural_prompt("text", ["sarcasm", "unknown_feature"])


# ---- strict parsing ---------------------------------------------------------------

def test_parse_cultural_answer_accepts_json_with_surrounding_prose():
    answer = 'Here you go: {"sarcasm": 1, "humor": 0, "local_reference": 1, "historical_reference": 0}'
    assert parse_cultural_answer(answer, COLS) == {
        "sarcasm": 1, "humor": 0, "local_reference": 1, "historical_reference": 0,
    }


def test_parse_cultural_answer_rejects_missing_key():
    with pytest.raises(ValueError):
        parse_cultural_answer('{"sarcasm": 1, "humor": 0}', COLS)


@pytest.mark.parametrize("bad_value", ['"1"', "2", "1.0", "null"])
def test_parse_cultural_answer_rejects_non_binary_values(bad_value):
    answer = '{"sarcasm": %s, "humor": 0, "local_reference": 0, "historical_reference": 0}' % bad_value
    with pytest.raises(ValueError):
        parse_cultural_answer(answer, COLS)


def test_parse_cultural_answer_accepts_json_booleans():
    answer = '{"sarcasm": true, "humor": false, "local_reference": 0, "historical_reference": 1}'
    out = parse_cultural_answer(answer, COLS)
    assert out["sarcasm"] == 1 and out["humor"] == 0


def test_parse_json_object_rejects_text_without_json():
    with pytest.raises(ValueError):
        parse_json_object("I cannot answer that.")


def test_parse_query_answer_returns_query_and_empty_string():
    assert parse_query_answer('{"query": " বাংলাদেশ নির্বাচন "}') == "বাংলাদেশ নির্বাচন"
    assert parse_query_answer('{"query": ""}') == ""


def test_parse_query_answer_truncates_long_query():
    assert len(parse_query_answer('{"query": "%s"}' % ("ক" * 300))) == 120


def test_parse_query_answer_rejects_missing_key():
    with pytest.raises(ValueError):
        parse_query_answer('{"text": "x"}')


def test_parse_query_answer_falls_back_on_stray_character():
    # Observed in a real run: a stray ')' made the JSON invalid.
    answer = '{"query": "বাংলাদেশ ইন্টারন্যাশনাল অলিম্পিয়াড")}'
    assert parse_query_answer(answer) == "বাংলাদেশ ইন্টারন্যাশনাল অলিম্পিয়াড"


def test_parse_query_answer_rejects_unreadable_answer():
    with pytest.raises(ValueError):
        parse_query_answer("I cannot find a subject.")


def test_parse_cultural_answer_falls_back_on_stray_character():
    answer = '{"sarcasm": 1, "humor": 0, "local_reference": 1, "historical_reference": 0})'
    assert parse_cultural_answer(answer, COLS) == {
        "sarcasm": 1, "humor": 0, "local_reference": 1, "historical_reference": 0,
    }


def test_parse_cultural_answer_fallback_still_rejects_string_values():
    answer = '{"sarcasm": "1", "humor": 0, "local_reference": 1, "historical_reference": 0})'
    with pytest.raises(ValueError):
        parse_cultural_answer(answer, COLS)


# ---- retries ------------------------------------------------------------------------

def test_call_with_retries_succeeds_after_transient_failures():
    calls = {"n": 0}
    waits = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert call_with_retries(flaky, max_retries=5, sleep=waits.append) == "ok"
    assert calls["n"] == 3
    assert waits == [2.0, 4.0]  # exponential back-off


def test_call_with_retries_raises_after_max_attempts():
    def always_fails():
        raise ValueError("bad answer")

    with pytest.raises(RuntimeError):
        call_with_retries(always_fails, max_retries=3, sleep=lambda s: None)


# ---- resumable cache ------------------------------------------------------------------

def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl"
    append_cache(path, "a", {"x": 1})
    append_cache(path, "b", [1, 2])
    cache, bad = load_cache(path)
    assert cache == {"a": {"x": 1}, "b": [1, 2]}
    assert bad == 0


def test_cache_skips_line_cut_by_disconnect(tmp_path):
    path = tmp_path / "cache.jsonl"
    append_cache(path, "a", 1)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "b", "val')  # interrupted write
    cache, bad = load_cache(path)
    assert cache == {"a": 1}
    assert bad == 1


def test_run_cached_resumes_without_recomputing(tmp_path):
    path = tmp_path / "cache.jsonl"
    calls = []

    def fn(payload):
        calls.append(payload)
        return payload * 2

    items = [("k1", 1), ("k2", 2)]
    first = run_cached(items, fn, path, workers=2)
    second = run_cached(items + [("k3", 3)], fn, path, workers=2)
    assert first == {"k1": 2, "k2": 4}
    assert second["k3"] == 6
    assert sorted(calls) == [1, 2, 3]  # k1 and k2 were not computed again


def test_run_cached_propagates_failure(tmp_path):
    def fn(payload):
        if payload == 2:
            raise RuntimeError("API failed")
        return payload

    with pytest.raises(RuntimeError):
        run_cached([("a", 1), ("b", 2)], fn, tmp_path / "c.jsonl", workers=1)


# ---- silver-label validation ---------------------------------------------------------

def test_cohen_kappa_known_value():
    # po = 0.75, pe = 0.5 -> kappa = 0.5
    assert cohen_kappa_binary([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa_binary([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)


def test_cohen_kappa_rejects_empty_input():
    with pytest.raises(ValueError):
        cohen_kappa_binary([], [])


def test_agreement_report_uses_only_filled_rows():
    silver = pd.DataFrame({"sample_id": ["1", "2", "3"], "sarcasm": [1, 0, 1], "humor": [0, 0, 1],
                           "local_reference": [0, 0, 0], "historical_reference": [1, 0, 0]})
    human = pd.DataFrame({"sample_id": ["1", "2", "3"], "sarcasm": [1, np.nan, 0],
                          "humor": [0, 0, 1], "local_reference": [0, 0, 0],
                          "historical_reference": [np.nan, np.nan, np.nan]})
    report = agreement_report(silver, human, COLS)
    assert report["sarcasm"]["n"] == 2
    assert report["sarcasm"]["agreement"] == 0.5
    assert report["humor"]["n"] == 3 and report["humor"]["agreement"] == 1.0
    assert report["historical_reference"]["n"] == 0


def test_agreement_report_rejects_no_overlap():
    silver = pd.DataFrame({"sample_id": ["1"], **{c: [0] for c in COLS}})
    human = pd.DataFrame({"sample_id": ["9"], **{c: [0] for c in COLS}})
    with pytest.raises(ValueError):
        agreement_report(silver, human, COLS)


def test_stratified_check_sample_counts_and_rejects_small_class():
    df = pd.DataFrame({"sample_id": [str(i) for i in range(30)],
                       "label": ["Authentic"] * 10 + ["Misleading"] * 10 + ["Satire"] * 10})
    sample = stratified_check_sample(df, "label", {"Authentic": 3, "Misleading": 4, "Satire": 2})
    assert len(sample) == 9
    assert sample["label"].value_counts().to_dict() == {"Authentic": 3, "Misleading": 4, "Satire": 2}
    with pytest.raises(ValueError):
        stratified_check_sample(df, "label", {"Satire": 11})


# ---- Wikipedia parameters and parsers ----------------------------------------------------

def test_search_params_contain_query_and_limit():
    params = build_search_params("বাংলাদেশ", 2)
    assert params["list"] == "search" and params["srsearch"] == "বাংলাদেশ" and params["srlimit"] == 2


def test_extract_params_join_page_ids():
    params = build_extract_params([10, 20])
    assert params["pageids"] == "10|20" and params["exintro"] == 1


def test_parse_search_ids_keeps_rank_order():
    payload = {"query": {"search": [{"pageid": 7, "title": "A"}, {"pageid": 3, "title": "B"}]}}
    assert parse_search_ids(payload) == ["7", "3"]


def test_parse_search_ids_handles_no_hits():
    assert parse_search_ids({"batchcomplete": True}) == []


def test_parse_extracts_skips_missing_and_empty_pages():
    payload = {"query": {"pages": [
        {"pageid": 1, "title": "A", "extract": "Lead text."},
        {"pageid": 2, "title": "B", "extract": "   "},
        {"title": "C", "missing": True},
    ]}}
    assert parse_extracts(payload) == {"1": {"title": "A", "extract": "Lead text."}}


# ---- passages and evidence rows ------------------------------------------------------------

def test_chunk_text_respects_max_chars_and_keeps_words():
    text = "প্রথম বাক্য। দ্বিতীয় বাক্যটি একটু বড়। " * 20
    chunks = chunk_text(text, max_chars=60)
    assert chunks and all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks).split() == " ".join(text.split()).split()


def test_chunk_text_splits_long_sentence_hard():
    chunks = chunk_text("a" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_text_empty_input():
    assert chunk_text("   ", max_chars=50) == []
    assert chunk_text(None, max_chars=50) == []


def test_chunk_text_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        chunk_text("text", max_chars=0)


def test_evidence_rows_schema_and_title_prefix():
    pages = {"5": {"title": "ঢাকা", "extract": "ঢাকা বাংলাদেশের রাজধানী। এটি বড় শহর।"}}
    ev = evidence_rows(pages, max_chars=500)
    assert list(ev.columns) == ["evidence_id", "group_id", "evidence_text", "source", "url"]
    assert ev.loc[0, "evidence_text"].startswith("ঢাকা. ")
    assert ev.loc[0, "group_id"] == "wiki_5"
    assert ev.loc[0, "url"] == "https://bn.wikipedia.org/?curid=5"
    assert ev["evidence_id"].is_unique


def test_evidence_rows_empty_pages_give_empty_frame():
    ev = evidence_rows({}, max_chars=500)
    assert ev.empty and "evidence_text" in ev.columns


def test_run_cached_on_error_continues_and_does_not_cache_failures(tmp_path):
    cache = tmp_path / "c.jsonl"

    def fn(payload):
        if payload == "bad":
            raise RuntimeError("timeout")
        return [payload]

    errors = []
    out = run_cached([("a", "a"), ("bad", "bad"), ("b", "b")], fn, cache, workers=1,
                     on_error=lambda key, exc: errors.append(key))
    assert out == {"a": ["a"], "b": ["b"]}
    assert errors == ["bad"]
    # the failed key is not in the cache, so a rerun computes it again
    assert "bad" not in cache.read_text(encoding="utf-8")


def test_run_cached_without_on_error_still_propagates(tmp_path):
    def fn(payload):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_cached([("a", "a")], fn, tmp_path / "c.jsonl", workers=1)
