"""Tests for src/ocr_metrics.py. Small hand-made strings only. These check the arithmetic, not OCR quality."""
from src.ocr_metrics import cer, corpus_cer, edit_distance, normalise, wer


def test_edit_distance_known_values():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("same", "same") == 0


def test_normalise_collapses_whitespace_and_keeps_bangla():
    assert normalise("  আমি   ভালো  \n আছি ") == "আমি ভালো আছি"


def test_cer_identical_is_zero_and_one_substitution():
    assert cer("বাংলা", "বাংলা") == 0.0
    # reference has 5 characters, one substitution gives 1/5
    assert abs(cer("abcde", "abcdx") - 0.2) < 1e-12


def test_cer_empty_reference_rules():
    assert cer("", "") == 0.0
    assert cer("", "x") == float("inf")


def test_wer_counts_words():
    assert abs(wer("one two three four", "one two tree four") - 0.25) < 1e-12


def test_corpus_cer_pools_by_characters():
    pairs = [("abcd", "abcd"), ("ab", "ax")]  # 0 edits / 4 chars, 1 edit / 2 chars
    out = corpus_cer(pairs)
    assert out["images"] == 2
    assert out["reference_chars"] == 6
    assert abs(out["pooled_cer"] - 1 / 6) < 1e-12
    assert abs(out["mean_per_image_cer"] - 0.25) < 1e-12
