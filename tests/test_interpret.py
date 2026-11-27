"""Tests for src/interpret.py. numpy only, no GPU and no model needed."""
import numpy as np
import pytest

from src.interpret import (
    accuracy_macro_f1,
    bangla_ratio,
    chunk_spans,
    confidence_band,
    highlight_html,
    input_checks,
    nearest_neighbours,
    occlusion_attribution,
    per_class_from_confusion,
    seed_agreement,
    top_drivers,
    truncation_curve,
)


def test_chunk_spans_cover_every_word_once():
    spans = chunk_spans(10, 3)
    assert spans == [(0, 4), (4, 8), (8, 10)]
    assert len(spans) <= 3


def test_chunk_spans_short_text_gives_single_words():
    assert chunk_spans(4, 30) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_chunk_spans_empty_and_bad_limit():
    assert chunk_spans(0, 5) == []
    with pytest.raises(ValueError):
        chunk_spans(5, 0)


def test_occlusion_positive_delta_for_supporting_word():
    # Target class 0 has probability 0.9 only while the word 'ভালো' is present.
    def prob_fn(t):
        return np.array([0.9, 0.1, 0.0]) if "ভালো" in t else np.array([0.2, 0.8, 0.0])

    res = occlusion_attribution("ভালো সংবাদ এখানে আছে", prob_fn, target_index=0, max_units=30)
    assert res["full_prob"] == pytest.approx(0.9)
    assert res["spans"][0]["text"] == "ভালো"
    assert res["spans"][0]["delta"] == pytest.approx(0.7)
    for s in res["spans"][1:]:
        assert s["delta"] == pytest.approx(0.0)
    support, oppose = top_drivers(res["spans"], k=5)
    assert [s["text"] for s in support] == ["ভালো"]
    assert oppose == []


def test_occlusion_negative_delta_for_opposing_word():
    # Target probability is 0.1 while 'খারাপ' is present, so removing it RAISES the target to 0.9.
    def prob_fn(t):
        return np.array([0.1, 0.9, 0.0]) if "খারাপ" in t else np.array([0.9, 0.1, 0.0])

    res = occlusion_attribution("এটা খারাপ খবর", prob_fn, target_index=0, max_units=30)
    assert res["spans"][1]["delta"] == pytest.approx(-0.8)
    support, oppose = top_drivers(res["spans"], k=5)
    assert support == []
    assert [s["text"] for s in oppose] == ["খারাপ"]


def test_occlusion_rejects_empty_text():
    with pytest.raises(ValueError):
        occlusion_attribution("   ", lambda t: np.array([1.0, 0.0, 0.0]), 0)


def test_highlight_escapes_html_and_uses_colours():
    spans = [
        {"text": "<script>x</script>", "delta": 0.5},
        {"text": "খারাপ", "delta": -0.2},
    ]
    out = highlight_html(spans)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "rgba(22,163,74" in out      # green for supporting span
    assert "rgba(220,38,38" in out      # red for opposing span


def test_truncation_curve_word_counts_and_probs():
    def prob_fn(t):
        n = len(t.split())
        return np.array([n / 10, 0.0, 1 - n / 10])

    rows = truncation_curve("একটি দুটি তিনটি চারটি", prob_fn, fractions=(0.5, 1.0))
    assert [r["words"] for r in rows] == [2, 4]
    assert rows[0]["probs"][0] == pytest.approx(0.2)
    assert rows[1]["probs"][0] == pytest.approx(0.4)


def test_seed_agreement_majority_and_spread():
    per_seed = [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1]]
    res = seed_agreement(per_seed)
    assert res["preds"] == [0, 1, 0]
    assert res["majority"] == 0
    assert res["agreement"] == pytest.approx(2 / 3)
    assert res["spread_majority_prob"] == pytest.approx(0.4)   # 0.6 - 0.2


def test_nearest_neighbours_orders_by_cosine():
    ref = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    hits = nearest_neighbours(np.array([1.0, 0.0]), ref, k=2)
    assert [i for i, _ in hits] == [0, 2]
    assert hits[0][1] == pytest.approx(1.0)


def test_nearest_neighbours_handles_zero_rows():
    ref = np.array([[0.0, 0.0], [1.0, 0.0]])
    hits = nearest_neighbours(np.array([1.0, 0.0]), ref, k=2)
    assert hits[0][0] == 1          # the zero row gets similarity 0, so it ranks last


def test_bangla_ratio_counts_letters_only():
    # 'বাংলা' has 5 Bengali letter code points (vowel signs included); 'abc' has 3 Latin letters.
    assert bangla_ratio("বাংলা abc 123!") == pytest.approx(5 / 8)
    assert bangla_ratio("123 !!!") == 0.0


def test_input_checks_clean_bangla_input_has_no_warnings():
    text = "এটি একটি দীর্ঘ বাংলা পোস্ট যা পরীক্ষার জন্য লেখা হয়েছে এবং এতে যথেষ্ট অক্ষর আছে।"
    assert input_checks(text, has_image=True, n_tokens=10) == []


def test_input_checks_flags_english_short_and_missing_image():
    msgs = input_checks("hi", has_image=False, n_tokens=3)
    levels = [lvl for lvl, _ in msgs]
    assert levels.count("warn") == 2          # low Bangla ratio and very short post
    assert levels.count("info") == 1          # no image


def test_input_checks_reports_truncation():
    text = "এটি একটি দীর্ঘ বাংলা পোস্ট যা পরীক্ষার জন্য লেখা হয়েছে এবং এতে যথেষ্ট অক্ষর আছে।"
    msgs = input_checks(text, has_image=True, n_tokens=200, max_tokens=128)
    assert len(msgs) == 1
    assert msgs[0][0] == "info"
    assert "128" in msgs[0][1]


def test_accuracy_and_macro_f1_small_example():
    # Class 0: P=1, R=0.5, F1=2/3. Class 1: P=2/3, R=1, F1=0.8. Class 2: F1=1. Accuracy 4/5.
    acc, f1 = accuracy_macro_f1([0, 0, 1, 1, 2], [0, 1, 1, 1, 2], n_classes=3)
    assert acc == pytest.approx(0.8)
    assert f1 == pytest.approx((2 / 3 + 0.8 + 1.0) / 3, abs=1e-6)


def test_macro_f1_ignores_classes_absent_from_the_batch():
    # Only Authentic (0) and Misleading (1) occur. Satire (2) must not drag the macro average down.
    acc, f1 = accuracy_macro_f1([1, 0, 0], [1, 0, 0], n_classes=3)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_per_class_precision_and_recall_from_confusion():
    cm = [[5, 1, 0], [2, 3, 0], [0, 0, 4]]
    rows = per_class_from_confusion(cm, ["A", "B", "C"])
    assert rows[0]["precision"] == pytest.approx(round(5 / 7, 3))   # column 0 sums to 7
    assert rows[0]["recall"] == pytest.approx(round(5 / 6, 3))      # row 0 sums to 6


def test_input_mode_routes_each_case():
    from src.interpret import input_mode

    assert input_mode("", False) == "empty"
    assert input_mode("   ", False) == "empty"
    assert input_mode("", True) == "image_only"
    assert input_mode("পোস্ট", False) == "text_only"
    assert input_mode("পোস্ট", True) == "text_and_image"


def test_confidence_band_is_a_display_label():
    assert confidence_band(0.85) == "বেশি"
    assert confidence_band(0.70) == "মাঝারি"
    assert confidence_band(0.30) == "কম"
