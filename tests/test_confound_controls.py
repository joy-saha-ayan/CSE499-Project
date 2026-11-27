"""Tests for src/confound_controls.py. All data here is small and synthetic, only to check the logic.
These tests say nothing about model accuracy.
"""
import numpy as np

from src.confound_controls import (
    cross_split_near_duplicates,
    exact_text_overlap,
    length_quartile_labels,
    near_duplicate_pair_table,
    shuffle_images_within_present,
)


def _feats(n, d=4):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_shuffle_keeps_presence_and_zero_rows():
    feats = _feats(10)
    has = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    feats[~has] = 0.0
    out, _ = shuffle_images_within_present(feats, has, seed=1)
    assert np.all(out[~has] == 0.0)
    assert np.all(np.abs(out[has]).sum(axis=1) > 0)


def test_shuffle_is_a_permutation_of_present_vectors():
    feats = _feats(12)
    has = np.ones(12, dtype=bool)
    has[[2, 7]] = False
    feats[~has] = 0.0
    out, perm = shuffle_images_within_present(feats, has, seed=3)
    before = sorted(map(tuple, np.round(feats[has], 6)))
    after = sorted(map(tuple, np.round(out[has], 6)))
    assert before == after
    assert sorted(perm[has]) == sorted(np.where(has)[0])


def test_shuffle_is_deterministic_for_a_seed():
    feats = _feats(20)
    has = np.ones(20, dtype=bool)
    a, _ = shuffle_images_within_present(feats, has, seed=7)
    b, _ = shuffle_images_within_present(feats, has, seed=7)
    c, _ = shuffle_images_within_present(feats, has, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shuffle_actually_changes_the_row_to_image_match():
    feats = _feats(50)
    has = np.ones(50, dtype=bool)
    out, perm = shuffle_images_within_present(feats, has, seed=42)
    assert np.mean(perm == np.arange(50)) < 0.2
    assert not np.array_equal(out, feats)


def test_shuffle_within_groups_keeps_group_membership():
    feats = _feats(8)
    has = np.ones(8, dtype=bool)
    groups = np.array(["train"] * 4 + ["test"] * 4)
    _, perm = shuffle_images_within_present(feats, has, groups=groups, seed=5)
    assert set(perm[:4]) == {0, 1, 2, 3}
    assert set(perm[4:]) == {4, 5, 6, 7}


def test_shuffle_rejects_length_mismatch():
    try:
        shuffle_images_within_present(_feats(5), np.ones(4, dtype=bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_length_quartiles_split_evenly():
    bins = length_quartile_labels([1, 2, 3, 4, 5, 6, 7, 8], n_bins=4)
    assert list(bins) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_exact_overlap_ignores_extra_spaces():
    ref = ["সরকার নতুন নীতি ঘোষণা করেছে", "আরেকটি খবর"]
    query = ["সরকার   নতুন নীতি ঘোষণা করেছে", "একদম নতুন লেখা", ""]
    assert exact_text_overlap(ref, query) == [0]


def test_near_duplicate_finds_small_edit_not_unrelated_text():
    ref = [
        "সরকার নতুন নীতি ঘোষণা করেছে আজ",
        "ক্রিকেট ম্যাচে বাংলাদেশ জয় পেয়েছে",
    ]
    query = [
        "সরকার নতুন নীতি ঘোষণা করেছে আজ!",
        "শেয়ারবাজারে আজ বড় পতন হয়েছে",
    ]
    hits = cross_split_near_duplicates(ref, query, threshold=0.8)
    pairs = {(q, r) for q, r, _ in hits}
    assert (0, 0) in pairs
    assert (1, 1) not in pairs
    assert all(q != 1 for q, _, _ in hits)


def test_near_duplicate_pair_table_maps_local_to_meta_rows():
    # query rows 5 and 9 of meta are local 0 and 1; reference rows 2 and 7 are local 0 and 1
    query_global = np.array([5, 9])
    ref_global = np.array([2, 7])
    sample_ids = np.array([f"id{i}" for i in range(10)])
    labels = np.array([0, 0, 1, 0, 0, 2, 0, 1, 0, 1])
    hits = [(0, 1, 0.9123), (1, 0, 0.8)]
    rows = near_duplicate_pair_table(hits, "val", query_global, ref_global, sample_ids, labels)
    assert rows[0]["query_sample_id"] == "id5" and rows[0]["ref_sample_id"] == "id7"
    assert rows[0]["query_y"] == 2 and rows[0]["ref_y"] == 1 and rows[0]["same_label"] == 0
    assert rows[1]["query_sample_id"] == "id9" and rows[1]["ref_sample_id"] == "id2"
    assert rows[1]["query_y"] == 1 and rows[1]["ref_y"] == 1 and rows[1]["same_label"] == 1
    assert rows[0]["similarity"] == 0.9123 and rows[0]["split"] == "val"

