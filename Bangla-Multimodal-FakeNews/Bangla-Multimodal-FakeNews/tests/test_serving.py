"""Tests for src/serving.py. The pure helpers need only numpy.

Tests that compare with the training code import torch, so they are skipped if torch cannot load.
"""
import numpy as np
import pytest

from src.serving import (
    IMAGE_DIM,
    TEXT_DIM,
    apply_stats,
    average_probs,
    build_fused_input,
    checkpoint_stem,
    fit_train_stats,
    leaderboard_rows,
)


def test_stats_use_only_train_rows():
    rng = np.random.default_rng(0)
    block = rng.normal(size=(50, 4)).astype(np.float32)
    mask = np.zeros(50, dtype=bool)
    mask[:30] = True
    mu1, sd1 = fit_train_stats(block, mask)
    changed = block.copy()
    changed[30:] = 999.0  # val and test rows must not affect the statistics
    mu2, sd2 = fit_train_stats(changed, mask)
    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(sd1, sd2)


def test_zero_rows_for_missing_images_are_counted_in_train_stats():
    # Training includes the zero vectors of rows without an image in the statistics.
    block = np.array([[0.0, 1.0], [0.0, 3.0], [2.0, 5.0], [4.0, 7.0]], dtype=np.float32)
    mask = np.array([True, True, True, True])
    mu, sd = fit_train_stats(block, mask)
    np.testing.assert_allclose(mu, block.mean(axis=0))
    np.testing.assert_allclose(sd, block.std(axis=0))


def test_zero_std_is_replaced_by_one():
    block = np.ones((5, 2), dtype=np.float32)
    mu, sd = fit_train_stats(block, np.ones(5, dtype=bool))
    assert np.all(sd == 1.0)
    out = apply_stats(block, mu, sd)
    assert np.isfinite(out).all()


def test_apply_stats_standardises_and_returns_float32():
    block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mu = np.array([2.0, 3.0], dtype=np.float32)
    sd = np.array([1.0, 2.0], dtype=np.float32)
    out = apply_stats(block, mu, sd)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[-1.0, -0.5], [1.0, 0.5]])


def test_fused_input_has_text_block_first_and_correct_width():
    text = np.arange(TEXT_DIM, dtype=np.float32)
    image = np.arange(IMAGE_DIM, dtype=np.float32)
    t_stats = (np.zeros(TEXT_DIM, np.float32), np.ones(TEXT_DIM, np.float32) * 2)
    i_stats = (np.zeros(IMAGE_DIM, np.float32), np.ones(IMAGE_DIM, np.float32) * 4)
    x = build_fused_input(text, image, t_stats, i_stats)
    assert x.shape == (1, TEXT_DIM + IMAGE_DIM)
    np.testing.assert_allclose(x[0, :TEXT_DIM], text / 2)
    np.testing.assert_allclose(x[0, TEXT_DIM:], image / 4)


def test_average_probs_is_equal_weight_and_sums_to_one():
    p1 = np.array([0.7, 0.2, 0.1])
    p2 = np.array([0.1, 0.8, 0.1])
    p3 = np.array([0.4, 0.4, 0.2])
    avg = average_probs([p1, p2, p3])
    np.testing.assert_allclose(avg, [(0.7 + 0.1 + 0.4) / 3, (0.2 + 0.8 + 0.4) / 3, (0.1 + 0.1 + 0.2) / 3])
    assert abs(avg.sum() - 1.0) < 1e-9


def test_checkpoint_stem_matches_training_file_names():
    assert checkpoint_stem("fusion:early:banglabert+clip") == "fusion_early_banglabert_clip"
    assert checkpoint_stem("text:banglabert") == "text_banglabert"
    assert checkpoint_stem("image:clip") == "image_clip"


def test_leaderboard_orders_by_kind_then_validation_f1_and_marks_winner():
    def row(kind, val, test):
        return {
            "kind": kind,
            "val_macro_f1_mean": val, "val_macro_f1_std": 0.01,
            "test_accuracy_mean": 0.8, "test_macro_f1_mean": test, "test_macro_f1_std": 0.02,
        }

    summary = {
        "fusion:a": row("fusion", 0.70, 0.66),
        "text:x": row("text", 0.60, 0.55),
        "fusion:b": row("fusion", 0.75, 0.72),
    }
    rows = leaderboard_rows(summary, winner="fusion:b")
    assert [r["config"] for r in rows] == ["fusion:b", "fusion:a", "text:x"]
    assert rows[0]["winner"] is True
    assert rows[1]["winner"] is False


def test_torch_standardisation_matches_training_code():
    try:
        from src.fusion_model import standardize  # imports torch at module level
    except Exception as exc:  # torch may be missing or fail to load (CUDA libraries)
        pytest.skip(f"torch not available: {type(exc).__name__}")
    rng = np.random.default_rng(1)
    train = rng.normal(size=(40, 6)).astype(np.float32)
    val = rng.normal(size=(10, 6)).astype(np.float32)
    mask = np.zeros(50, dtype=bool)
    mask[:40] = True
    full = np.vstack([train, val])
    mu, sd = fit_train_stats(full, mask)
    _, mu_ref, sd_ref = standardize(train, val)
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sd, sd_ref, rtol=1e-5, atol=1e-6)
