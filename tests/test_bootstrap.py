"""Tests for src/bootstrap.py. Synthetic arrays only: these check the statistics, not the model."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.bootstrap import macro_f1_np, paired_bootstrap_macro_f1


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = rng.integers(0, 3, 200)
    ours = macro_f1_np(y, p, 3)
    ref = f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
    assert ours == pytest.approx(ref, abs=1e-12)


def test_macro_f1_absent_class_counts_as_zero_like_sklearn():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    ours = macro_f1_np(y, p, 3)
    ref = f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
    assert ours == pytest.approx(ref)
    assert ours == pytest.approx(2 / 3)


def test_identical_models_give_zero_difference_and_zero_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 100)
    p = rng.integers(0, 3, 100)
    res = paired_bootstrap_macro_f1(y, p, p, 3, n_boot=200, seed=42)
    assert res["diff_b_minus_a"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0


def test_clearly_better_model_has_positive_interval():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 300)
    good = y.copy()
    flip = rng.random(300) < 0.4
    bad = np.where(flip, (y + 1) % 3, y)
    res = paired_bootstrap_macro_f1(y, bad, good, 3, n_boot=300, seed=42)
    assert res["diff_b_minus_a"] > 0
    assert res["ci_low"] > 0
    assert res["p_b_better"] > 0.95


def test_bootstrap_is_reproducible_for_a_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 150)
    a = rng.integers(0, 3, 150)
    b = rng.integers(0, 3, 150)
    r1 = paired_bootstrap_macro_f1(y, a, b, 3, n_boot=100, seed=7)
    r2 = paired_bootstrap_macro_f1(y, a, b, 3, n_boot=100, seed=7)
    assert r1 == r2


def test_rejects_length_mismatch_and_empty():
    with pytest.raises(ValueError):
        paired_bootstrap_macro_f1([0, 1], [0], [0, 1], 3, n_boot=5)
    with pytest.raises(ValueError):
        paired_bootstrap_macro_f1([], [], [], 3, n_boot=5)


def test_class_level_bootstrap_uses_one_vs_rest_f1():
    from src.bootstrap import f1_class_np
    y = np.array([0, 1, 2, 2, 2, 0, 1, 2])
    p = np.array([0, 1, 2, 2, 0, 0, 1, 1])
    # class 2: true rows 2,3,4,7 (four); predicted 2 at rows 2,3 (tp=2, fp=0, fn=2) -> F1 = 4/6
    assert f1_class_np(y, p, 3, 2) == pytest.approx(4 / 6)
    res = paired_bootstrap_macro_f1(y, p, p, 3, n_boot=50, seed=1, cls=2)
    assert res["metric"] == "f1_class_2"
    assert res["diff_b_minus_a"] == 0.0
