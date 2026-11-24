"""Tests for src/image_presence.py. Pure numpy, no GPU and no CLIP model needed."""
import numpy as np
import pytest

from src.image_presence import (
    has_image_feature,
    image_subset_metrics,
    verify_missing_image_encoding,
    zero_vector_rows,
)

LABELS = ["Authentic", "Misleading", "Satire"]


def _features():
    # rows 0 and 2 have an image (real values); rows 1 and 3 are missing (zero vectors)
    return np.array(
        [[0.2, 0.1], [0.0, 0.0], [0.5, 0.4], [0.0, 0.0]], dtype=np.float32
    )


def test_zero_vector_rows_marks_only_all_zero_rows():
    mask = zero_vector_rows(_features())
    assert mask.tolist() == [False, True, False, True]


def test_zero_vector_rows_rejects_non_2d_input():
    with pytest.raises(ValueError):
        zero_vector_rows(np.zeros(4))


def test_verify_passes_when_zero_rows_match_metadata():
    has_image = np.array([True, False, True, False])
    info = verify_missing_image_encoding(_features(), has_image)
    assert info == {"rows": 4, "with_image": 2, "without_image": 2}


def test_verify_raises_when_an_image_row_is_all_zero():
    has_image = np.array([True, False, True, False])
    features = _features()
    features[0] = 0.0  # row 0 is marked as having an image but encoded to zeros
    with pytest.raises(ValueError, match="inconsistent"):
        verify_missing_image_encoding(features, has_image)


def test_verify_raises_when_a_missing_row_has_values():
    has_image = np.array([True, False, True, False])
    features = _features()
    features[1] = [0.3, 0.3]  # row 1 is marked as missing but holds real values
    with pytest.raises(ValueError, match="inconsistent"):
        verify_missing_image_encoding(features, has_image)


def test_verify_raises_on_row_count_mismatch():
    with pytest.raises(ValueError, match="row count"):
        verify_missing_image_encoding(_features(), np.array([True, False]))


def test_has_image_feature_shape_and_values():
    block = has_image_feature([True, False, True])
    assert block.shape == (3, 1)
    assert block.dtype == np.float32
    assert block.ravel().tolist() == [1.0, 0.0, 1.0]


def test_image_subset_metrics_uses_only_image_rows():
    y_true = np.array(["Authentic", "Satire", "Misleading", "Authentic"])
    y_pred = np.array(["Authentic", "Authentic", "Misleading", "Misleading"])
    has_image = np.array([True, False, True, True])
    sub = image_subset_metrics(y_true, y_pred, has_image, LABELS)
    # rows used: 0 (correct), 2 (correct), 3 (wrong) -> accuracy 2/3
    assert sub["n"] == 3
    assert sub["accuracy"] == pytest.approx(2 / 3)


def test_image_subset_metrics_rejects_empty_subset():
    y = np.array(["Authentic", "Satire"])
    with pytest.raises(ValueError, match="no image-present rows"):
        image_subset_metrics(y, y, np.array([False, False]), LABELS)
