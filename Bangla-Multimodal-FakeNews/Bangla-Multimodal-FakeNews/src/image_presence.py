"""Explicit handling of posts that have no image.

encode_images() fills a missing image with a zero vector, and image_text_similarity() then
returns exactly 0 for that row. Standardisation turns the zero vector into a fixed pattern,
so a classifier can learn "no image" from the feature values. If image presence is correlated
with the label (it is: see the dataset card), part of an image model's score is then a
presence shortcut, not image content.

This module makes presence explicit in three ways:
  1. it verifies that the zero-vector rows of the image features are exactly the rows without an image;
  2. it saves presence as its own feature block, so a presence-only control experiment can run;
  3. it computes metrics on the image-present subset of the test split, so image models can be
     judged on posts where an image actually exists.
"""
import numpy as np

from src.metrics import classification_metrics


def zero_vector_rows(features) -> np.ndarray:
    """Return a boolean mask of rows whose values are all exactly zero."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    return np.all(features == 0, axis=1)


def verify_missing_image_encoding(features, has_image) -> dict:
    """Check that zero rows in the image features are exactly the rows without an image.

    Raises ValueError on any disagreement, because a disagreement means the feature file
    does not match the metadata and every image result built on it is suspect.
    Returns the row counts for the report.
    """
    has_image = np.asarray(has_image, dtype=bool)
    features = np.asarray(features)
    if features.shape[0] != has_image.shape[0]:
        raise ValueError(
            f"row count mismatch: features {features.shape[0]} vs has_image {has_image.shape[0]}"
        )
    zero = zero_vector_rows(features)
    bad_zero = int((zero & has_image).sum())      # an image row that encoded to zeros
    bad_nonzero = int((~zero & ~has_image).sum())  # a missing-image row with real values
    if bad_zero or bad_nonzero:
        raise ValueError(
            "missing-image encoding is inconsistent with metadata: "
            f"{bad_zero} rows with an image are all-zero, "
            f"{bad_nonzero} rows without an image are non-zero"
        )
    return {
        "rows": int(has_image.shape[0]),
        "with_image": int(has_image.sum()),
        "without_image": int((~has_image).sum()),
    }


def has_image_feature(has_image) -> np.ndarray:
    """Presence as a float32 column of shape (N, 1), suitable for the feature-block loader."""
    return np.asarray(has_image, dtype=np.float32).reshape(-1, 1)


def image_subset_metrics(y_true, y_pred, has_image_mask, labels) -> dict:
    """Accuracy and macro-F1 on the rows that have an image.

    y_true and y_pred are label strings; has_image_mask is a boolean array of the same length.
    """
    mask = np.asarray(has_image_mask, dtype=bool)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not mask.any():
        raise ValueError("no image-present rows in this subset")
    m = classification_metrics(y_true[mask], y_pred[mask], labels)
    return {"n": int(mask.sum()), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
