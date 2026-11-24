"""Tests for the model-selection grid. The pure part runs without torch; the torch part is skipped
when torch is not installed."""
import numpy as np
import pytest

from src.selection import (
    aggregate_over_seeds,
    confusion_text,
    fusion_configs,
    mean_confusion,
    select_best,
    unimodal_configs,
)

TEXTS = ["banglabert", "mbert", "xlmr"]
IMAGES = ["clip", "vit", "swin"]
FUSION = ["early", "intermediate", "late"]


def test_grid_sizes():
    assert len(unimodal_configs(TEXTS, IMAGES)) == 3 + 3 + 2
    assert len(fusion_configs(TEXTS, IMAGES, FUSION)) == 27


def test_fusion_names_are_unique():
    names = [c["name"] for c in fusion_configs(TEXTS, IMAGES, FUSION)]
    assert len(set(names)) == len(names)


def _record(name, kind, seed, val_f1, test_f1=0.5):
    return {"name": name, "kind": kind, "seed": seed,
            "metrics": {"val_macro_f1": val_f1, "test_macro_f1": test_f1, "test_accuracy": 0.6}}


def test_aggregate_mean_and_std_over_seeds():
    recs = [_record("text:a", "text", 42, 0.5), _record("text:a", "text", 43, 0.7)]
    s = aggregate_over_seeds(recs)["text:a"]
    assert s["val_macro_f1_mean"] == pytest.approx(0.6)
    assert s["val_macro_f1_std"] == pytest.approx(0.1)
    assert s["seeds"] == [42, 43]


def test_select_best_uses_validation_not_test():
    recs = [
        _record("text:a", "text", 42, val_f1=0.60, test_f1=0.10),
        _record("text:b", "text", 42, val_f1=0.55, test_f1=0.99),
    ]
    summary = aggregate_over_seeds(recs)
    assert select_best(summary, "text") == "text:a"


def test_select_best_tie_goes_to_smaller_std():
    recs = [
        _record("text:a", "text", 42, 0.60), _record("text:a", "text", 43, 0.80),  # mean 0.70, std 0.1
        _record("text:b", "text", 42, 0.68), _record("text:b", "text", 43, 0.72),  # mean 0.70, std 0.02
    ]
    summary = aggregate_over_seeds(recs)
    assert select_best(summary, "text") == "text:b"


def test_select_best_ignores_other_kinds_and_handles_none():
    recs = [_record("image:x", "image", 42, 0.9)]
    summary = aggregate_over_seeds(recs)
    assert select_best(summary, "fusion") is None
    assert select_best(summary, "image") == "image:x"


def test_confusion_text_layout():
    text = confusion_text([[5, 1, 0], [2, 3, 0], [0, 0, 4]], ["A", "M", "S"])
    lines = text.splitlines()
    assert lines[0].startswith("true")
    assert len(lines) == 4
    assert lines[1].split()[-3:] == ["5", "1", "0"]


def test_mean_confusion_averages_counts():
    avg = mean_confusion([[[2, 0], [0, 2]], [[4, 0], [0, 0]]])
    assert avg.tolist() == [[3.0, 0.0], [0.0, 1.0]]


def test_confusion_text_shows_fractional_means():
    text = confusion_text(np.array([[2.5, 0.0], [0.0, 1.0]]), ["A", "B"])
    assert "2.5" in text


try:
    import torch
    _TORCH_OK = True
except Exception:  # missing or broken torch install: the torch tests are skipped, not failed
    torch = None
    _TORCH_OK = False

needs_torch = pytest.mark.skipif(not _TORCH_OK, reason="torch is not importable here")


@needs_torch
def test_masked_mean_pool_ignores_padding():
    from src.backbones import masked_mean_pool

    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    out = masked_mean_pool(hidden, mask)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


@needs_torch
def test_intermediate_head_output_shape():
    from src.fusion_heads import IntermediateHead

    model = IntermediateHead(d_text=8, d_image=5, n_classes=3, hidden=16)
    out = model(torch.zeros(4, 8), torch.zeros(4, 5))
    assert out.shape == (4, 3)


def _synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    text = rng.normal(size=(n, 6)).astype(np.float32) + y[:, None]
    image = rng.normal(size=(n, 4)).astype(np.float32)
    split = {
        "train": np.arange(n) < 80,
        "val": (np.arange(n) >= 80) & (np.arange(n) < 100),
        "test": np.arange(n) >= 100,
    }
    return y, text, image, split


@needs_torch
@pytest.mark.parametrize("kind", ["early", "intermediate", "late"])
def test_fit_and_predict_returns_valid_probabilities(kind):
    from src.fusion_heads import fit_and_predict

    y, text, image, split = _synthetic()
    out = fit_and_predict(kind, {"text": text, "image": image}, y, split,
                          seed=1, epochs=3, patience=3)
    assert out["val"].shape == (20, 3) and out["test"].shape == (20, 3)
    assert np.allclose(out["test"].sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= out["val_macro_f1"] <= 1.0


@needs_torch
def test_fit_and_predict_rejects_unknown_kind():
    from src.fusion_heads import fit_and_predict

    y, text, image, split = _synthetic()
    with pytest.raises(ValueError):
        fit_and_predict("mystery", {"text": text, "image": image}, y, split, epochs=1)
