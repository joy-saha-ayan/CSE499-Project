"""Pure-logic tests: run without torch or a GPU. Run: pytest -q tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from configs.config import LABELS_3  # noqa: E402
from src.cultural_context import fit_cultural_classifiers  # noqa: E402
from src.data_prep import build_unified  # noqa: E402
from src.decision import choose_min_confidence, final_decision  # noqa: E402
from src.evidence_retrieval import drop_label_source_evidence, hybrid_scores, recall_at_k  # noqa: E402
from src.evidence_verification import (  # noqa: E402
    aggregate_evidence,
    map_nli_labels,
    verdict_from_scores,
)
from src.metrics import classification_metrics, expected_calibration_error  # noqa: E402
from src.preprocessing import clean_bangla_text, map_label  # noqa: E402
from src.splitting import check_split, group_split  # noqa: E402


def make_df(n_groups=60, per_group=5):
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            rows.append({
                "sample_id": f"{g}-{k}",
                "text": f"text {g} {k}",
                "label": LABELS_3[(g + k) % 3],
                "group_id": f"g{g}",
            })
    return pd.DataFrame(rows)


def test_map_label_rules():
    assert map_label("Real") == "Authentic"
    assert map_label("Misinformation") == "Misleading"
    assert map_label("Clickbait") is None
    with pytest.raises(ValueError):
        map_label("misinformation")  # a lowercase typo must be caught


def test_clean_text_keeps_negation_and_drops_url():
    s = clean_bangla_text("  না  এটা সত্যি নয় https://x.com/a   ")
    assert "না" in s and "https" not in s and s == s.strip()


def test_group_split_has_no_leakage():
    df = group_split(make_df(), seed=1)
    check_split(df)
    assert set(df["split"]) == {"train", "val", "test"}


def test_check_split_detects_leakage():
    df = make_df(9, 3)
    df["split"] = np.repeat(["train", "val", "test"], 9)
    df.loc[0, "split"] = "test"  # group g0 appears in both train and test
    with pytest.raises(AssertionError):
        check_split(df)


def test_final_decision_rules():
    assert final_decision("Authentic", "supports", 0.9, 0.5) == "Authentic"
    assert final_decision("Misleading", "refutes", 0.9, 0.5) == "Misleading"
    assert final_decision("Authentic", "refutes", 0.9, 0.5) == "Insufficient"
    assert final_decision("Satire", "insufficient", 0.9, 0.5) == "Satire"
    assert final_decision("Authentic", "supports", 0.3, 0.5) == "Insufficient"


def test_min_confidence_matches_target_coverage():
    conf = np.linspace(0.34, 1.0, 100)
    t = choose_min_confidence(conf, target_coverage=0.85)
    assert abs((conf >= t).mean() - 0.85) < 0.03


def test_metrics_and_ece():
    m = classification_metrics(
        ["Authentic", "Misleading", "Satire", "Satire"],
        ["Authentic", "Satire", "Satire", "Satire"],
        LABELS_3,
    )
    assert m["accuracy"] == 0.75
    assert m["confusion_matrix"][1][2] == 1  # Misleading predicted as Satire
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0


def test_cultural_train_only_fitting():
    """Changing the labels of val/test or unannotated rows must not change their predictions."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "sample_id": [str(i) for i in range(n)],
        "split": ["train"] * 120 + ["val"] * 40 + ["test"] * 40,
    })
    emb = rng.normal(size=(n, 8))
    cols = ["sarcasm", "humor"]
    annot = pd.DataFrame({
        "sample_id": [str(i) for i in range(160)],
        "sarcasm": rng.integers(0, 2, 160),
        "humor": rng.integers(0, 2, 160),
    })
    probs1, _ = fit_cultural_classifiers(emb, df, annot, cols, seed=0)

    annot2 = annot.copy()
    mask = annot2["sample_id"].astype(int) >= 120
    for c in cols:
        annot2.loc[mask, c] = 1 - annot2.loc[mask, c]
    probs2, _ = fit_cultural_classifiers(emb, df, annot2, cols, seed=0)

    assert np.allclose(probs1[120:], probs2[120:])


def test_label_source_guard_drops_url_variants():
    ev = pd.DataFrame({
        "url": ["https://a.com/x/?utm=1", "https://b.com/y", None],
        "evidence_text": ["a", "b", "c"],
    })
    kept, dropped = drop_label_source_evidence(ev, ["https://A.com/x"])
    assert dropped == 1 and len(kept) == 2


def test_recall_and_hybrid_scores():
    assert recall_at_k(["g1", "g2", "g3"], "g3", k=5) == 1.0
    assert recall_at_k(["g1", "g2", "g3"], "g3", k=2) == 0.0
    assert np.allclose(hybrid_scores([1, 2, 3], [3, 2, 1], alpha=0.5), [0.5, 0.5, 0.5])


def test_nli_mapping_and_verdicts():
    assert map_nli_labels({0: "entailment", 1: "neutral", 2: "contradiction"}) == {
        "entailment": 0, "neutral": 1, "contradiction": 2,
    }
    agg = aggregate_evidence(np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]))
    assert np.allclose(agg, [0.8, 0.7, 0.15])
    assert aggregate_evidence([]).tolist() == [0.0, 0.0, 1.0]
    assert verdict_from_scores(0.8, 0.1, 0.5) == "supports"
    assert verdict_from_scores(0.1, 0.8, 0.5) == "refutes"
    assert verdict_from_scores(0.2, 0.2, 0.5) == "insufficient"


def test_build_unified_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "content": ["বাংলা সংবাদ এক", "বাংলা সংবাদ এক", "ভুয়া খবর", "মজার মিম", "ক্লিকবেট"],
        "lab": ["Real", "Real", "Misinformation", "Satire", "Clickbait"],
        "url": ["https://x.com/a", "https://x.com/a", "https://x.com/b", None, "https://x.com/c"],
    }).to_csv(raw / "t.csv", index=False)
    schema = {
        "table_file": "t.csv", "id_col": "id", "text_col": "content", "label_col": "lab",
        "image_col": None, "image_root": "", "group_col": None, "source_url_col": "url",
    }
    out, stats = build_unified(schema=schema, raw_dir=raw, out_dir=tmp_path / "proc")
    assert set(out["label"]) == {"Authentic", "Misleading", "Satire"}
    assert len(out) == 3
    assert stats["excluded_by_label"] == {"Clickbait": 1}
    assert stats["dropped_duplicate_text_label"] == 1


def test_build_unified_requires_schema_fields(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"id": ["1"], "content": ["x"], "lab": ["Real"]}).to_csv(raw / "t.csv", index=False)
    schema = {"table_file": "t.csv", "id_col": "id", "text_col": None, "label_col": "lab"}
    with pytest.raises(ValueError):
        build_unified(schema=schema, raw_dir=raw, out_dir=tmp_path / "proc")


def test_resolve_image_handles_multi_image_values(tmp_path):
    """A ';'-separated value must return the first image that exists on disk."""
    from src.data_prep import _resolve_image

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "b.png").write_bytes(b"x")
    value = "images/a.png;images/b.png"
    assert _resolve_image(value, tmp_path, "") == str(tmp_path / "images" / "b.png")
    assert _resolve_image("images/none.png", tmp_path, "") is None
    assert _resolve_image(None, tmp_path, "") is None


def test_group_falls_back_to_text_hash_when_heading_missing(tmp_path):
    """When the heading is empty, the row gets a text hash instead of sharing one big group."""
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({
        "id": ["1", "2", "3"],
        "heading": ["খবর এক", None, None],
        "text": [None, "আলাদা লেখা", "তৃতীয় লেখা"],
        "lab": ["Real", "Misinformation", "Satire"],
    }).to_csv(raw / "t.csv", index=False)
    schema = {
        "table_file": "t.csv", "id_col": "id", "text_col": ["heading", "text"],
        "label_col": "lab", "image_col": None, "image_root": "", "group_col": "heading",
        "source_url_col": None,
    }
    out, _ = build_unified(schema=schema, raw_dir=raw, out_dir=tmp_path / "proc")
    assert out["group_id"].nunique() == len(out)
    assert "খবর এক" in out["group_id"].tolist()
