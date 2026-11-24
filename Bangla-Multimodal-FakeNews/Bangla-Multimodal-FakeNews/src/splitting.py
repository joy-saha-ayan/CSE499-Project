"""Group-aware train/val/test split. The same group (claim or event) never appears in two splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from configs.config import LABELS_3


def group_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    outer = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    train_idx, temp_idx = next(outer.split(df, groups=df["group_id"]))
    temp = df.iloc[temp_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed
    )
    val_rel, test_rel = next(inner.split(temp, groups=temp["group_id"]))

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[temp_idx[val_rel]] = "val"
    split[temp_idx[test_rel]] = "test"
    out = df.copy()
    out["split"] = split
    check_split(out)
    return out


def check_split(df: pd.DataFrame) -> None:
    """Raise AssertionError on group leakage, an empty split, or a missing class."""
    groups = {s: set(df.loc[df["split"] == s, "group_id"]) for s in ("train", "val", "test")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = groups[a] & groups[b]
        if overlap:
            raise AssertionError(f"Group leakage: {a} and {b} share {len(overlap)} group(s)")
    for s in ("train", "val", "test"):
        part = df.loc[df["split"] == s]
        if len(part) == 0:
            raise AssertionError(f"'{s}' split is empty")
        missing = set(LABELS_3) - set(part["label"])
        if missing:
            raise AssertionError(f"'{s}' split has no samples of classes: {sorted(missing)}")
