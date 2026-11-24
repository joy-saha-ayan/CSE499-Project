"""Build the unified table from the raw BTMD slim copy. This module does not import torch.

Output columns: sample_id, text, label_raw, label, group_id, source_url, image_path, has_image
"""
import hashlib
from pathlib import Path

import pandas as pd

from configs.btmd_schema import BTMD_SCHEMA
from configs.config import PROCESSED_DIR, RAW_DIR
from src.preprocessing import clean_bangla_text, map_label, normalize_url
from src.utils import save_json


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def _require(schema: dict, key: str):
    value = schema.get(key)
    if value is None:
        raise ValueError(
            f"'{key}' is not filled in configs/btmd_schema.py. "
            "Fill it in from the inspection output first."
        )
    return value


def _need_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Table columns: {df.columns.tolist()}")


def _resolve_image(value, raw_dir: Path, image_root: str):
    """One BTMD post can hold several images separated by ';'.

    The first image that exists on disk is used.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    for part in str(value).split(";"):
        part = part.strip()
        if not part:
            continue
        path = raw_dir / image_root / part
        if path.is_file():
            return str(path)
    return None


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_unified(schema=BTMD_SCHEMA, raw_dir=RAW_DIR, out_dir=PROCESSED_DIR):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    table_path = raw_dir / _require(schema, "table_file")
    df = load_table(table_path)

    id_col = _require(schema, "id_col")
    text_cols = _require(schema, "text_col")
    text_cols = text_cols if isinstance(text_cols, list) else [text_cols]
    label_col = _require(schema, "label_col")
    for col in [id_col, label_col, *text_cols]:
        _need_col(df, col)

    stats = {"rows_raw": len(df)}
    # Join the text columns (e.g. heading + text), treating missing values as empty strings.
    raw_text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    out = pd.DataFrame({
        "sample_id": df[id_col].astype(str),
        "text": raw_text.map(clean_bangla_text),
        "label_raw": df[label_col].astype(str),
    })
    out["label"] = out["label_raw"].map(map_label)
    excluded = out["label"].isna()
    stats["excluded_by_label"] = out.loc[excluded, "label_raw"].value_counts().to_dict()
    out = out[~excluded].copy()

    if schema.get("group_col"):
        _need_col(df, schema["group_col"])
        g = df.loc[out.index, schema["group_col"]]
        has_g = g.notna() & (g.astype(str).str.strip() != "")
        # If the group value is empty, fall back to a text hash. Otherwise all empty rows
        # would share one group and the split would be unbalanced.
        out["group_id"] = g.astype(str).str.strip().where(has_g, out["text"].map(_text_hash))
        stats["group_source"] = f"column '{schema['group_col']}' (text hash when empty)"
    else:
        out["group_id"] = out["text"].map(_text_hash)
        stats["group_source"] = "text_hash_fallback (weak: paraphrase/same-claim leak possible)"

    if schema.get("source_url_col"):
        _need_col(df, schema["source_url_col"])
        out["source_url"] = df.loc[out.index, schema["source_url_col"]].map(normalize_url)
    else:
        out["source_url"] = None

    image_col = schema.get("image_col")
    if image_col:
        _need_col(df, image_col)
        raw_img = df.loc[out.index, image_col]
        image_root = schema.get("image_root", "")
        out["image_path"] = raw_img.map(lambda v: _resolve_image(v, raw_dir, image_root))
        stats["image_value_present_but_file_missing"] = int(
            (raw_img.notna() & out["image_path"].isna()).sum()
        )
    else:
        out["image_path"] = None
    out["has_image"] = out["image_path"].notna()

    empty = out["text"].str.len() == 0
    stats["dropped_empty_text"] = int(empty.sum())
    out = out[~empty]

    before = len(out)
    out = out.drop_duplicates(subset=["text", "label"], keep="first")
    stats["dropped_duplicate_text_label"] = before - len(out)

    if out["has_image"].any():
        has_img = out["has_image"]
        out["image_md5"] = None
        out.loc[has_img, "image_md5"] = out.loc[has_img, "image_path"].map(_file_md5)
        dup_img = out["image_md5"].notna() & out.duplicated(subset=["image_md5"], keep="first")
        stats["dropped_duplicate_image"] = int(dup_img.sum())
        out = out[~dup_img]

    out = out.reset_index(drop=True)
    stats["rows_final"] = len(out)
    stats["rows_with_image"] = int(out["has_image"].sum())
    stats["label_counts"] = {k: int(v) for k, v in out["label"].value_counts().items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "unified.csv", index=False)
    save_json(stats, out_dir / "data_stats.json")
    return out, stats
