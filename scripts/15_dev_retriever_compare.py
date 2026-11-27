"""Step 15: compare BM25 and hybrid retrievers on DEV rows only (D-11).

Input : ANNOTATION_DIR/evidence_eval_sheet.csv      (dev rows labelled; BM25 top-3; eval rows must stay blank)
        RESULTS_DIR/evidence/dev_hybrid_top3.csv     (written by scripts/14: hybrid top-3 per dev post)
        --new-labelled CSV                           (dev_new_passages file after labelling; relevant_0_or_1 filled)
Output: RESULTS_DIR/evidence/dev_retriever_compare.json

Rules:
- Dev rows only. The script refuses if any eval row has a label.
- Every hybrid top-3 pair needs a label, from the sheet or from the new-labelled file. Otherwise it stops.
- The new-labelled file must be exactly the hybrid pairs that are not in the sheet. Otherwise it stops.
- No threshold and no selection rule is applied here. The numbers are reported. The decision is written in DECISIONS.md.
- Intervals: post-clustered bootstrap, 2000 resamples, seed 42 (same design as the dev analysis).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.config import ANNOTATION_DIR, RESULTS_DIR  # noqa: E402
from src.utils import save_json  # noqa: E402

SHEET = ANNOTATION_DIR / "evidence_eval_sheet.csv"
HYBRID = RESULTS_DIR / "evidence" / "dev_hybrid_top3.csv"
OUT = RESULTS_DIR / "evidence" / "dev_retriever_compare.json"
PASSAGE_CHARS = 400  # same cut as scripts/14 and src/evidence_eval.py
TOP_K = 3
N_BOOT = 2000
SEED = 42


def _key(post_id, text):
    return (str(post_id), str(text)[:PASSAGE_CHARS])


def _labels(series, where):
    vals = series.astype(str).str.strip()
    bad = ~vals.isin(["0", "1", "0.0", "1.0"])
    if bad.any():
        raise SystemExit(f"{where}: {int(bad.sum())} rows have a label that is not 0 or 1. Fix them first.")
    return vals.astype(float).astype(int)


def load_sheet_dev():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if (sheet.loc[sheet["role"] == "eval", "relevant_0_or_1"].str.strip() != "").any():
        raise SystemExit("Eval rows have labels. This comparison reads dev only. Stop.")
    dev = sheet[sheet["role"] == "dev"].copy()
    if (dev["relevant_0_or_1"].str.strip() == "").any():
        raise SystemExit("Some dev rows in the sheet are unlabelled. Finish dev labelling first.")
    dev["label"] = _labels(dev["relevant_0_or_1"], "sheet dev")
    dev["rank"] = dev["rank"].astype(int)
    dev["key"] = [_key(p, t) for p, t in zip(dev["post_id"], dev["passage"])]
    if dev["key"].duplicated().any():
        raise SystemExit("Duplicate (post, passage) pairs in the sheet dev rows.")
    return dev


def load_new_labelled(path):
    new = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if (new["relevant_0_or_1"].str.strip() == "").any():
        raise SystemExit("The new-labelled file has unlabelled rows. Label all of them first.")
    new["label"] = _labels(new["relevant_0_or_1"], "new labelled")
    new["key"] = [_key(p, t) for p, t in zip(new["post_id"], new["passage"])]
    if new["key"].duplicated().any():
        raise SystemExit("Duplicate (post, passage) pairs in the new-labelled file.")
    return new


def per_post_table(df, rank_col):
    """One row per post: share of relevant in top-3, relevant at rank 1, any relevant in top-3."""
    out = []
    for post_id, g in df.groupby("post_id"):
        g = g.sort_values(rank_col)
        ranks = g[rank_col].tolist()
        if ranks != list(range(1, TOP_K + 1)):
            raise SystemExit(f"Post {post_id} does not have ranks 1-3 exactly: {ranks}")
        labels = g["label"].to_numpy()
        out.append({"post_id": post_id, "share": labels.mean(), "p_at_1": labels[0], "any": float(labels.max()),
                    "r1": labels[0], "r2": labels[1], "r3": labels[2]})
    return pd.DataFrame(out).sort_values("post_id").reset_index(drop=True)


def point_and_ci(values, idx_matrix):
    point = float(values.mean())
    boots = values[idx_matrix].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(point, 3), [round(float(lo), 3), round(float(hi), 3)]


def summarise(table, idx_matrix):
    return {
        "share_all_3": dict(zip(["value", "ci95"], point_and_ci(table["share"].to_numpy(), idx_matrix))),
        "precision_at_1": dict(zip(["value", "ci95"], point_and_ci(table["p_at_1"].to_numpy(), idx_matrix))),
        "any_relevant_top3": dict(zip(["value", "ci95"], point_and_ci(table["any"].to_numpy(), idx_matrix))),
        "relevant_by_rank": {"1": int(table["r1"].sum()), "2": int(table["r2"].sum()), "3": int(table["r3"].sum())},
        "posts_with_no_relevant": int((table["any"] == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-labelled", required=True, help="Path to the labelled dev_new_passages CSV")
    parser.add_argument("--hybrid", default=str(HYBRID))
    args = parser.parse_args()

    dev = load_sheet_dev()
    bm25 = dev[["post_id", "rank", "label", "key"]].copy()
    sheet_map = dict(zip(dev["key"], dev["label"]))

    new = load_new_labelled(args.new_labelled)
    new_map = dict(zip(new["key"], new["label"]))
    if set(new_map) & set(sheet_map):
        raise SystemExit("Some new-labelled pairs are already in the sheet. They must be removed from the new file.")

    hyb = pd.read_csv(args.hybrid, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    hyb["rank"] = hyb["hybrid_rank"].astype(int)
    hyb["key"] = [_key(p, t) for p, t in zip(hyb["post_id"], hyb["passage"])]
    if set(hyb["post_id"]) != set(dev["post_id"]):
        raise SystemExit("Hybrid posts and sheet dev posts differ. The files come from different runs.")
    if hyb["key"].duplicated().any():
        raise SystemExit("Duplicate (post, passage) pairs in the hybrid top-3 file.")

    expected_new = set(hyb["key"]) - set(sheet_map)
    if set(new_map) != expected_new:
        missing = len(expected_new - set(new_map))
        extra = len(set(new_map) - expected_new)
        raise SystemExit(f"New-labelled file does not match the hybrid run: {missing} hybrid pairs missing, "
                         f"{extra} extra. Use the file written by the same scripts/14 run.")

    hyb["label"] = [sheet_map.get(k, new_map.get(k)) for k in hyb["key"]]
    if hyb["label"].isna().any():
        raise SystemExit("Some hybrid pairs still have no label.")
    hyb["label"] = hyb["label"].astype(int)

    bm25_table = per_post_table(bm25, "rank")
    hyb_table = per_post_table(hyb, "rank")
    if bm25_table["post_id"].tolist() != hyb_table["post_id"].tolist():
        raise SystemExit("Post sets differ between BM25 and hybrid tables.")

    rng = np.random.default_rng(SEED)
    n_posts = len(bm25_table)
    idx_matrix = rng.integers(0, n_posts, size=(N_BOOT, n_posts))

    diff_table = pd.DataFrame({
        "share": hyb_table["share"] - bm25_table["share"],
        "p_at_1": hyb_table["p_at_1"] - bm25_table["p_at_1"],
        "any": hyb_table["any"] - bm25_table["any"],
    })
    diff = {
        "share_all_3": dict(zip(["value", "ci95"], point_and_ci(diff_table["share"].to_numpy(), idx_matrix))),
        "precision_at_1": dict(zip(["value", "ci95"], point_and_ci(diff_table["p_at_1"].to_numpy(), idx_matrix))),
        "any_relevant_top3": dict(zip(["value", "ci95"], point_and_ci(diff_table["any"].to_numpy(), idx_matrix))),
    }

    from_sheet = int(hyb["key"].isin(set(sheet_map)).sum())
    result = {
        "posts": n_posts,
        "bm25_top3_pairs": int(len(bm25)),
        "hybrid_top3_pairs": int(len(hyb)),
        "hybrid_pairs_from_sheet": from_sheet,
        "hybrid_pairs_from_new_labels": int(len(hyb) - from_sheet),
        "bm25": summarise(bm25_table, idx_matrix),
        "hybrid": summarise(hyb_table, idx_matrix),
        "paired_difference_hybrid_minus_bm25": diff,
        "n_boot": N_BOOT,
        "seed": SEED,
        "note": ("Dev rows only. No threshold or selection rule is applied here. "
                 "Interval design: post-clustered bootstrap. The decision is recorded in DECISIONS.md."),
    }
    save_json(result, OUT)
    print(f"Wrote {OUT}")
    print(f"BM25   share {result['bm25']['share_all_3']['value']}  p@1 {result['bm25']['precision_at_1']['value']}  "
          f"any {result['bm25']['any_relevant_top3']['value']}")
    print(f"Hybrid share {result['hybrid']['share_all_3']['value']}  p@1 {result['hybrid']['precision_at_1']['value']}  "
          f"any {result['hybrid']['any_relevant_top3']['value']}")


if __name__ == "__main__":
    main()
