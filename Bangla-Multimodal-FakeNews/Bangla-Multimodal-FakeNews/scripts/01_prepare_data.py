"""Step 1: raw BTMD slim copy -> unified table -> group-aware split.

Run: python scripts/01_prepare_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import PROCESSED_DIR, SEED  # noqa: E402
from src.data_prep import build_unified  # noqa: E402
from src.splitting import group_split  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402


def main():
    set_seed(SEED)
    df, stats = build_unified()
    split_df = group_split(df, seed=SEED)
    split_df.to_csv(PROCESSED_DIR / "unified_split.csv", index=False)

    summary = split_df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary)
    save_json({"per_split_label": summary.to_dict()}, PROCESSED_DIR / "split_summary.json")
    print("\nData stats:", stats)


if __name__ == "__main__":
    main()
