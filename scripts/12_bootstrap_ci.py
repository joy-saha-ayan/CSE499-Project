"""Step 12: paired bootstrap CIs on the test split, for two saved models (decision D-8).

Why: the test split has 588 rows and only 32 Satire rows. A single macro-F1 hides how much it moves.
The paired bootstrap reports the difference between two models on the same resampled rows.

Input : FEATURE_DIR/meta.csv (test rows, labels y), SELECTION_DIR/probs/<name>_seed<k>.npy
        written by script 08. The test rows are in meta order, as script 08 saved them.
Output: SELECTION_DIR/bootstrap_ci.json

Run (names are the config names used by script 08, with ':' and '+' replaced by '_'):
    python scripts/12_bootstrap_ci.py --a text_banglabert --b fusion_early_banglabert_clip
Without arguments the script lists the probability files it can find.

Note: the test split has now been read more than once (D-8 first read, D-10 second read).
The 'p_b_better' field is the share of bootstrap resamples where b beats a. It is not a p-value.

Rules: this is a read-only comparison of saved test probabilities. It changes no selection. The
seeds are averaged first (script 08 averages probabilities over seeds in the same way).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import FEATURE_DIR, LABELS_3, SEEDS, SELECTION_DIR  # noqa: E402
from src.bootstrap import paired_bootstrap_macro_f1  # noqa: E402
from src.utils import save_json  # noqa: E402

PROBS_DIR = SELECTION_DIR / "probs"
SATIRE = LABELS_3.index("Satire")


def load_mean_probs(name: str, probs_dir: Path) -> np.ndarray:
    files = [probs_dir / f"{name}_seed{s}.npy" for s in SEEDS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing probability files for '{name}': {missing}")
    return np.mean([np.load(f) for f in files], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", help="baseline config name")
    parser.add_argument("--b", help="comparison config name")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--probs-dir", help="folder with <name>_seed<k>.npy files (default: script 08 probs)")
    parser.add_argument("--out", help="output json (default: results/selection/bootstrap_ci.json)")
    args = parser.parse_args()
    probs_dir = Path(args.probs_dir) if args.probs_dir else PROBS_DIR
    out_path = Path(args.out) if args.out else SELECTION_DIR / "bootstrap_ci.json"

    if not probs_dir.exists():
        raise SystemExit(f"{probs_dir} not found. Run script 08 (or 10) first.")
    if not args.a or not args.b:
        names = sorted({p.name.rsplit("_seed", 1)[0] for p in probs_dir.glob("*_seed*.npy")})
        print("Available names:")
        for n in names:
            print("  ", n)
        return

    meta = pd.read_csv(FEATURE_DIR / "meta.csv")
    y_test = meta.loc[meta["split"] == "test", "y"].astype(int).to_numpy()

    pa = load_mean_probs(args.a, probs_dir)
    pb = load_mean_probs(args.b, probs_dir)
    if len(pa) != len(y_test) or len(pb) != len(y_test):
        raise SystemExit("probability rows do not match the number of test rows in meta.csv")
    a_pred, b_pred = pa.argmax(axis=1), pb.argmax(axis=1)

    result = {
        "a": args.a,
        "b": args.b,
        "test_rows": int(len(y_test)),
        "satire_test_rows": int((y_test == SATIRE).sum()),
        "macro_f1": paired_bootstrap_macro_f1(y_test, a_pred, b_pred, len(LABELS_3), n_boot=args.n_boot),
        "satire_f1": paired_bootstrap_macro_f1(y_test, a_pred, b_pred, len(LABELS_3),
                                               n_boot=args.n_boot, cls=SATIRE),
        "note": "Test split, read for pre-registered contrasts (D-8, D-10). Not used to choose any model. "
                "p_b_better is the share of bootstrap resamples where b beats a, not a p-value.",
    }
    save_json(result, out_path)
    m = result["macro_f1"]
    s = result["satire_f1"]
    print(f"Macro-F1 {args.b} - {args.a}: {m['diff_b_minus_a']:+.4f}  95% CI [{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]")
    print(f"Satire F1 {args.b} - {args.a}: {s['diff_b_minus_a']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  (n_satire={result['satire_test_rows']})")


if __name__ == "__main__":
    main()
