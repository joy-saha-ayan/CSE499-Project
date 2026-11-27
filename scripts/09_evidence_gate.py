"""Step 9: compute the evidence relevance gate from the human-labelled 40-row sheet (decision D-4).

Input:  ANNOTATION_DIR/evidence_relevance_check.csv, with the column relevant_0_or_1 filled by a human
        (1 = the Wikipedia passage is about the post's claim, 0 = not).
Output: RESULTS_DIR/evidence_gate.json. The app's Evidence tab reads it.

The threshold comes from configs/config.py (EVIDENCE_RELEVANCE_MIN, pre-set in D-4). This script does
not choose a threshold and does not change the sheet. Every row must be labelled.
Run: python scripts/09_evidence_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    EVIDENCE_GATE_JSON,
    EVIDENCE_RELEVANCE_CHECK_CSV,
    EVIDENCE_RELEVANCE_MIN,
    RESULTS_DIR,
)
from src.evidence_app import GATE_FAIL, GATE_PASS, relevance_share  # noqa: E402
from src.utils import save_json  # noqa: E402

LABEL_COLUMN = "relevant_0_or_1"


def main():
    if not EVIDENCE_RELEVANCE_CHECK_CSV.exists():
        raise FileNotFoundError(f"relevance sheet not found: {EVIDENCE_RELEVANCE_CHECK_CSV}")
    sheet = pd.read_csv(EVIDENCE_RELEVANCE_CHECK_CSV, dtype=str, keep_default_na=False)
    if LABEL_COLUMN not in sheet.columns:
        raise KeyError(f"sheet has no column '{LABEL_COLUMN}'")

    share, n, n_rel = relevance_share(sheet[LABEL_COLUMN].tolist())
    status = GATE_PASS if share >= EVIDENCE_RELEVANCE_MIN else GATE_FAIL
    gate = {
        "check": "evidence relevance (Wikipedia passage vs post claim)",
        "threshold": EVIDENCE_RELEVANCE_MIN,
        "threshold_source": "D-4",
        "n": n,
        "relevant": n_rel,
        "relevance_share": round(share, 4),
        "status": status,
        "sheet": str(EVIDENCE_RELEVANCE_CHECK_CSV.name),
        "note": "Evidence is shown in the app but not used in the prediction unless the gate passes "
                "and an E5 validation gain is recorded (D-5).",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(gate, EVIDENCE_GATE_JSON)
    print(f"Relevance gate: {n_rel} of {n} rows relevant, share {share:.3f}, "
          f"threshold {EVIDENCE_RELEVANCE_MIN} -> {status}")
    print(f"Written: {EVIDENCE_GATE_JSON}")


if __name__ == "__main__":
    main()
