"""Step 5: cultural silver labels from an LLM, plus a blind 40-row human check.

What it does:
1. Sends every post's text (never its class label) to the LLM and asks for 4 binary cultural features.
2. Saves the answers to ANNOTATION_DIR/cultural_silver_labels.csv. Script 02 uses this file
   only when no manual cultural_annotations.csv exists.
3. Writes a 40-row check sheet with the text only (no LLM answer, no class label). A human
   fills in the 4 columns. Once the sheet is filled, the script reports agreement with the LLM.

Resumable: answers are cached in CACHE_DIR/cultural_cache.jsonl. Rerun the script after a
disconnect; finished rows are not sent again.

Needs ANTHROPIC_API_KEY in the environment.
Run: python scripts/05_build_cultural_silver.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    ANNOTATION_DIR,
    CACHE_DIR,
    CULTURAL_CHECK_CSV,
    CULTURAL_COLS,
    CULTURAL_SILVER_CSV,
    LLM_MAX_INPUT_CHARS,
    LLM_BACKEND,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_WORKERS,
    LOCAL_LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
)
from src.llm_labeling import (  # noqa: E402
    make_client,
    agreement_report,
    call_with_retries,
    cultural_prompt,
    parse_cultural_answer,
    run_cached,
    stratified_check_sample,
)
from src.utils import save_json  # noqa: E402

CHECK_PER_LABEL = {"Authentic": 14, "Misleading": 14, "Satire": 12}  # 40 rows, stratified by class


def main():
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DIR / "unified_split.csv")
    df["sample_id"] = df["sample_id"].astype(str)
    df["text"] = df["text"].fillna("")

    client, llm_workers = make_client(
        max_tokens=120, backend=LLM_BACKEND, local_model=LOCAL_LLM_MODEL,
        anthropic_model=LLM_MODEL, local_max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
        anthropic_workers=LLM_WORKERS,
    )

    def label_one(text):
        return call_with_retries(
            lambda: parse_cultural_answer(
                client.complete(cultural_prompt(text, CULTURAL_COLS, LLM_MAX_INPUT_CHARS)),
                CULTURAL_COLS,
            ),
            max_retries=1 if LLM_BACKEND == "local" else LLM_MAX_RETRIES,
        )

    items = list(zip(df["sample_id"], df["text"]))
    answers = run_cached(items, label_one, CACHE_DIR / "cultural_cache.jsonl", workers=llm_workers)

    silver = pd.DataFrame([{"sample_id": sid, **answers[sid]} for sid in df["sample_id"]])
    silver.to_csv(CULTURAL_SILVER_CSV, index=False)
    rates = {c: round(float(silver[c].mean()), 4) for c in CULTURAL_COLS}
    print(f"Silver labels: {len(silver)} rows -> {CULTURAL_SILVER_CSV}")
    print("Positive rate per feature:", rates)
    save_json({"rows": len(silver), "positive_rate": rates, "model": LLM_MODEL,
               "input": "post text only, cut to %d chars" % LLM_MAX_INPUT_CHARS},
              RESULTS_DIR / "cultural_silver_summary.json")

    # ---- blind human check: created once, never overwritten ----------------
    if not CULTURAL_CHECK_CSV.exists():
        sample = stratified_check_sample(df, "label", CHECK_PER_LABEL, seed=SEED)
        sheet = pd.DataFrame({
            "sample_id": sample["sample_id"],
            "text": sample["text"].str.slice(0, 800),
        })
        for c in CULTURAL_COLS:
            sheet[c] = ""
        sheet.to_csv(CULTURAL_CHECK_CSV, index=False)
        print(f"Human check sheet written ({len(sheet)} rows): {CULTURAL_CHECK_CSV}")
        print("Fill the 4 columns with 0 or 1, then rerun this script to get agreement.")
        return

    human = pd.read_csv(CULTURAL_CHECK_CSV)
    human["sample_id"] = human["sample_id"].astype(str)
    filled = human[CULTURAL_COLS].notna().all(axis=1).sum()
    print(f"Human check: {filled} of {len(human)} rows fully labelled")
    if filled == 0:
        print("No filled rows yet; agreement not computed.")
        return
    report = agreement_report(silver, human, CULTURAL_COLS)
    save_json(report, RESULTS_DIR / "cultural_agreement.json")
    for c, r in report.items():
        print(c, r)


if __name__ == "__main__":
    main()
