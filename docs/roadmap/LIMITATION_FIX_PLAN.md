# Limitation fix plan (status as of 2026-11-26)

Each limitation has one of four statuses:
- **Done in code:** implemented and unit-tested in this repo. The run needs Colab.
- **Needs a Colab run:** the script exists. Output is not yet known.
- **Needs a human:** cannot be done by the assistant under the labelling rules (D-4, D-7).
- **Closed:** reported as a negative result or left out on purpose (decision recorded).

| # | Limitation | Status | What was done | Next step | Where it is decided |
|---|---|---|---|---|---|
| 1 | Image presence shortcut | Done (run) | `scripts/10`: image vectors shuffled within each split, presence kept; `text_plus_presence` control | Validation: +0.044 beyond presence. Test: +0.075 (CI +0.024 to +0.126); shuffled control +0.076 (CI +0.017 to +0.132). Presence on text alone not shown (CI includes 0). See CANONICAL_NUMBERS.md | D-6, D-10 |
| 2 | Text length shortcut | Done (run) | Length quartiles on validation rows (`scripts/10`) | Image gain positive in all four length quartiles (+0.013 to +0.075). Length is strongly tied to label (70% of Misleading in the two shortest quartiles). Stated as a confound | D-6 |
| 3 | Group split leaks near-copies | Done (run) | Cross-split near-duplicate audit, threshold 0.8 (`scripts/10`) | 5 pairs at similarity ≥ 0.8 (4 val, 1 test), all same label. Small; stated in README. Sensitivity of test contrasts to the test pair not computed | D-6 |
| 4 | Satire has 32 test rows | Done (run) | Paired bootstrap CI for macro-F1 and Satire F1 (`scripts/12`) | CIs in CANONICAL_NUMBERS.md and README. Satire differences treated as suggestive | D-8 |
| 5 | Evidence gate FAIL (13/40) | Needs a human | Dev/eval sheet of 240 rows, frozen split, retriever comparison plan (`scripts/11`, `src/evidence_eval.py`) | A person labels the dev rows first, then the eval rows; retriever chosen on dev; gate read once on eval | D-4, D-7 |
| 6 | Evidence has not shown a gain | Needs a human | Same as 5; the failure categories must be written from dev labels | Write the failure categories; the final evidence result may still be negative. That is a valid report | D-7 |
| 7 | Cultural features fail kappa check | Closed | Path B: feature stays off, reported as a negative result | Nothing more unless a human-labelled sample exists | D-2, D-9 |
| 8 | Text-only test number missing from the explainer | Partly done | Text-only numbers in README from the controls run | Image-only and selection-grid rows still need `results/selection/summary.json` | D-3 |
| 9 | Explainer overstates the group split | Done in code | README draft and explainer corrected | Check the README before pushing | — |
| 10 | 128-token truncation | Closed for this round | Stated as a limitation | Retrain with longer input only if time allows | — |
| 11 | Single dataset, no external test | Closed for this round | Stated as a limitation | A second dataset needs an owner decision | — |
| 12 | Explanations not evaluated by people | Closed for this round | Stated as a limitation | A human study needs a protocol and annotators | — |
| 13 | OCR accuracy not measured | Needs a human | CER script ready: `scripts/13_ocr_cer.py` + `src/ocr_metrics.py` (tested). Needs `data/annotations/ocr_gold.csv` with about 20 human-typed references | A person types reference_text for about 20 images (reference_source = human), then run script 13 | — |
| 14 | Model text includes OCR text | Done in code (documented) | `scripts/02` builds `model_text = text + OCR text` | Say so in the README and the viva notes | — |

## Status summary (2026-11-26)

- Done or closed: 1, 2, 3, 4, 7, 9, 10, 11, 12, 14 (10 of 14).
- Partly done, needs one file: 8 (`results/selection/summary.json`).
- Blocked on a human: 5, 6, 13.

## Run order in Colab

1. `python -m pytest -q tests/test_confound_controls.py tests/test_evidence_eval.py tests/test_bootstrap.py`
2. `python scripts/10_confound_controls.py` → `results/controls/controls.md`
3. `python scripts/12_bootstrap_ci.py` (lists names) → then `--a <baseline> --b <fusion>`
4. `python scripts/11_evidence_eval_sheet.py` → a person labels `data/annotations/evidence_eval_sheet.csv`
5. After labels exist: dev analysis, then the eval gate (the gate script for the new sheet is not written yet; it will be written only after the dev analysis)
