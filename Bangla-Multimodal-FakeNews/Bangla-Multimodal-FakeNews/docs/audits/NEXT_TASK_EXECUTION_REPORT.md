# Next Task Execution Report: T-VAL-01

Scope: image-presence confound control (see `docs/roadmap/NEXT_TASK_SPEC.md`).
Evidence tags: [CODE] verified in this sandbox, [COLAB] from user-run outputs, UNVERIFIED = not run.

## A. Completed

- `src/image_presence.py` created: zero-row detection, presence verification against metadata, `has_image` feature column, image-subset metrics. [CODE]
- `scripts/02_extract_features.py` modified: verifies zero CLIP rows against `has_image` metadata (raises on mismatch), saves `has_image.npy`, prints presence counts. [CODE: compile only]
- `scripts/04_train_fusion.py` modified: adds control experiment E0 (`has_image` only); records image-subset accuracy and macro-F1 for every run; adds `image_subset_*` fields to the summary. [CODE: compile, AST-extracted `summarize()` smoke test with fake runs]
- `tests/test_image_presence.py` created with 9 tests. [CODE]
- README: E0 row and limitation item 10 added. [CODE]
- `docs/audits/PROJECT_PROGRESS_AUDIT.md`, `docs/roadmap/NEXT_TASK_SPEC.md` and this report written. [CODE]

## B. Tests

Command run in the project root:

```
python3 -m pytest -q tests
```

Result: 24 passed in 0.91 s (15 pre-existing tests in `tests/test_pure_logic.py`, 9 new tests in `tests/test_image_presence.py`).

Compile check:

```
python3 -m py_compile src/image_presence.py scripts/02_extract_features.py scripts/04_train_fusion.py
```

Result: OK.

Not run:
- `scripts/02_extract_features.py` and `scripts/04_train_fusion.py` end-to-end. Reason: `torch` is not installed in this sandbox (`ModuleNotFoundError`). UNVERIFIED until the Colab run.
- Windows local environment: not checked in this task.

## C. Files changed, added, deleted

Added:
- `src/image_presence.py`
- `tests/test_image_presence.py`
- `docs/roadmap/NEXT_TASK_SPEC.md`
- `docs/audits/NEXT_TASK_EXECUTION_REPORT.md`
- `docs/audits/PROJECT_PROGRESS_AUDIT.md`

Changed:
- `scripts/02_extract_features.py`
- `scripts/04_train_fusion.py`
- `README.md`

Deleted: none.

Git status: UNVERIFIED. The sandbox copy is not a Git repository, and the user's local repository at `D:\` was not available. No commit, push or branch was created.

## D. Remaining work for this task (Colab, user-run)

1. Upload the rebuilt zip to Drive and unpack it.
2. Run script 02. Expected: presence check passes, `features/has_image.npy` with shape (3910, 1), and with-image / without-image counts match `meta.csv`. If the check raises, stop and report the message.
3. Run script 04. Expected: `E0_has_image_only` in `results/fusion_results.json`, and `image_subset_n`, `image_subset_accuracy_mean`, `image_subset_macro_f1_mean`, `image_subset_macro_f1_std` for E1f, E2 and E3.
4. Report the printed output and the E0 and subset numbers. These numbers, not this report, decide whether the image-presence shortcut is material.

## E. Research impact

- Known from earlier Colab output [COLAB]: image presence differs by label (Authentic 62.9 percent, Misleading 55.2 percent, Satire 90.7 percent with an image). E2 and E3 may therefore partly learn presence, not image content.
- This task does not change any existing result. It adds a control (E0) and a subset metric so the size of the shortcut can be measured.
- Until E0 and the image-subset numbers are reported, the E2 and E3 headline results must be described as "includes missing-image shortcut risk", not as image-content effects.
- The length-only baseline (notebook cell 16) is still not reported and is needed for the length confound.
- Image-subset Satire counts are very small (32 Satire rows in test overall), so subset per-class numbers are noisy.

## F. Remaining issues (not solved here)

- Zero vectors for missing images remain in E2 and E3 features. Not removed, to keep results comparable.
- Owner decision needed: evidence source (Wikipedia vs fact-checking sources, per research direction).
- Owner decision needed: cultural label policy (silver LLM labels vs README limitation 8, "must not be guessed").
- E4, E5 and E6 still blocked (no cultural annotation, no evidence file, `source_url` guard).
- Per-sample predictions for fusion models (for confusion matrices) still to be added (Phase 2).

## G. Recommended next task (not started)

T-DEC-02: resolve the two owner decisions, then add the source-URL guard flag and run E4 and E5 on Colab. Do not start until both decisions are made.
