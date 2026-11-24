# Next Task Specification

## 1. Task ID
T-VAL-01

## 2. Task title
Image-presence confound control: explicit missing-image handling and image-subset evaluation

## 3. Why this task is next
The priority order in the master prompt puts critical research validity first. The audit found one validity problem that changes the meaning of results the report already relies on:

- Posts without an image receive a zero CLIP vector and a similarity of exactly 0 (`src/image_encoder.py`, `scripts/02_extract_features.py`).
- Image presence is strongly tied to the label (Authentic 62.9 percent, Misleading 55.2 percent, Satire 90.7 percent of posts have an image) [COLAB].
- So E2 (CLIP image only, macro-F1 0.5413) and E3 (fusion, macro-F1 0.7448) can score partly on presence, not on image content.

The other critical items were not chosen:
- The length confound needs the length-only baseline to be run and reported, which is a Colab action. It does not need new code.
- The research direction conflict (evidence source) and the label policy conflict need the owner's decision. An implementation task cannot resolve them.
- The source-URL guard blocks E5 and E6, but it affects only experiments that are not yet started. The image shortcut contaminates completed results.

This task needs no new data and no new model, so it can be finished and tested here.

## 4. Previous task dependencies
- Dataset build and `meta.csv` with `has_image`, `split`, `label`, `y`, `group_id` [COLAB, present on Drive].
- CLIP image features `clip_img.npy` and similarity `sim.npy` [COLAB].
- Fusion script with feature blocks and `classification_metrics` [in repository].
- No dependency on E4, E5 or E6.

## 5. Exact implementation scope
1. New pure module `src/image_presence.py`:
   - `zero_vector_rows(features)`: mask of all-zero rows.
   - `verify_missing_image_encoding(features, has_image)`: raises on any disagreement between zero rows and missing images; returns counts.
   - `has_image_feature(has_image)`: float32 column of shape (N, 1).
   - `image_subset_metrics(y_true, y_pred, has_image_mask, labels)`: accuracy and macro-F1 on image-present rows.
2. `scripts/02_extract_features.py`: after encoding, verify zero rows against metadata and save `has_image.npy`.
3. `scripts/04_train_fusion.py`:
   - add experiment E0 (`has_image` only) as a control;
   - for every experiment, record image-subset accuracy and macro-F1 on the image-present test rows;
   - add image-subset fields to each summarised experiment in `fusion_results.json`.
   - The `summary.md` table format is unchanged.
4. Tests in `tests/test_image_presence.py`.
5. README: E0 row and limitation item 10.

Out of scope: changing the fusion architecture, changing `BASE_BLOCKS`, retraining E1, changing labels, evidence, cultural features, or any research decision.

## 6. Required files
- `src/image_presence.py` (new)
- `tests/test_image_presence.py` (new)
- `scripts/02_extract_features.py` (modified)
- `scripts/04_train_fusion.py` (modified)
- `README.md` (modified)
- `docs/audits/PROJECT_PROGRESS_AUDIT.md`, `docs/roadmap/NEXT_TASK_SPEC.md`, `docs/audits/NEXT_TASK_EXECUTION_REPORT.md` (new)

## 7. Expected outputs
Local:
- Passing unit tests (24 total).
- Modified scripts that compile and parse.

On Colab (after the zip is uploaded to Drive):
- `features/has_image.npy` with shape (3910, 1).
- A printed presence summary from script 02 with with-image and without-image counts that match `meta.csv`.
- `results/fusion_results.json` containing `E0_has_image_only` and `image_subset_*` fields for E1f, E2, E3.

## 8. Testing requirements
- Unit tests for every function in `src/image_presence.py`, including the failure cases (row-count mismatch, image row that is all zero, missing row with values, empty subset).
- Syntax check and AST parse of the modified scripts.
- Smoke test of `summarize()` from script 04 with fake runs (function extracted without importing torch).
- Colab: run script 02 (presence check must pass) and script 04 (E0 and subset fields must appear).

## 9. Acceptance criteria
1. `python -m pytest -q tests` passes, including the new tests.
2. `verify_missing_image_encoding` raises a `ValueError` for each inconsistent case and returns correct counts for a consistent case.
3. Script 02 saves `has_image.npy` and fails loudly if the zero rows disagree with metadata.
4. Script 04 writes `image_subset` for every experiment and `E0_has_image_only` when `has_image.npy` exists.
5. On Colab: the presence check passes on the real data, and the E0 and image-subset numbers are reported (even if they are poor).
6. No change to training code, feature blocks or labels beyond the above.

## 10. Known limitations
- Presence is only a control. It does not prove the shortcut size; the Colab numbers do.
- Image-subset evaluation reduces test size: about 61 percent of rows, with only about 32 Satire rows in test overall, so per-class subset results are noisy.
- Zero vectors remain in the feature matrix for E2 and E3. This task does not remove them, to keep existing results comparable. A later task can drop them or use a presence-gated fusion.
- Local runs cannot execute scripts 02 and 04 (torch is not installed locally). The end-to-end run is UNVERIFIED until Colab.
- The length confound is not addressed by this task.

## 11. Next task after successful completion
T-DEC-02: resolve the two owner decisions (evidence source; cultural label policy), then add the source-URL guard flag and run E4 and E5 on Colab. This task must not start until the owner decides both points.
