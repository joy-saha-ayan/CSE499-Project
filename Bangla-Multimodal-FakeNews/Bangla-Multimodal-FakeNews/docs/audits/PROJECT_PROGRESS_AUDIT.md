# Project Progress Audit: Bangla Multimodal Fake News Detection

Audit date: 2026-11-18. Audited copy: the project folder in the working sandbox (the same code as the zip delivered earlier, plus this task's changes). The Windows repository at `D:\Capstone\Bangla-Multimodal-FakeNews` was not reachable from this environment, so Git state for that repository is UNVERIFIED.

Evidence labels used below:
- **[LOCAL]**: reproduced in this sandbox (pytest, static checks).
- **[COLAB]**: output the user pasted from Colab runs. The data and features are on the user's Google Drive, so these numbers are not re-derived here.
- **[CODE]**: read from the source code only.
- **UNVERIFIED**: could not be checked.

---

## 1. Executive Summary

**Overall stage.** The data pipeline is built and has run on the real data [COLAB]. The first four experiments (E1, E1f, E2, E3) have produced results [COLAB]. The cultural module (E4), evidence verification (E5) and the decision layer (E6) have not run. The project is at the end of the baseline stage, before the evidence stage.

**Genuinely completed (with evidence):**
- Dataset build, label mapping, group split and dataset statistics [COLAB]: 6,058 raw rows become 3,910 rows, and the drop counts reconcile exactly.
- Unit tests: 15 passed at the start of this audit, and 24 pass after this task [LOCAL].
- Frozen BanglaBERT, CLIP image, CLIP text and similarity features were built and used by E1f, E2 and E3 [COLAB].
- E1 (fine-tuned BanglaBERT, 3 seeds), E1f, E2 and E3 (3 seeds each) produced `summary.md` [COLAB].

**Not complete:**
- E4 cultural features: no annotation file exists.
- E5 evidence verification and E6 decision layer: no evidence file exists, and the feature script stops when `source_url` is missing (BTMD has no source URL).
- The length-only baseline is implemented but its result has not been reported.
- The E1 run that produced the reported numbers did not save per-sample predictions, so confusion matrices for E1 come only from the metric dictionaries.

**Most important blockers and risks:**
1. Missing-image shortcut: image models can learn "no image" from a zero vector (see Section 3.4). E2 and E3 results may be partly inflated. Addressed in this task, but final numbers need a Colab run.
2. Length confound: Authentic posts have median 167 words, Misleading 16 and Satire 10. The length-only baseline is the test for this.
3. Research direction conflict: the master research direction says evidence comes from "external fact-checking sources", but the implementation plan uses Bangla Wikipedia and excludes fact-check sites. This changes the research design and needs the owner's explicit decision.
4. Label policy conflict: `README.md` section 8 says cultural annotations "must not be guessed", but the silver-label plan uses an LLM to label training posts. This needs a decision before E4 runs.
5. Group split is nearly per-post: 3,906 groups for 3,910 posts (median group size 1), so "no shared story across splits" is not guaranteed.

**Recommended next task:** T-VAL-01, the image-presence confound control (see `docs/roadmap/NEXT_TASK_SPEC.md`). Reasons are in Section 7.

---

## 2. Task Completion Matrix

| Component | Status | Evidence | Problem | Required Action |
|---|---|---|---|---|
| Config (`configs/config.py`) | IMPLEMENTED BUT NOT VERIFIED | [CODE] single source of paths and hyperparameters; uses `BFND_BASE` env var | The master prompt expects `configs/config.yaml`, which does not exist. The project uses Python config. | Record the decision in the README. No change needed for correctness. |
| BTMD schema (`configs/btmd_schema.py`) | COMPLETED AND VERIFIED | [COLAB] build ran with `label_col = multi_category`, 3,910 rows | None found | None |
| Label mapping and Clickbait exclusion | COMPLETED AND VERIFIED | [LOCAL] tests; [COLAB] `excluded_by_label: Clickbait 576` | Clickbait exclusion is a documented decision, not reviewed with the owner | Confirm the exclusion in the report |
| Text cleaning (`src/preprocessing.py`) | COMPLETED AND VERIFIED | [LOCAL] tests for NFC, URL removal, whitespace, negation and emoji retention | Emoji and punctuation kept for BanglaBERT; the effect is not measured | Optional ablation |
| Data build (`src/data_prep.py`, script 01) | COMPLETED AND VERIFIED | [COLAB] `rows_raw 6058 - 576 - 1566 - 1 - 5 = 3910`; [LOCAL] end-to-end test | None found | None |
| Group-aware split (`src/splitting.py`) | NEEDS REVISION | [COLAB] 3,906 groups, 0 cross-split groups; [LOCAL] tests | Grouping by heading makes nearly every post its own group, so the split does not stop same-story leakage | Group by a normalised story key or cluster similar text; document the limit meanwhile |
| Leakage guard for label source (`drop_label_source_evidence`) | IMPLEMENTED BUT NOT VERIFIED | [LOCAL] unit test with a URL column | Cannot run on BTMD: there is no `source_url` | Needs a flag and a source policy (see E5 row) |
| Frozen BanglaBERT features (script 02) | COMPLETED AND VERIFIED | [COLAB] E1f ran on `bert.npy` | Shape check output not in the record | Save shapes to `meta` report |
| CLIP image and text features (script 02) | COMPLETED AND VERIFIED | [COLAB] E2 and E3 ran; the earlier modality error was fixed by separating the image and text CLIP models | Missing images are zero vectors (see the shortcut row) | See next row |
| Missing-image handling | NEEDS REVISION, addressed by T-VAL-01 | [CODE] `encode_images` writes zeros; `image_text_similarity` returns 0 for those rows | Shortcut risk for E2 and E3 | T-VAL-01: verify, save `has_image`, run E0, report image-subset metrics |
| E1 fine-tuned BanglaBERT (script 03) | NEEDS REVISION | [COLAB] acc 0.8095 ± 0.0084, macro-F1 0.5457 ± 0.0057 | Predictions were not saved, so no per-sample analysis. Run was on CPU for seeds 42 and 43, so timing notes were wrong. | Resumable run with per-seed predictions is implemented (not yet verified on Colab) |
| E1f, E2, E3 (script 04) | NEEDS REVISION | [COLAB] E1f 0.6271, E2 0.5413, E3 0.7448 macro-F1 | Shortcut and length confounds not yet controlled | T-VAL-01 plus the length baseline |
| Length-only baseline (notebook cell) | IMPLEMENTED BUT NOT VERIFIED | Code in notebook cell 16 | Output not reported | Run and report |
| Cultural features E4 (script 02, `src/cultural_context.py`) | NOT STARTED | No `cultural_annotations.csv` | Label policy conflicts with README; annotation not done | Decide label policy, then annotate or label |
| Evidence retrieval (`src/evidence_retrieval.py`) | IMPLEMENTED BUT NOT VERIFIED | [LOCAL] recall@k and hybrid score tests | No evidence file; retrieval quality not measured on real claims | Build evidence; measure recall@k |
| NLI verification (`src/evidence_verification.py`) | IMPLEMENTED BUT NOT VERIFIED | [LOCAL] mapping and verdict tests | Bangla NLI performance not measured | Build claim-evidence pairs and measure |
| E5 evidence features (script 02 and 04) | BLOCKED | Script 02 raises when `source_url` is all NaN | Evidence source and guard policy undecided | Decision on evidence source plus a guard flag |
| E6 decision layer (`src/decision.py`) | BLOCKED | [LOCAL] rule tests; no run | Depends on E5 | After E5 |
| Fusion model (`src/fusion_model.py`) | COMPLETED AND VERIFIED (runs) | [COLAB] E1f, E2, E3 trained with early stopping | Confounds (above) | None for the model itself |
| Metrics and ECE (`src/metrics.py`) | COMPLETED AND VERIFIED | [LOCAL] tests including ECE | None | None |
| Explainability (`src/explainability.py`) | IMPLEMENTED BUT NOT VERIFIED | [LOCAL] markdown builder only | No real output checked | Check on a real verdict |
| Gradio demo (`app/app.py`) | IMPLEMENTED BUT NOT VERIFIED | Code only | Never run; needs E5 artifacts | Run after E5 |
| Preflight (`scripts/00_preflight.py`) | COMPLETED AND VERIFIED | [COLAB] models reachable, schema OK, Tesla T4 detected | Earlier CPU hang on `import torch` | None |
| Dataset overview figures (notebook cells) | PARTIALLY COMPLETED | [COLAB] tables and one figure produced; one cell failed with a pandas error, fixed in the reply | Confusion matrices for fusion models are not yet saved | Save per-sample predictions (Phase 2) |
| Unit tests (`tests/`) | COMPLETED AND VERIFIED | [LOCAL] 15 passed before, 24 passed after | Tests cover pure logic only, not GPU training | Add integration tests on a small fixture later |
| Documentation (`README.md`) | NEEDS REVISION | Limitations section; E0 added in this task | Section 8 item 8 conflicts with the silver-label plan | Owner decision |

---

## 3. Repository Audit

### 3.1 Existing files [LOCAL]
`README.md`, `requirements.txt`, `.gitignore`, `configs/` (`config.py`, `btmd_schema.py`), `src/` (16 modules), `scripts/` (00 to 04), `app/app.py`, `notebooks/01_preprocessing.ipynb`, `notebooks/02_model_training.ipynb`, `data_templates/` (three CSV templates), `tests/test_pure_logic.py`, and (new in this task) `src/image_presence.py`, `tests/test_image_presence.py`, `docs/`.

### 3.2 Missing files
- `configs/config.yaml`: not present. The project uses `configs/config.py`.
- TASK_00_DECISION_REVIEW_v0.3.zip, TASK_00.4_DATASET_FEASIBILITY_AUDIT.zip, TASK_00.5_CRITICAL_DATASET_VERIFICATION.zip: not present in the sandbox. Their contents could not be inspected, and no decision from them is assumed here.
- Dataset files (`data/raw`, `data/processed`): empty in the sandbox (`.gitkeep` only). The data is on the user's Drive.
- Git history: the sandbox copy is not a Git repository. The Windows repository could not be reached.

### 3.3 Test results
| Command | Expected | Actual | Result |
|---|---|---|---|
| `python3 -m pytest -q tests` (start of audit) | all pass | 15 passed | PASS |
| `python3 -m pytest -q tests` (after T-VAL-01) | all pass | 24 passed | PASS |
| `python3 -m py_compile` on changed files | no syntax errors | OK | PASS |
| `import torch` locally | torch available for scripts 02 and 04 | `ModuleNotFoundError` | FAIL (environment); end-to-end run of scripts 02 and 04 is UNVERIFIED locally |
| `summarize()` from script 04 with fake runs | correct means and subset fields | matches expected values | PASS (AST-extracted function) |

### 3.4 The missing-image shortcut (validity issue)
- `src/image_encoder.py::encode_images` fills rows without an image with zeros.
- `image_text_similarity` returns exactly 0 for those rows.
- After standardisation, the zero rows become one fixed vector per dimension, so a classifier can detect presence.
- Presence is correlated with the label [COLAB]: 62.9 percent of Authentic, 55.2 percent of Misleading and 90.7 percent of Satire posts have an image.
- Consequence: E2 and E3 may score partly on presence. This is a mechanism-level finding [CODE]. Its size is UNVERIFIED until the E0 control and image-subset results are produced on Colab.

### 3.5 Code quality issues
| Issue | Location | Severity | Status |
|---|---|---|---|
| Missing-image zeros are not explicit | `image_encoder.py`, `02` | Critical (validity) | Addressed by T-VAL-01 |
| E1 predictions not saved, so no per-sample analysis | `03`, `text_encoder.py` | High | Addressed in code (per-seed predictions); not yet run on Colab |
| E1 results file written only after all seeds, so a disconnect loses all seeds | `03` | High | Addressed in code (per-seed metrics, resume from checkpoint); not yet run on Colab |
| Feature script raises when evidence exists but `source_url` is missing | `02` | High (blocks E5, E6) | Open; needs a flag and a source policy |
| Notebook text said E1 takes 20 to 40 minutes | notebook markdown | Low | Measured CPU run took about 2 hours 20 minutes per seed; GPU run about 100 seconds of training per seed. The markdown in the user's notebook is wrong and should be edited. |
| Cell that drew the example table failed on a pandas `groupby` change | notebook cell | Low | Fixed in the reply (no `include_groups` dependency) |
| README says cultural labels must not be guessed, but the plan uses an LLM | `README.md` section 8 | High (method) | Open; owner decision |
| The graph cell parses `summary.md` by column position | notebook graph cell | Low | `summary.md` table format left unchanged so the parser keeps working |

---

## 4. Dataset Audit

### 4.1 Status of each dataset stage
- **Planned:** BTMD (Bangla multimodal fake-news dataset, gated on Hugging Face).
- **Downloaded:** yes, by the user, on the user's machine and Drive [COLAB].
- **Verified:** partly. Counts and drop reasons reconcile [COLAB]. Source and annotation process are UNVERIFIED.
- **Preprocessed:** yes [COLAB].
- **Experiment-ready:** yes for E1, E1f, E2 and E3. Not for E4, E5 and E6.

### 4.2 Verified statistics (from the user's Colab output)
| Quantity | Value | Evidence |
|---|---|---|
| Raw rows | 6,058 | `data_stats.json` |
| Excluded: Clickbait | 576 | `data_stats.json` |
| Dropped: empty text | 1,566 | `data_stats.json` |
| Dropped: duplicate text and label | 1 | `data_stats.json` |
| Dropped: duplicate image | 5 | `data_stats.json` |
| Final rows | 3,910 | `data_stats.json` (reconciles: 6,058 − 576 − 1,566 − 1 − 5) |
| Rows with an image | 2,397 (61.3%) | `data_stats.json` and dataset card |
| Authentic / Misleading / Satire | 2,208 / 1,509 / 193 | `data_stats.json` |
| Median words: Authentic / Misleading / Satire | 167 / 16 / 10 | dataset card |
| Image share: Authentic / Misleading / Satire | 62.9% / 55.2% / 90.7% | dataset card |
| Split train / val / test | 2,736 / 586 / 588 | split table (70/15/15 by row count) |
| Satire in test | 32 | split table |
| Groups | 3,906; median size 1 | dataset card and group check |
| Groups spanning more than one split | 0 | group check |
| Missing image files referenced | 0 | `data_stats.json` |

Satire is 193 of 3,910 rows (4.9 percent).

### 4.3 Provenance and quality
- Source credibility: UNVERIFIED. The README states that Satire posts come mostly from one platform. That claim is not checked in this audit.
- Annotation process of the original dataset: UNVERIFIED.
- Label consistency and annotation quality: not audited. The README already lists label ambiguity (gossip labelled Misinformation, satire-like posts labelled Misleading).
- Duplicates: text and label duplicates and image duplicates were removed. Near-duplicates (same story, different wording) were not checked.
- Broken media: no missing image files were referenced in the build.
- Video: video files were excluded in the slim copy. Video availability in the original is UNVERIFIED.

### 4.4 Leakage and confound risks
| Risk | Evidence | Severity |
|---|---|---|
| Length confound | Median words 167 / 16 / 10 | Critical |
| Image presence confound | Share with image 62.9 / 55.2 / 90.7 percent | Critical |
| Near-per-post groups | 3,906 groups for 3,910 posts | High |
| Source confound | README claim, UNVERIFIED | High |
| Small Satire class | 193 total, 32 in test | Medium |
| Image resize | Working copy at 512 px; original sizes UNVERIFIED | Low (report note) |

### 4.5 Remaining dataset work
1. Evidence source (requires the decision in Section 5).
2. Cultural annotation or silver labels (requires the decision in Section 5).
3. Source URL or source category, if evidence is to be checked for leakage.
4. Near-duplicate and same-story check across splits.
5. Label audit on a random sample.

---

## 5. Research Audit

### 5.1 Confirmed research direction
From the master prompt: a Bangla multimodal fake-news system covering text understanding, image-text analysis, cultural context, evidence retrieval, evidence-grounded verification and explainability, with reliable evaluation. The implemented baseline (E1 to E3) matches the first two parts.

### 5.2 Unresolved research decisions (owner must decide)
1. **Evidence source.** The research direction says "external fact-checking sources". The implementation plan uses Bangla Wikipedia and excludes fact-checking sites because BTMD labels probably come from fact-checkers. This is a design change. It is not made here.
2. **Label policy for cultural features.** `README.md` says labels must not be guessed. The plan uses LLM silver labels. One of the two must change, and the report must state the choice.
3. **Explainability evaluation.** Not defined. The prompt asks for explainability; no evaluation protocol exists.
4. **Generalisation evaluation.** Not defined. No second dataset or source-held-out split exists.

### 5.3 Methodological risks
- The image-presence shortcut and the length confound can each produce the reported gains (Sections 3.4 and 4.4).
- Three seeds give only rough variance. Report confidence intervals or more seeds for the final comparison.
- Macro-F1 with 32 Satire test posts is noisy. Report per-class CIs.
- No ablation yet isolates image content from presence.

### 5.4 Evaluation requirements (not yet met)
- Per-class precision, recall, F1 and confusion matrices for every reported model.
- Length-only baseline result reported next to every text and fusion result.
- Image-present subset metrics for every model that uses image features.
- E0 presence-only control.
- Retrieval: recall@k on claim-evidence pairs. NLI: accuracy on labelled Bangla pairs.
- Calibration (ECE) and selective accuracy for E6.

---

## 6. Architecture Readiness

| Component | Research purpose | Input | Output | Status | Dependencies | Verification requirement |
|---|---|---|---|---|---|---|
| Text encoder (BanglaBERT, frozen) | Bangla text representation | cleaned text | 768-d vector | COMPLETED (Colab) | `csebuetnlp/banglabert` | Shape check; model access |
| Text classifier (BanglaBERT fine-tuned, E1) | text-only baseline | text, labels | class probabilities | COMPLETED (Colab, 3 seeds) | GPU | Per-seed predictions; resume test |
| Image encoder (CLIP ViT-B/32) | image representation | image | 512-d | COMPLETED (Colab) | CLIP image model | Zero-row check (T-VAL-01) |
| Text encoder for CLIP (multilingual) | Bangla text in CLIP space | text | 512-d | COMPLETED (Colab) | multilingual CLIP text model | Shape check |
| Image-text similarity | consistency between image and text | two 512-d vectors | 1-d | COMPLETED (Colab) | above | Presence handling (T-VAL-01) |
| Presence flag | control for the shortcut | metadata | 1-d | IMPLEMENTED BY T-VAL-01 | metadata | Consistency with zero rows |
| Cultural module (E4) | sarcasm, humour, local and historical references | frozen text embedding, annotations | 4 scores | NOT STARTED | annotations or labels | Agreement check; train-only fit |
| Evidence retrieval | find supporting or refuting text | claim, evidence base | top-k evidence | IMPLEMENTED BY CODE, NOT RUN | evidence file | recall@k on pairs |
| NLI verification | judge claim against evidence | claim, evidence | entail, contradict, neutral | IMPLEMENTED BY CODE, NOT RUN | NLI model; pairs | Bangla accuracy on pairs |
| Fusion MLP (CCEF) | combine features | feature blocks | 3-class probabilities | COMPLETED (Colab) | feature blocks | Subset and CM reporting |
| Decision layer (E6) | four-class verdict with Insufficient | fusion probs, evidence verdict | Authentic, Misleading, Satire, Insufficient | IMPLEMENTED BY CODE, NOT RUN | E5 | Coverage and selective accuracy on test |
| Explanation builder | evidence-grounded text | verdict, evidence | markdown | IMPLEMENTED BY CODE, NOT RUN | verdict | Human check on examples |
| Gradio app | demonstration | any post | verdict | IMPLEMENTED BY CODE, NOT RUN | artifacts | Run in Colab |

Architecture design and implementation are distinguished: every row above with "CODE" or "NOT RUN" is design plus code, not verified behaviour.

---

## 7. Risk Register

| Rank | Issue | Consequence | Mitigation |
|---|---|---|---|
| Critical | Missing-image shortcut in E2 and E3 | Image gains may be presence, not content | T-VAL-01: E0, image-subset metrics, zero-row check |
| Critical | Length confound (167 / 16 / 10 words) | Gains may be length | Length-only baseline reported beside every model |
| Critical | Research direction: evidence source changed without decision | Report may contradict the approved direction | Owner decision before E5 |
| Critical | Label policy conflict (README vs silver labels) | E4 may be ethically or methodologically unsupported | Owner decision; state the label source in the report |
| High | Source URL guard blocks E5 and E6 | Evidence cannot be tested | Flag plus source policy |
| High | Group split is nearly per-post | Same-story leakage possible across splits | Story-level grouping |
| High | E1 numbers have no saved predictions | No confusion matrix or error analysis for E1 | Per-seed predictions (implemented, to be run) |
| Medium | Satire 32 test posts | Wide uncertainty | Confidence intervals; more seeds |
| Medium | Truncation at 128 tokens | Long posts lose content | Ablation with longer context |
| Medium | NLI not checked on Bangla | Evidence verdicts may be wrong | Pairs check |
| Low | Notebook timing text wrong | Misleading to readers | Edit markdown |
| Low | Image resized to 512 px | Report must state it | Note in report |

---

## Appendix: Items that could not be inspected

- TASK_00 zip artifacts: not available in the sandbox.
- `configs/config.yaml`: not present in the project.
- Windows repository `D:\Capstone\Bangla-Multimodal-FakeNews`: not reachable; Git history, branch and uncommitted changes are UNVERIFIED.
- Dataset files and Colab outputs: on the user's Drive; the numbers above come from pasted Colab output.
