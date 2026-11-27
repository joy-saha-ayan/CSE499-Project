# Research decisions log

Every change to a research decision is recorded here, with the date and who decided it.
A decision that is not recorded here is not a decision.

## D-1: Evidence source = Bangla Wikipedia (decided 2026-11-18 by the owner, Joy Saha)

- **Status:** accepted, implemented in `scripts/06_build_evidence_wiki.py` and `src/wiki_evidence.py`.
- **Replaces:** the open owner decision "evidence source: Wikipedia or fact-checking sources" (listed as pending in the progress audit and the execution report).
- **How evidence is built:** an LLM writes one Bangla search query from the post text. The MediaWiki search returns up to `WIKI_SEARCH_LIMIT` articles. The lead section of each article is split into passages of at most `WIKI_CHUNK_CHARS` characters.
- **Leakage rule:** the class label is never part of any prompt or query. The retrieval uses the post text only.
- **Label-source guard:** not applicable. BTMD has no source URL column, so `drop_label_source_evidence` cannot run. This is recorded as `label_source_guard` in `results/retrieval_report.json`. It is acceptable only because of the leakage rule above.
- **Known risks (accepted by the owner):**
  - Wikipedia covers established facts, not most current-event claims. Coverage is measured as `coverage_share` in `results/evidence_build_report.json`. If coverage is low, E5 and E6 will show small gains, and the report must say so.
  - Wikipedia is not an independent fact-checking source. "Refutes" verdicts may be rare.
- **Still required:** a 40-row relevance check (`annotations/evidence_relevance_check.csv`) and the NLI Bangla check (`claim_evidence_pairs.csv`). Results are not reported as validated until both are done.

## D-2: Cultural labels = LLM silver labels, validated by a blind human check (decided 2026-11-18 by the owner, Joy Saha)

- **Status:** accepted, implemented in `scripts/05_build_cultural_silver.py` and `src/llm_labeling.py`.
- **Replaces:** the README limitation "Labels must not be guessed" (old limitation 8). That limitation is now replaced by the rule below.
- **Backend (updated 2026-11-18, owner choice: free route):** `LLM_BACKEND = "local"`. The model is `LOCAL_LLM_MODEL` (default `Qwen/Qwen2.5-3B-Instruct`), run on the Colab GPU with greedy decoding. No API key and no cost. The Anthropic backend (`LLM_BACKEND = "anthropic"`, model `claude-haiku-5-5`) is kept as an option.
- **Bangla quality is unverified:** the base model's card does not document Bangla support. The 40-row check decides whether the labels are usable. If agreement is poor, the next option is a larger model (for example Qwen2.5-7B-Instruct in 4-bit), or the Anthropic backend.
- **Cache warning:** `CACHE_DIR` stores answers by sample_id, not by backend. Delete `cultural_cache.jsonl` and `query_cache.jsonl` before switching backend, or the two backends will mix.
- **How labels are made:** the LLM receives the post text, cut to `LLM_MAX_INPUT_CHARS`, and answers 4 binary features as strict JSON. The class label is never sent.
- **Which labels are used:** a manual file `annotations/cultural_annotations.csv` overrides the silver labels when it exists. Otherwise the silver file `annotations/cultural_silver_labels.csv` is used. `results/cultural_source.json` records which one was used.
- **Validation:** a blind 40-row sheet (`annotations/cultural_human_check.csv`, text only, no LLM answer, no class label) is labelled by a human. Agreement (agreement rate and Cohen's kappa per feature) is written to `results/cultural_agreement.json`.
- **Reporting rule:** every result that uses cultural features must say "LLM silver labels" and give the agreement numbers. Kappa from 40 rows is noisy and must be reported with that caveat.
- **Known risks:** the LLM may share biases with the annotators. The features measure what the LLM believes, not ground truth.

### D-2 update (2026-11-21): human check result and decision

- **Human check:** 40 blind rows, labelled by the owner (one labeller). Output of `scripts/05_build_cultural_silver.py` after the sheet was filled:

| Feature | n | Agreement | Kappa | Verdict (pre-set rule) | Human positive rate | LLM positive rate |
|---|---|---|---|---|---|---|
| sarcasm | 40 | 0.75 | 0.219 | FAIL | 30.0% | 5.0% |
| humor | 40 | 0.80 | 0.162 | FAIL | 22.5% | 2.5% |
| local_reference | 40 | 0.65 | 0.314 | FAIL | 55.0% | 40.0% |
| historical_reference | 40 | 0.95 | 0.474 | REVIEW | 5.0% (2 rows) | 5.0% |

- **Reading:** high agreement is mostly from the negative class. Kappa is the number to use. The LLM under-detects sarcasm and humor by about 6 to 10 times. The historical_reference result rests on 2 positive rows, so it cannot support a decision.
- **Decision:** `USE_CULTURAL_FEATURES = False` in `configs/config.py`. Cultural features are not used in E4, and E5 is E3 + evidence. Script 02 does not build `cultural.npy` while the flag is off. The decision was taken by the assistant on the owner's instruction ("best option koro"). The owner can override it by setting the flag to True.
- **Effect on earlier results:** the earlier E4 result (macro-F1 0.730, below E3 at 0.745) is not reported as evidence for cultural features. Any old `cultural.npy` file on Drive is ignored.
- **Re-open rule (proposed):** cultural features return only after new labels pass a fresh blind check with kappa of at least 0.6 on every feature used. The sample must be large enough for a stable kappa (proposed at least 100 rows, stratified so each feature has enough positives). A new decision entry is then required.

## D-3: Model-selection protocol (decided 2026-11-18, owner choice: full grid, winners chosen on validation)

- **Status:** accepted, implemented in `scripts/07_extract_backbones.py`, `scripts/08_model_selection.py`, `src/selection.py`, `src/fusion_heads.py`, and `notebooks/03_model_selection.ipynb`.
- **Grid:** text (BanglaBERT, mBERT, XLM-R-base) x image (CLIP ViT-B/32, ViT-B/16, Swin-T) x fusion (early, intermediate, late) = 27 combinations, plus 3 text-only, 3 image-only and 2 controls (length only, has_image only). 35 configurations x 3 seeds = 105 runs.
- **Frozen backbones:** every backbone gives frozen embeddings, and the same head is trained on them. The BanglaBERT fine-tune (E1) scored lower macro-F1 than the frozen version (E1f: 0.627 vs E1: 0.546), so frozen embeddings were kept for selection. End-to-end fine-tuning of the final winner is a later step.
- **Fusion definitions:** early = concatenation, then one MLP (the CCEF head). Intermediate = per-modality projections, then a joint layer over [h_text, h_image, h_text * h_image]. Late = one MLP per modality, probabilities averaged with equal weight (no tuned weight).
- **Winner rule:** highest mean validation macro-F1 over the 3 seeds, separately for text, image and fusion. Ties go to the smaller std. Accuracy and recall are reported but never pick the winner. The test set is reported and never used to choose.
- **Known differences from E3:** the grid fusion uses text and image vectors only. E3 also had the CLIP text vector and the image-text similarity, so E3 is not directly comparable with the grid numbers.
- **Warning:** the image-only winner and the fusion winner can differ from the best text model. The paper that motivated this grid (BanClick, Table 11) showed one pairing collapsing under intermediate fusion, so the seed spread must be read with the mean.

## D-4: Evidence relevance gate (threshold 0.6, FAIL)

- **Rule:** the share of relevant passages on the 40-row blind sheet (`annotations/evidence_relevance_check.csv`, column `relevant_0_or_1`) must be at least `EVIDENCE_RELEVANCE_MIN = 0.6` for the gate to PASS. The threshold is in `configs/config.py`. `scripts/09_evidence_gate.py` writes `results/evidence_gate.json` and refuses a file whose recorded threshold differs.
- **Result:** 13 of 40 rows relevant, share 0.325, status **FAIL**. Evidence is therefore not used in the final model.
- **Where the numbers come from:** the result was recorded in the project session. Re-run `scripts/09_evidence_gate.py` on the filled sheet to regenerate `results/evidence_gate.json` and confirm it.
- **Open item for the owner:** confirm the date on which the 0.6 threshold was first set. This entry does not invent that date.
- **Rule for the future:** a threshold must not be lowered after a result is seen. A new gate needs a new entry, a new labelled sheet, and a threshold written down before the sheet is labelled.

## D-5: Evidence in the app is display-only (decided 2026-11-22 by the assistant, on the owner's instruction to add evidence properly)

- **Status:** accepted, implemented in `src/evidence_app.py`, `scripts/09_evidence_gate.py`, and the Evidence tab of `app/capstone_app.py`. The owner can override it.
- **What the app does:** the Evidence tab takes the typed post text, searches the Bangla Wikipedia passages in `data/evidence/evidence.csv` with BM25, and shows the top `EVIDENCE_TOP_K` passages with their scores and URLs. It shows the gate status from D-4.
- **What the app does not do:** evidence does not change the prediction. The verdict stays the fused text + image model. A passing gate alone does not change this; using evidence in a prediction needs a validation gain in E5 and a new entry.
- **Retriever:** BM25 with the Lucene idf, written in pure Python so that it runs without torch. The dense and hybrid retrievers in `src/evidence_retrieval.py` are not used yet. Choosing between BM25, dense and hybrid must be done on a labelled dev set, never on the test set.
- **Leakage rule:** the query is the post text only. No class label enters the query or the passage choice. The guard `FACT_CHECK_DOMAINS` drops any passage whose URL host, or a subdomain of it, is on the list. The list is empty for D-1 (Wikipedia is not a fact-checker). It must be filled if a fact-check corpus is ever used as evidence, because BTMD labels probably come from fact-checkers.
- **Status of evidence:** implemented, not validated. The gate is FAIL (D-4), so the Evidence tab is a view of the retrieval, not a verified claim.
- **Needs an owner decision before the next evidence experiment:** (a) the evidence source for a label-safe experiment, for example the fact-check corpus of the 499B project, kept separate from BTMD; (b) which retriever to use, chosen on a dev set.

## D-6: Confound controls for image presence and text length (decided 2026-11-25 by the assistant; diagnostic only)

- **Status:** proposed by the assistant, not yet confirmed by the owner. Implemented in `src/confound_controls.py` and `scripts/10_confound_controls.py`; unit-tested on synthetic data; the script was smoke-tested with a stand-in head (numbers from that run are meaningless).
- **What it adds:** (1) `fusion_image_shuffled`: the image vectors are permuted inside each split among rows that have an image, so presence is kept and the image-post match is broken; (2) `text_plus_presence`: text plus the presence flag only; (3) per-length-quartile scores on validation rows; (4) a cross-split near-duplicate audit (TF-IDF character n-grams, threshold 0.8, reference = train).
- **What it does not change:** selection (D-3) stays where it was. Validation numbers are used to read the controls. Test numbers are written to the output but not used for any choice.
- **How to read it:** image content matters beyond presence only if `fusion_real` beats `text_plus_presence` on validation and `fusion_image_shuffled` does not. Otherwise the image gain is mostly presence.
- **Owner decision needed:** confirm that this control is in scope for the report, and that the threshold 0.8 for near-duplicates is accepted or changed before the run (the change must be written here before the run).

## D-7: New evidence relevance check with a frozen dev / eval split (decided 2026-11-25 by the assistant; owner to confirm)

- **Why:** D-4 used the 40-row sheet to decide the gate. Any retriever change must now be chosen on a dev set, and the gate read once on an eval set.
- **What:** `scripts/11_evidence_eval_sheet.py` samples 40 dev posts and 40 eval posts from the validation split only (seed 42, fixed in `evidence_eval_roles.json`), retrieves the top 3 Wikipedia passages per post with BM25, and writes `evidence_eval_sheet.csv` with an empty `relevant_0_or_1` column. The script refuses to overwrite an existing sheet.
- **Rules:** the query is the post text only. No label is used to pick passages. The test split is never read. A human labels every row before any gate is computed.
- **Order of work:** (1) label the dev rows and find why relevance fails (failure categories written down); (2) choose the retriever on dev only (BM25, dense, or hybrid); (3) freeze the retriever; (4) label the eval rows and run the gate with a threshold written here before labelling; (5) only then report.
- **Threshold:** `EVIDENCE_RELEVANCE_MIN` stays at 0.6 from D-4 unless the owner writes a new value here before the eval labels are seen.
- **Owner decision needed:** confirm the sheet size (40 + 40 posts, 3 passages each = 240 rows) and who labels it.

### D-6 status update (2026-11-25): threshold decided
- The near-duplicate threshold stays at 0.8 (character n-gram TF-IDF cosine, reference = train). Decided by the assistant on the owner's instruction, before any run. The run has not happened yet in this environment.

### D-7 status update (2026-11-25): labelling is human-only, not yet done
- The 240-row sheet (`scripts/11_evidence_eval_sheet.py`) must be labelled by a person. The assistant does not label gate rows, because a language model's labels are not the human check that D-4 and the README require. The eval gate stays NOT RUN until human eval labels exist.
- Dev labels may be used to choose a retriever. Eval labels are read once.

## D-8: Paired bootstrap CIs on the test split (decided 2026-11-25 by the assistant; owner to confirm)

- **Why:** the test split has 588 rows and 32 Satire rows. A single macro-F1 hides the uncertainty, and the fusion-vs-text comparison needs a paired test.
- **What:** `src/bootstrap.py` and `scripts/12_bootstrap_ci.py` resample the test rows, score both saved models on the same resample, and report the difference, its 95% CI, and P(B > A). Macro-F1 and Satire one-vs-rest F1 are both reported.
- **Rules:** read-only on saved test probabilities (`results/selection/probs/`), seeds averaged first, nothing selected from it. Reported once in the final table.

## D-9: Cultural features closed as a negative result (decided 2026-11-25 by the owner's instruction, carried out by the assistant)

- **Decision:** path B. There is no time for a new human-labelled cultural sample. Cultural features stay off (`USE_CULTURAL_FEATURES = False`).
- **What is reported:** the blind human check failed (kappa sarcasm 0.219, humour 0.162, local reference 0.314). The LLM silver labels are not used to train, select or report any result as validated.
- **What is not claimed:** that cultural context helps or hurts the classifier. It was not tested on validated labels.
- **Reopen only if:** a new human-labelled sample with a written rubric exists and two annotators reach kappa ≥ 0.6 on it. A new entry is needed before any run.

## D-10: Test-set contrasts for image content (decided 2026-11-26 by the assistant; owner to confirm)

- **Why:** the validation contrast (fusion_real − text_plus_presence = +0.039) says image content adds more than presence. Its test-set version is needed before the report claims it.
- **What is read on test:** three paired bootstrap contrasts, chosen before the run: (1) fusion_real − text_plus_presence (image content beyond presence), (2) fusion_real − fusion_image_shuffled (real pairing beats broken pairing), (3) text_plus_presence − text_banglabert (presence alone on text). Macro-F1 and Satire F1 for each.
- **Test-read count:** this is a second read of the test split (the first was the text vs fusion bootstrap). The three contrasts above are fixed here. Nothing is added or dropped after seeing the numbers.
- **Rules:** no model, threshold or feature is chosen from these numbers. If a contrast is weaker than expected, the report says so.

## D-11: Dev labels for the evidence sheet (recorded 2026-11-26)

- **What was done:** 120 dev rows of `evidence_eval_sheet.csv` were labelled by hand (1 = 29, 0 = 91). Eval rows are blank and were not touched. An audit pass corrected 3 labels (two 1→0 for the theft post at ranks 1 and 2, one 0→1 for the cabinet post at rank 2). Post-audit labels: 1 = 29, 0 = 91.
- **Format:** the saved CSV stores the labels as "0.0" and "1.0" (float conversion). The analysis uses a working copy with integer 0/1. The labelled file itself is not edited.
- **Protocol timing:** the label rule (D-7, audit revision in `annotation_methodology_fake_news_detection_D7.md`) was written after the labels were made. It is not a pre-registered protocol. The report must say this.
- **Not recorded:** the number of uncertain cases during the first labelling pass. The labeller's name is not in `labeller_name.txt` yet ("Not provided").
- **Dev analysis (post-clustered bootstrap, 2000 resamples, seed 42):**
  - Share of relevant passages: 0.242 (95% CI 0.158–0.333).
  - Precision at rank 1: 0.325 (CI 0.175–0.475).
  - At least one relevant passage in the top 3: 0.500 (CI 0.350–0.675). 20 of 40 posts have none.
  - Relevant by rank: rank 1 = 13, rank 2 = 11, rank 3 = 5.
- **Gate:** the threshold stays at 0.6 (D-4). The dev share is far below it. The gate result remains FAIL. The eval gate must not be read until the retriever is frozen.
- **Retriever choice (planned):** the hybrid retriever in `src/evidence_retrieval.py` (BM25 + dense, alpha = 0.5, the existing default; alpha must not be tuned on dev labels) will be compared with BM25 on dev only. If the hybrid top-3 adds passages not in the labelled set, those new passages need dev labels before the comparison counts.
- **Rules unchanged:** eval rows are read once, after the retriever is frozen. No label is used to pick passages for the eval sheet.

## D-11 addendum (2026-11-27): overlap check result and a bug fix

- **First run (Colab, 2026-11-26):** hybrid pairs 120; "already labelled" 50; unlabelled 70 (39 posts). The labelled check matched passage text only, not (post, passage) pairs. A passage labelled for one post could count as labelled for another. The first run's 50 / 70 split is therefore **not reliable**. The 70 passages that were not in the sheet at all are certainly unlabelled. The true count of unlabelled pairs is at least 70, and it may be higher.
- **Fix:** `scripts/14_dev_retriever_overlap.py` now matches (post_id, passage). Its output has the same columns as the labelled sheet (`pair_id, role, post_id, post_text, rank, score, passage, source, url, relevant_0_or_1`) plus `in_bm25_top3`. The label column is blank.
- **Rule for labellers:** label each row using `post_text` and `passage` only. Hide `rank`, `score`, `in_bm25_top3`, `source` and `url` during labelling (D-7).
- **Decision:** per the rule above, the retriever comparison counts only after every unlabelled dev pair has a label. The new pairs will be labelled under the same D-7 rule.
- **Eval rows:** unchanged. Still blank until the retriever is frozen.

## D-12: Evidence retriever frozen on dev (recorded 2026-11-27)

- **What was run:** `scripts/15_dev_retriever_compare.py` on Colab (GPU). Dev rows only: 40 posts. BM25 top-3 = 120 pairs. Hybrid top-3 (alpha 0.5, not tuned) = 120 pairs: 49 already labelled, 71 labelled now under the D-7 rule. The 71 labels were given without rank, score or retriever information.
- **Result (post-clustered bootstrap, 2000 resamples, seed 42):**

| Metric | BM25 | Hybrid | Hybrid minus BM25 (95% CI) |
|---|---|---|---|
| Share of relevant passages | 0.242 (0.158–0.325) | 0.225 (0.142–0.317) | −0.017 (−0.067 to 0.042) |
| Precision at rank 1 | 0.325 (0.200–0.475) | 0.350 (0.200–0.500) | +0.025 (−0.100 to 0.150) |
| At least one relevant in top 3 | 0.500 (0.350–0.650) | 0.475 (0.325–0.625) | −0.025 (−0.100 to 0.050) |

- **Relevant by rank:** BM25 13 / 11 / 5; hybrid 14 / 7 / 6.
- **Decision:** BM25 (`src/evidence_app.py`) is the evidence retriever for the rest of the project. Reasons: the hybrid shows no measured gain on any dev metric, every paired interval includes 0, and BM25 runs in pure Python with no model download, so it is easier to reproduce. The hybrid is not used.
- **Limit of this decision:** the dev set has 40 posts. Interval half-width is about ±0.05 on the share, so differences smaller than that cannot be detected. The decision was taken after seeing dev results. It is a dev-based choice, not a pre-registered test.
- **Interval method:** these intervals replace the dev analysis intervals in D-11. The point estimates are the same. The resampling method differs slightly. Use the numbers above in the report.
- **Gate:** threshold stays at 0.6 (D-4). Gate result remains FAIL (dev share 0.242, far below 0.6). Evidence stays display-only (D-5).
- **Eval rows:** still blank. BM25 top-3 pairs are the existing 120 eval rows, so no new eval pairs are needed. The eval gate is read once, after the 120 eval rows are labelled under D-7.
- **Owner to confirm:** the owner must confirm this decision.

## D-13: Eval gate read once (recorded 2026-11-27)

- **Input:** `evidence_eval_sheet.csv`, the 120 eval rows (40 posts x 3 BM25 passages), labelled by the owner under the D-7 rule. Dev rows unchanged (1 = 29, 0 = 91). Labelled file checked: 240 rows, eval labels all 0 or 1, no other column changed.
- **Retriever at the time of reading:** BM25, frozen by D-12.
- **Threshold:** 0.6 (D-4, unchanged). It was set before the eval labels were seen.
- **Result:** 49 of 120 eval passages relevant. Share 0.408. Status **FAIL**.
- **How it was computed:** `relevance_share` from `src/evidence_app.py`, the same function `scripts/09_evidence_gate.py` uses. Script 09 itself reads the old 40-row D-4 sheet, so it was not used on the eval sheet. No script or config was changed.
- **Consequence:** evidence stays display-only (D-5). It is not used in the prediction.
- **Rule:** the eval gate is now read. It must not be read again on this sheet. Any new retriever or new threshold needs a new entry and new labels.
- **Owner to confirm:** the owner must confirm this entry.

## D-14: Pre-registered fresh evidence test (written 2026-11-27, before any label on the new sheet)

- **Why:** D-13 read the eval gate once (share 0.408, FAIL). A new retrieval method needs new posts and new labels. The old eval rows cannot be reused.
- **Fresh sample:** 40 posts from the validation split with non-empty model_text. They must not be in the D-7 dev or eval roles (`evidence_eval_roles.json`). Sampled with `random.Random(43)` from the sorted post ids. The test split is never read.
- **Two query methods, both fixed now:**
  - **baseline:** the post text, as in D-7 (BM25, top 3).
  - **keyword (candidate):** tokens of the post text (`tokenize` from `src/evidence_app.py`, lower-cased), minus a fixed stopword list. Each token is scored by idf over the passage table, idf = log((N+1)/(df+1)). The top 8 unique tokens by idf, ties broken alphabetically, form the query. BM25 top 3 with that query.
- **Labelling:** the union of both methods' top-3 passages, deduplicated by (post_id, first 400 characters of passage). Labelled under the D-7 rule, blind to method, rank and score.
- **Gate metric:** share of relevant passages among the keyword method's top-3 pairs, on the fresh sample only. Threshold **0.6** (D-4). PASS if share >= 0.6.
- **Baseline share** is reported for comparison. It has no role in the gate.
- **Read once.** If the gate FAILS, evidence stays display-only (D-5). No other method is tried on this sample.
- **Owner to confirm** the sample size, both query rules and the threshold before labelling starts.

## D-15: Result of the pre-registered fresh test (D-14), read once (recorded 2026-11-27)

- **Sheet:** `evidence_fresh_sheet.csv`, 193 unique pairs from 40 fresh validation posts (not in the D-7 roles). Labelled by the owner under the D-7 rule. All 193 labels are 0 or 1. The method map (`fresh_method_map.csv`) was used only to select the gate set.
- **Gate set (candidate, keyword method):** 120 pairs. Relevant 27. Share **0.225**. Per rank: 9 / 12 / 6.
- **Baseline (post text), for reference only:** 120 pairs. Relevant 37. Share 0.308.
- **Threshold:** 0.6 (D-4). Status **FAIL**.
- **Consequence:** the keyword query does not improve on the baseline on this sample. Evidence stays display-only (D-5).
- **Rule:** the fresh sample is now read. No other query method is tried on it (D-14).
- **Owner to confirm:** the owner must confirm this entry.

## D-16: OCR reference set (recorded 2026-11-27)

- **Pick:** 20 validation images, random seed 42 from the unified split table. Reference text was typed by the owner, from the image, without the OCR output.
- **Excluded:** 6 images with no readable text in the image (empty reference). They are dropped because a CER needs a non-empty reference. The exclusion uses the image content only, not the OCR result.
- **Result set:** 14 images with human references (`reference_source = human`). This is fewer than the 20 that `scripts/13_ocr_cer.py` prefers. The script will warn about the small set.
- **Limit:** the excluded images are all no-text cases. The CER therefore describes OCR on text-bearing memes only.
- **Owner to confirm:** the owner must confirm this entry.

- **OCR result (2026-11-27, pilot):** 14 images, 1,725 reference characters. Pooled CER 0.5994; mean per-image CER 0.6016. Script 13 warned that 14 is below 20 images. Punctuation counts toward CER (normalisation covers only Unicode NFC and whitespace). This is a pilot, not an accuracy estimate.
