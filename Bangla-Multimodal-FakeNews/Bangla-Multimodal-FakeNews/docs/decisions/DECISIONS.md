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
