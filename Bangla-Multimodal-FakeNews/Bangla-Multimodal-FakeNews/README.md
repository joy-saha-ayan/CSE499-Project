# Bangla Multimodal Fake News Detection

Multimodal fake-news detection for Bangla meme and social-media posts. Components: BanglaBERT (text), CLIP (image and image-text consistency), cultural context features, BM25 + dense evidence retrieval, NLI-based evidence verification, a CCEF fusion MLP, and a four-class decision layer.

The dataset (BTMD) is **not** included in this zip. You must keep your own copy in Google Drive.

## 1. Project structure

```
configs/          config.py (paths, models, hyperparameters), btmd_schema.py (BTMD column mapping)
src/              data_prep, splitting, text/image encoders, cultural, evidence, fusion, decision, metrics
scripts/          00 preflight -> 01 data -> 02 features -> 03 E1 fine-tune -> 04 fusion + E6 + evidence check
                  05 cultural silver labels (LLM) + blind human check, 06 Wikipedia evidence build (LLM + MediaWiki)
                  07 frozen backbone features, 08 model-selection grid (35 configurations x 3 seeds)
notebooks/        00_supervised_run_guide.ipynb (START HERE: phase-by-phase run with stop points)
                  01_preprocessing.ipynb (data preparation, CPU)
                  02_model_training.ipynb (feature extraction and training, T4 GPU)
                  03_model_selection.ipynb (selection grid and winners, T4 GPU)
app/app.py        Gradio demo
tests/            pure-logic tests (no GPU, API key or network needed): pytest -q tests/
docs/decisions/   DECISIONS.md: owner decisions D-1 (evidence source), D-2 (cultural labels), D-3 (selection protocol)
data_templates/   templates for the evidence, cultural annotation and claim-evidence pair files
```

## 2. Google Drive layout

The code zip and the data are kept separately:

```
MyDrive/
├── Bangla-Multimodal-FakeNews.zip          <- this project
└── BFND-data/
    ├── data/
    │   ├── raw/                            <- slim BTMD (table file + images)
    │   ├── processed/                      <- created by step 1
    │   ├── evidence/evidence.csv           <- (needed for E5/E6) see data_templates
    │   └── annotations/
    │       ├── cultural_annotations.csv    <- (needed for E4+) manual annotation
    │       └── claim_evidence_pairs.csv    <- (optional) for measuring NLI accuracy
    ├── features/                           <- created automatically
    ├── checkpoints/                        <- created automatically
    └── results/                            <- summary.md and JSON files
```

A Colab session wipes `/content` when it ends, so data, checkpoints and results are stored on Drive.

## 3. Preparing the slim BTMD copy (local PC)

1. Remove the video files. The pipeline uses images and text only.
2. Keep the images, and the rows that have text and a label.
3. Put the table file (CSV, JSON or parquet) and the image folder in one folder, for example `BTMD_slim/`.
4. Upload the contents of that folder to `MyDrive/BFND-data/data/raw/`.

BTMD is a gated dataset. Use it under its terms. Do not publish the raw data or the slim copy.

## 4. Schema

`configs/btmd_schema.py` maps the BTMD columns to the pipeline. It is already filled in for the metadata.parquet table:

- `label_col` is `multi_category`, not `label`. The `label` column only holds Fake/Real, and Satire and Clickbait are inside Fake.
- `text_col` is `["heading", "text"]`. The two columns are joined.
- `image_col` is `image_path`, and `image_root` is empty because the values already start with `images/`.
- `group_col` is `heading`. When the heading is empty, a text hash is used.
- `source_url_col` is `None`, because BTMD has no source URL column.

If you change the table, re-run the inspection cell and update this file. `01_prepare_data.py` stops with an error if a required field is missing.

## 5. How to run

Open the notebooks in Colab, in this order:

1. `notebooks/01_preprocessing.ipynb` (CPU is enough). It unpacks the code, checks the version, runs the preflight, builds the unified table, makes the group split and runs the sanity checks.
2. `notebooks/02_model_training.ipynb` (T4 GPU). It unpacks the code again, checks the GPU, extracts features, trains E1 and the fusion models, and prints the summary.

Terminal equivalent:

```bash
python scripts/00_preflight.py            # model IDs, schema, data files, GPU; stop if FAIL
python scripts/01_prepare_data.py         # unified table, group split, leakage check
python scripts/02_extract_features.py     # BanglaBERT, CLIP, cultural, evidence -> features/
python scripts/03_train_text_baseline.py  # E1: fine-tuned BanglaBERT, 3 seeds
python scripts/04_train_fusion.py         # E1f, E2-E5, E6, evidence verification -> results/
python app/app.py                         # demo
```

After a runtime restart, `/content` is empty. Run the unpack and install cells again. Data on Drive is kept.

## 6. Experiments

| ID | What | Where |
|---|---|---|
| E1 | BanglaBERT fine-tuned, text only | `03_train_text_baseline.py` |
| E0 | Image presence flag only (control for the missing-image shortcut) | `04_train_fusion.py` |
| E1f | Frozen BanglaBERT + MLP (fair comparison) | `04_train_fusion.py` |
| E2 | CLIP image only | `04_train_fusion.py` |
| E3 | BanglaBERT + CLIP image + CLIP text + similarity | `04_train_fusion.py` |
| E4 | E3 + cultural | `04_train_fusion.py` |
| E5 | E4 + evidence (NLI features) | `04_train_fusion.py` |
| E6 | E5 + four-class decision layer | `04_train_fusion.py` |

Every experiment uses the same test set and three seeds (42, 43, 44). Results are reported as mean ± std.

## 7. Design decisions

- **Group split**: the same claim or event never appears in more than one split. With no usable group column, a text hash is used, which is weaker. Paraphrases of the same event can still cross splits.
- **Frozen BanglaBERT for fusion**: the training-set embeddings of a fine-tuned model are over-fitted, so the fusion model receives pretrained embeddings instead.
- **Cultural classifier fitted on the train split only**: train rows get out-of-fold predictions.
- **Evidence source (D-1)**: Bangla Wikipedia lead sections. An LLM writes a search query from the post text only; no class label is sent at any step. See `docs/decisions/DECISIONS.md`.
- **Cultural labels (D-2)**: LLM silver labels from post text only, used when no manual file exists. A blind 40-row human check measures agreement (Cohen's kappa) before the silver labels are trusted.
- **Evidence leakage guard**: if the evidence database contains the fact-check article the label came from, retrieval simply returns the label. `drop_label_source_evidence` removes those URLs. BTMD has no source URL, so the guard does not apply. This is recorded in `retrieval_report.json` (`label_source_guard`). It is safe only because the Wikipedia query never sees the label.
- **"Insufficient" is not a dataset label**: it comes from the decision layer, so its accuracy cannot be measured. The report shows coverage and selective accuracy instead.
- **Clickbait is excluded** from the main 3-class task. It can be run as a separate experiment later.
- **BanFakeMM is not used for the main result**: its fake part was generated with LLMs and image generators, so it is adversarial data. Keep it as a separate robustness test.
- **CLIP uses two models**: the original `clip-ViT-B-32` for images, and the multilingual variant for Bangla text. The multilingual model is text-only.

## 8. Known limitations (to be written up in the report)

1. **Length confound**: in the current data, Authentic texts are much longer than Misleading and Satire texts (median about 1136 characters versus 100 and 68). A model can use length alone to get a good score. A length-only baseline is included in the training notebook for this reason.
2. **Small Satire class**: only 193 Satire rows have text, and about 32 of them are in the test split. Satire scores will be noisy, so report variance across seeds and confidence intervals. Most Satire posts are image-only memes, so OCR may recover more of them.
3. **Source confound**: almost all Satire posts come from Facebook. `source_category` must not be used as a model input.
4. **Label ambiguity**: some Misinformation posts are gossip or engagement bait, and some Satire-like posts are labelled Misleading. Label noise therefore affects the results and should be discussed.
5. **Truncation**: BanglaBERT uses `MAX_LEN = 128` tokens. For long articles, only the heading and the first lines are seen.
6. **Wikipedia is not a fact-check source**: it covers established facts, not most current-event claims. Coverage (share of posts with at least one article) is reported in `evidence_build_report.json`. Low coverage will limit E5 and E6 and must be stated in the report.
7. **NLI model not checked on Bangla**: `NLI_MODEL` is multilingual, but its Bangla performance must be measured with `claim_evidence_pairs.csv`. Evidence relevance (Wikipedia article for the post) is checked separately with `evidence_relevance_check.csv`.
8. **Cultural labels are LLM silver labels**: they are produced by an open model (`LOCAL_LLM_MODEL`, free, run on the GPU) from the post text, not by people. Its Bangla quality is unverified. They are valid only if the blind 40-row human check gives acceptable agreement (`results/cultural_agreement.json`). The results must say "silver labels" and report that agreement. A manual file overrides them when it exists.
9. **OCR**: `USE_OCR = False` by default. Enable it for an ablation.
10. **Missing-image shortcut**: a post without an image gets a zero CLIP vector and a similarity of exactly 0, and image presence is correlated with the label (Satire 90.7 percent with an image, Misleading 55.2 percent). Image models can therefore score partly on presence. `02_extract_features.py` verifies the zero rows against the metadata and saves `has_image.npy`. E0 measures the presence-only score, and `04_train_fusion.py` reports every experiment's macro-F1 on the image-present test rows (`image_subset` in `fusion_results.json`).
11. **Threshold**: `TARGET_COVERAGE` is chosen on validation only, never on test.
12. **Image resolution**: images were resized to at most 512 px in the slim copy. This has little effect on CLIP (which uses 224 px) but may hurt OCR on small text.

## 9. Models

The zip contains no model weights. Pretrained models are downloaded from Hugging Face the first time they are used. Training produces:

- Fine-tuned BanglaBERT (E1): `checkpoints/text_finetune_seed*/best/`
- Fusion MLP and its scaler (E5): `checkpoints/e5_artifacts.joblib`
- Cultural classifiers: `checkpoints/cultural_models.joblib`

| Component | Model | Licence or condition |
|---|---|---|
| Text | `csebuetnlp/banglabert` | cc-by-nc-sa-4.0 (non-commercial) |
| Image | `sentence-transformers/clip-ViT-B-32` | original CLIP image encoder |
| Bangla text (CLIP space) | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | text only; passing an image raises an error |
| Retrieval | `intfloat/multilingual-e5-base` | MIT |
| NLI | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | MIT |

The BanglaBERT model card says the pretraining used a specific normaliser (`csebuetnlp/normalizer`). If possible, apply the same normaliser to the text. This can be added later as an ablation.

## 10. What has been verified

- The pure-logic part (cleaning, label mapping, splitting, leakage check, decision rule, metrics, cultural fitting, NLI aggregation, data preparation) passes 15 pytest tests.
- Step 1 (data preparation) runs on the real BTMD table: 6,058 rows in, 3,910 rows out.
- Model training, NLI, CLIP and OCR have not yet been run to completion. The first run in Colab must be checked on its full output.
