<div align="center">

<img src="docs/assets/499b_banner.png" alt="Bangla Multimodal Fake News Detection — BanglaBERT + CLIP early fusion" width="100%">

# Bangla Multimodal Fake News Detection

**CSE 499B Capstone · North South University · Department of CSE**

BanglaBERT + CLIP · Early Fusion · BTMD Dataset

<img src="https://img.shields.io/badge/Status-Research%20Prototype-1f9e89?style=for-the-badge" alt="Research prototype"> <img src="https://img.shields.io/badge/Approach-Multimodal-24436b?style=for-the-badge" alt="Multimodal approach"> <img src="https://img.shields.io/badge/Language-Bangla-5b6b82?style=for-the-badge" alt="Bangla language">

</div>

A research prototype for multimodal classification of Bangla social-media posts using text and associated images from the **Bengali Trimodal Misinformation Dataset (BTMD)**.

**Quick navigation:** [499B Results](#2-main-result) · [Architecture](#4-method-and-architecture) · [Experiments](#5-controls-and-ablation-studies) · [Reproduction](#12-setup-and-reproduction) · [Earlier 499A project](#cse-499a--earlier-project)

> **Important:** This project predicts a dataset label from the text and image of a post. It does not verify claims against authoritative sources and should not be used as an automated fact-checker.

---

## Project branches

This repository contains two capstone stages maintained separately:

| Branch | Capstone | Project | Scope |
|---|---|---|---|
| `main` | CSE 499A | English multimodal fake-news classification | Fakeddit; XLM-RoBERTa and ResNet50 with late fusion; Streamlit application |
| `cse499b-capstone` | CSE 499B | Bangla multimodal misinformation classification | BTMD; BanglaBERT and CLIP with early fusion, shortcut controls, and evidence/OCR studies |

The detailed documentation below focuses on **CSE 499B**, which is maintained on `cse499b-capstone`.

## CSE 499A — Earlier project

CSE 499A explored English multimodal fake-news classification on Fakeddit. The project combined an XLM-RoBERTa text branch and a ResNet50 image branch using late fusion, and included a Streamlit demonstration application.

The saved output in `CSE499A_Final.ipynb` reports the following 499A test accuracies: **XLM-RoBERTa: 84.83%**, **ResNet50: 78.70%**, and **late fusion: 78.91%**. In that saved run, late fusion performed **5.92 percentage points below** the XLM-RoBERTa text model; therefore, fusion did not outperform the best single modality. These are historical 499A notebook results, not results from CSE 499B. The 499A code and artifacts remain on `main`.

<div align="center">
<img src="docs/assets/499a_historical_comparison.png" alt="CSE 499A historical test accuracy comparison" width="88%">
</div>

> **Research takeaway:** The 499A experiment showed that the tested late-fusion setup did not improve over the strongest text-only model. CSE 499B studies a separate Bangla dataset and early-fusion setup; its results are not a direct continuation on the same benchmark.

---

# CSE 499B — Bangla BTMD pipeline

## 1. Research question

**Does the image associated with a Bangla social-media post provide useful classification information beyond the text alone and beyond simple shortcut signals such as post length or image presence?**

The project studies whether multimodal early fusion improves classification over text-only and controlled alternatives. It also documents shortcut signals and exploratory components that did not meet their predefined criteria.

## 2. Main result

On the held-out test split, the BanglaBERT + CLIP early-fusion model achieved **0.722 macro-F1**, compared with **0.618** for BanglaBERT text-only. The paired bootstrap interval for the difference was **+0.060 to +0.152**.

This is a measured result from the current experiment, not a final generalization claim. The test split was read twice, the dataset is the only evaluation dataset so far, and the satire class is small. See [Results](#7-results) and [Limitations](#10-limitations).

<div align="center">
<img src="docs/assets/499b_results.png" alt="CSE 499B validation and test macro-F1 results" width="100%">
</div>

## 3. Dataset and label definition

The project uses BTMD (Bengali Trimodal Misinformation Dataset). The dataset and image files are not included in this repository; users must obtain them separately and comply with the dataset's license and terms.

| Item | Description |
|---|---|
| Dataset | BTMD — Bengali Trimodal Misinformation Dataset |
| Label mapping | `Real` → `Authentic`; `Misinformation` → `Misleading`; `Satire` → `Satire` |
| Excluded class | Clickbait |
| Final dataset size | 3,910 posts |
| Authentic | 2,208 |
| Misleading | 1,509 |
| Satire | 193 |
| Train split | 2,736 |
| Validation split | 586 |
| Test split | 588 |
| Split strategy | Grouped by heading to reduce heading-level leakage |

The validation split contains 30 satire posts and the test split contains 32. This limited support makes satire metrics less stable than the overall metrics.

## 4. Method and architecture

### 4.1 Modalities and encoders

- **Text encoder:** BanglaBERT
- **Image encoder:** CLIP
- **Fusion:** CCEF early-fusion classification head
- **Feature preparation:** Features are standardized using training-set statistics only, then concatenated and passed to an MLP classifier.
- **Encoder status:** The encoders are frozen in the reported experiment; fine-tuning was not evaluated under the same protocol.

### 4.2 Pipeline overview

```text
BTMD posts + associated images
             │
             ▼
     Dataset preparation
  label mapping / exclusions
 grouped heading-based split
             │
       ┌─────┴─────┐
       ▼           ▼
 Bangla text     Post image
 preprocessing   preprocessing
       │           │
       ▼           ▼
  BanglaBERT       CLIP
 text features   image features
       │           │
       └─────┬─────┘
             ▼
  Train-stat standardization
       + feature concatenation
             │
             ▼
       CCEF MLP head
             │
             ▼
 Authentic / Misleading / Satire
             │
             ▼
 Validation model selection
             │
             ▼
 Test evaluation + controls
```

### 4.3 Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | 64 |
| Maximum epochs | 30 |
| Early stopping | Patience 5 |
| Selection metric | Validation macro-F1 |
| Loss | Class-weighted cross-entropy |
| Random seeds | 42, 43, 44 |
| Headline prediction | Average probabilities across the three seeds |

Model selection is performed on validation data only. The test set is reserved for the specified evaluation contrasts, with the test-read caveat reported explicitly.

## 5. Controls and ablation studies

The project compares the fusion model with controls designed to test whether the improvement could be explained by simpler signals.

| Model / control | What it investigates |
|---|---|
| Text only (BanglaBERT) | Classification performance from text alone |
| Text + image-presence flag | Whether simply having an image explains the gain |
| Text + shuffled image | Whether the actual image content matters rather than an arbitrary image representation |
| Length only | Whether post length is predictive of the labels |
| Presence only | Whether image availability alone predicts the labels |
| Early fusion (BanglaBERT + CLIP) | Whether joint text-image features improve classification |

The controls are reported alongside the headline model rather than omitted when they produce unfavorable or confounding results.

## 6. Validation results

Mean ± standard deviation over three seeds. Validation was used for model selection.

| Model | Validation macro-F1 |
|---|---:|
| Text only (BanglaBERT) | 0.692 ± 0.005 |
| Text + presence flag | 0.701 ± 0.006 |
| Text + shuffled image | 0.681 ± 0.009 |
| **Early fusion (BanglaBERT + CLIP)** | **0.745 ± 0.013** |
| Length only | 0.452 ± 0.013 |
| Presence only | 0.179 |

## 7. Test results

Headline test scores are from the ensemble of the three seeds.

| Model | Test macro-F1 | Satire F1 |
|---|---:|---:|
| Text only (BanglaBERT) | 0.618 | 0.262 |
| Text + presence flag | 0.647 | 0.344 |
| Text + shuffled image | 0.646 | 0.327 |
| **Early fusion (BanglaBERT + CLIP)** | **0.722** | **0.480** |

The fusion model's per-seed test macro-F1 mean was **0.728**, with a range of **0.721–0.736**. The ensemble score of **0.722** is the headline result.

### Paired bootstrap contrasts

Fusion minus control, test macro-F1; paired bootstrap with 2,000 resamples and seed 42.

| Comparison | Difference | 95% bootstrap CI |
|---|---:|---:|
| Text only → fusion | +0.105 | [+0.060, +0.152] |
| Text + presence → fusion | +0.075 | [+0.024, +0.126] |
| Text + shuffled image → fusion | +0.076 | [+0.017, +0.132] |

The reported `p_b_better` is a bootstrap resample share, **not a p-value**. The three contrasts from the second test read were not corrected for multiple comparisons. Satire has only 32 test examples, so its confidence intervals are wide.

## 8. Shortcut and data-quality checks

### Post length

Post length is strongly associated with the label distribution:

- 88% of posts in the longest validation quartile are Authentic.
- 70% of Misleading posts fall in the two shortest quartiles.
- The length-only model achieves validation macro-F1 of 0.452.

Length is therefore treated as a measured confound, not ignored. The fusion improvement over text + presence was positive across all four validation length quartiles.

<div align="center">
<img src="docs/assets/499b_length_quartiles.png" alt="Validation macro-F1 by post-length quartile" width="100%">
</div>

### Near-duplicates

Five cross-split pairs had TF-IDF cosine similarity of at least 0.8. All five pairs shared a label; one pair involved a test row (out of 588 test rows).

### Reproducibility check

A rerun in the same environment reproduced the validation means, test per-seed values, and saved probabilities exactly (maximum difference 0.0). The complete environment record is still being improved; see [Limitations](#10-limitations).

## 9. Negative and exploratory results

These outcomes are documented to make the research process transparent.

| Component | Observed result | Current decision |
|---|---|---|
| Cultural features from LLM silver labels | Failed the blind human check | Disabled |
| Bangla Wikipedia BM25 evidence retrieval | Relevance below the pre-set 0.6 threshold: 13/40 (0.325), eval 49/120 (0.408), fresh pre-registered test 27/120 (0.225) | Display-only; does not affect predictions |
| Dense / hybrid retrieval | No gain over BM25 on dev (0.225 vs 0.242) | BM25 retained for the study |
| OCR text from images | Pilot on 14 hand-typed references had pooled character error rate 0.599 | Not used to support a model-performance claim |

## 10. Limitations

- **Single dataset:** no external-dataset or source-held-out evaluation has been completed.
- **Class imbalance:** satire has 193 examples in total and only 32 in the test split.
- **Shortcut signals:** post length and image presence correlate with labels; these were measured but not eliminated.
- **Test-set reuse:** the test split was read twice. Results should be treated as measured experimental findings, not a pristine final estimate.
- **Text truncation:** text is limited to 128 tokens.
- **Frozen encoders:** encoder fine-tuning has not been compared under the same evaluation protocol.
- **Explanations:** model sensitivity or feature attribution is not evidence that a claim is true; no human explanation evaluation has been conducted.
- **Environment logging:** GPU type, PyTorch version, and feature-file date are not yet recorded consistently for every run.

## 11. Repository structure

The branch is organized around a reproducible pipeline, research decisions, and auditability.

```text
.
├── app/                  # Gradio capstone app and consistency checker
├── configs/              # Configuration and BTMD schema
├── data/                 # Local raw/processed data (gitignored)
├── data_templates/       # Empty annotation and evidence templates
├── docs/
│   ├── decisions/        # DECISIONS.md: research decisions D-1 to D-16
│   ├── results/          # CANONICAL_NUMBERS.md: canonical reported numbers
│   ├── roadmap/          # Limitation fixes and next-task specifications
│   └── audits/           # Progress audits
├── notebooks/            # Colab/research notebooks; see the branch for the current inventory
├── scripts/              # Numbered pipeline scripts 00–16
├── src/                  # Data, features, fusion, metrics, evidence, OCR
├── tests/                # CPU-only pure-logic tests
└── README.md
```

Scripts 15 and 16 cover the evidence study. The canonical source for reported numbers is `docs/results/CANONICAL_NUMBERS.md`; research decisions are documented in `docs/decisions/DECISIONS.md`.

## 12. Setup and reproduction

### 12.1 Requirements

- Python environment compatible with the pinned dependencies in `requirements.txt`
- Google Colab with a T4 GPU for feature extraction (recommended for the full pipeline)
- BTMD table and associated images obtained separately
- Google Drive for storing the dataset, extracted features, and results

The raw dataset, features, checkpoints, and result files are intentionally not committed to Git.

### 12.2 Run local tests (no dataset or GPU required)

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and run the test suite:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows PowerShell:**

```powershell
$env:PYTHONPATH="."; pytest -q tests/
```

**macOS / Linux:**

```bash
PYTHONPATH=. pytest -q tests/
```

The test suite is designed to run on CPU; a GPU is not required for these unit tests. A GPU is recommended for the full feature-extraction/training workflow.

### 12.3 Run the full pipeline in Google Colab

Keep the repository and dataset in separate Drive directories, for example:

```text
MyDrive/
├── Bangla-Multimodal-FakeNews/   # Repository checkout
└── BFND-data/
    ├── data/raw/                 # BTMD table and images (user-provided)
    ├── features/                 # Generated feature files
    └── results/                  # Generated experiment outputs
```

1. Open the `notebooks/` directory on the `cse499b-capstone` branch and start with the supervised-run guide or the notebook documented as the current entry point. The branch contains additional application and project notebooks beyond the numbered core workflow; this README intentionally does not present an incomplete notebook inventory.
2. Mount Google Drive and complete the input checks in the notebook.
3. Select **Runtime → Change runtime type → T4 GPU**. A GPU is recommended for feature extraction.
4. Run the notebooks and numbered scripts in their documented order.
5. Use the generated results and `docs/results/CANONICAL_NUMBERS.md` when preparing reports.

### 12.4 Reproducing the reported contrasts

The current documentation identifies the following scripts for reproducing the model-selection, control, and bootstrap analyses:

- Script `08`: model selection
- Script `10`: control experiments
- Script `12`: paired bootstrap analysis

These scripts write outputs to the configured results directory. Confirm the input paths and configuration before running them. Do not use the test set for additional model selection or tuning.

## 13. Data and responsible-use policy

- BTMD is used under its own license and terms. The dataset and images are not redistributed in this repository.
- Do not commit raw data, features, checkpoints, or results. These directories are excluded through `.gitignore`.
- Predictions are research outputs, not verified statements about truth.
- Do not use the model to label real people, journalists, publications, or organizations as fake or deceptive.
- A model prediction should not replace independent evidence review or professional fact-checking.

## 14. Team and supervision

**Faculty advisor:** Dr. Mohammad Shifat-E-Rabbi (MSRB), Assistant Professor, Department of Computer Science & Engineering, North South University, Dhaka, Bangladesh.

| Team member | Student ID | Email |
|---|---|---|
| Anas Ahmed | 2121264042 | anas.ahmed@northsouth.edu |
| Subroto Ghosh | 2212150642 | subroto.ghosh@northsouth.edu |
| Joy Saha | 2312466042 | joy.saha1@northsouth.edu |
| Md. Abu Sufian Protik | 2312234042 | abu.protik@northsouth.edu |

## 15. Research documentation

For the detailed experiment history and exact decisions, consult:

- `docs/decisions/DECISIONS.md` — decisions D-1 through D-16
- `docs/results/CANONICAL_NUMBERS.md` — canonical metrics and reported values
- `docs/roadmap/` — limitation-fix plan and next-task specifications
- `docs/audits/` — project progress and audit records

---

**Project status:** Research prototype · CSE 499B · Bangla multimodal classification
