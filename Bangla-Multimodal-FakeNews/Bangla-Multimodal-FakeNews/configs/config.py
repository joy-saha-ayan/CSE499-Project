"""Central configuration: paths, labels, model names and hyperparameters.

Data, features and checkpoints live on Google Drive (BASE). The code itself runs from /content.
Set the environment variable BFND_BASE to change BASE.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("BFND_BASE", "/content/drive/MyDrive/BFND-data"))

RAW_DIR = BASE / "data" / "raw"                 # slim BTMD: table file + image folder
PROCESSED_DIR = BASE / "data" / "processed"     # unified.csv, unified_split.csv
EVIDENCE_CSV = BASE / "data" / "evidence" / "evidence.csv"
ANNOTATION_DIR = BASE / "data" / "annotations"
FEATURE_DIR = BASE / "features"
CHECKPOINT_DIR = BASE / "checkpoints"
RESULTS_DIR = BASE / "results"

# ---- Labels -----------------------------------------------------------------
LABELS_3 = ["Authentic", "Misleading", "Satire"]          # classifier output
LABEL2ID = {name: i for i, name in enumerate(LABELS_3)}
FINAL_LABELS = LABELS_3 + ["Insufficient"]                 # decision layer output

# BTMD label -> project label. An unknown label makes data_prep raise an error.
BTMD_LABEL_MAP = {"Real": "Authentic", "Misinformation": "Misleading", "Satire": "Satire"}
EXCLUDED_BTMD_LABELS = {"Clickbait"}   # left out of the main 3-class task; reported separately

CULTURAL_COLS = ["sarcasm", "humor", "local_reference", "historical_reference"]

# ---- AI labelling and evidence (decisions D-1 and D-2, see docs/decisions/DECISIONS.md) ----
CACHE_DIR = BASE / "cache"                       # resumable JSONL caches for LLM and Wikipedia calls
CULTURAL_HUMAN_CSV = ANNOTATION_DIR / "cultural_annotations.csv"   # manual labels (used if present)
CULTURAL_SILVER_CSV = ANNOTATION_DIR / "cultural_silver_labels.csv"  # LLM labels (used if no manual file)
CULTURAL_CHECK_CSV = ANNOTATION_DIR / "cultural_human_check.csv"    # 40 blind rows for a human to label
# Decision D-2 (2026-11-21): the 40-row human check failed for sarcasm, humor and local_reference
# (kappa < 0.4). Cultural features stay out of E4 and E5 until new labels pass the check.
USE_CULTURAL_FEATURES = False
EVIDENCE_RELEVANCE_CHECK_CSV = ANNOTATION_DIR / "evidence_relevance_check.csv"  # 40 rows to judge Wikipedia relevance
EVIDENCE_SOURCE = "wikipedia"                    # D-1: Bangla Wikipedia, retrieved from claim text only
LLM_BACKEND = "local"                            # "local": free open model on the Colab GPU; "anthropic": paid API
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"     # fits a T4 in fp16; Bangla quality must be checked with the 40-row sheets
LOCAL_LLM_MAX_NEW_TOKENS = 80                    # the JSON answers are short
LLM_MODEL = "claude-haiku-5-5"                   # used only when LLM_BACKEND = "anthropic"
LLM_MAX_INPUT_CHARS = 2000                       # post text is cut to this length before it is sent to the LLM
LLM_WORKERS = 8                                  # parallel API calls (anthropic backend; local backend uses 1)
LLM_MAX_RETRIES = 5
WIKI_API = "https://bn.wikipedia.org/w/api.php"
WIKI_SEARCH_LIMIT = 2                            # Wikipedia pages kept per search query
WIKI_CHUNK_CHARS = 500                           # passage length for retrieval
WIKI_BATCH = 20                                  # page ids per extract request

# ---- Pretrained models (Hugging Face) ----------------------------------------
TEXT_MODEL = "csebuetnlp/banglabert"
CLIP_IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"                # image encoder (original CLIP)
CLIP_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # Bangla text encoder (text only)
EMBED_MODEL = "intfloat/multilingual-e5-base"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
OCR_LANGS = ["bn", "en"]

# ---- Model-selection grid (scripts/07, 08; notebooks/03_model_selection.ipynb) -------------
# Frozen backbones: every backbone gets the same protocol (frozen embedding + the same head).
# Existing feature files are reused so that nothing is recomputed: bert.npy (BanglaBERT) and
# clip_img.npy (CLIP image). The other backbones are extracted by scripts/07.
TEXT_BACKBONES = {
    "banglabert": TEXT_MODEL,
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}
TEXT_FEATURE_FILES = {"banglabert": "bert.npy", "mbert": "text_mbert.npy", "xlmr": "text_xlmr.npy"}
IMAGE_BACKBONES = {
    "clip": CLIP_IMAGE_MODEL,
    "vit": "google/vit-base-patch16-224",
    "swin": "microsoft/swin-tiny-patch4-window7-224",
}
IMAGE_FEATURE_FILES = {"clip": "clip_img.npy", "vit": "img_vit.npy", "swin": "img_swin.npy"}
FUSION_KINDS = ["early", "intermediate", "late"]
SELECTION_DIR = RESULTS_DIR / "selection"
SELECTION_CKPT_DIR = CHECKPOINT_DIR / "selection"

# ---- Training hyperparameters -------------------------------------------------
SEED = 42
SEEDS = [42, 43, 44]
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3

FUSION_EPOCHS = 30
FUSION_LR = 1e-4
FUSION_PATIENCE = 5

# ---- Pipeline switches ---------------------------------------------------------
USE_OCR = False          # start with False for the baseline; set True later for the ablation
TOP_K = 5
HYBRID_ALPHA = 0.5       # weight of the BM25 score versus the dense score
NLI_THRESHOLD = 0.5
TARGET_COVERAGE = 0.85   # share of validation samples that receive an answer (the rest = Insufficient)
