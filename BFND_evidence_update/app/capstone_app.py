"""Capstone web app: Bangla multimodal fake-news detection with explanations.

Ten features, one per tab or part of a tab:
  1. Predict            fused verdict from text + optional image, with Bangla meaning and input checks
  2. Modality check     text-only vs image-only vs fused, and whether the image changed the decision
  3. Explain            word-level occlusion: green = pushes towards the predicted class, red = away from it
  4. Robustness         seed vote and truncation curve (first 25 / 50 / 75 / 100 % of the words)
  5. Similar posts      nearest TRAIN posts by BanglaBERT similarity, with their labels
  6. Batch CSV          many texts at once, downloadable predictions, metrics if a label column exists
  7. Leaderboard        every configuration; winners chosen on validation only
  8. Error analysis     per-class precision / recall / F1 and confusion matrix of the winner
  9. Model card         what the model is, what it is not, known issues, decision log
 10. Evidence           Bangla Wikipedia passages for the post (BM25), with the relevance-gate status.
                        Display only: the prediction does not use them (decision D-5).

Needs the outputs of: 01 (data), 02 (features), 08 (checkpoints and results/selection).
The Evidence tab also needs data/evidence/evidence.csv (script 06) and, for the gate status,
results/evidence_gate.json (script 09).
Run in Colab after those steps:
    from app.capstone_app import load_runtime, build_ui
    load_runtime()
    demo = build_ui()
    demo.launch()                      # inline if the notebook shows it; otherwise see the notes

Public link: off by default. Set BFND_PUBLIC_LINK=1 only after you have decided to share it.

The prediction uses the text and the image only. Wikipedia evidence and cultural features are not
part of it (see the Model card tab). The Evidence tab shows retrieved passages and never changes a verdict.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    CLIP_IMAGE_MODEL,
    EVIDENCE_CSV,
    EVIDENCE_GATE_JSON,
    EVIDENCE_RELEVANCE_MIN,
    EVIDENCE_TOP_K,
    FACT_CHECK_DOMAINS,
    FEATURE_DIR,
    OCR_LANGS,
    RESULTS_DIR,
    SEEDS,
    SELECTION_CKPT_DIR,
    TEXT_MODEL,
)
from src.evidence_app import (  # noqa: E402
    GATE_FAIL,
    GATE_PASS,
    BM25Index,
    drop_fact_check_evidence,
    gate_status,
    load_gate,
    top_k_passages,
)
from src.interpret import (  # noqa: E402
    LABEL_MEANING,
    accuracy_macro_f1,
    confidence_band,
    highlight_html,
    input_checks,
    input_mode,
    nearest_neighbours,
    occlusion_attribution,
    per_class_from_confusion,
    seed_agreement,
    top_drivers,
    truncation_curve,
)
from src.serving import (  # noqa: E402
    LABELS,
    ImageEncoder,
    TextEncoder,
    fused_probs_for_text,
    image_only_probs,
    leaderboard_rows,
    load_bundle,
    load_reference,
    predict_post,
)

NEED_TEXT_MSG = ("এই tab-এর জন্য text দরকার। Text লিখুন, অথবা ছবি দিয়ে "
                 "'ছবি থেকে text বের করুন (OCR)' button চাপুন।")
OCR_TRAINING_NOTE = ("OCR text training-এ ছিল না (training-এ USE_OCR = False ছিল)। "
                     "তাই OCR-এর পরের prediction-কে নিশ্চিত ধরবেন না, পড়ে দেখে নিন।")
_OCR_READER = None

MAX_BATCH_ROWS = 500


@dataclass
class Runtime:
    device: str
    bundle: object
    text_enc: TextEncoder
    img_enc: ImageEncoder
    ref: dict
    ev_df: pd.DataFrame | None = None     # Wikipedia passages after the leakage guard (None = not built)
    ev_index: BM25Index | None = None
    ev_gate: dict | None = None           # results/evidence_gate.json (None = not written yet)


RT: Runtime | None = None


def _load_evidence():
    """Return (passages, BM25 index), or (None, None) when script 06 has not written evidence.csv."""
    if not Path(EVIDENCE_CSV).exists():
        return None, None
    ev = pd.read_csv(EVIDENCE_CSV).fillna("")
    ev, _ = drop_fact_check_evidence(ev, FACT_CHECK_DOMAINS)
    return ev, BM25Index(ev["evidence_text"].astype(str).tolist())


def load_runtime() -> Runtime:
    """Load the models and the saved heads once. Call this before build_ui()."""
    global RT
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = load_bundle(FEATURE_DIR, SELECTION_CKPT_DIR, RESULTS_DIR, SEEDS)
    ev_df, ev_index = _load_evidence()
    RT = Runtime(
        device=device,
        bundle=bundle,
        text_enc=TextEncoder(TEXT_MODEL, device),
        img_enc=ImageEncoder(CLIP_IMAGE_MODEL, device),
        ref=load_reference(FEATURE_DIR),
        ev_df=ev_df,
        ev_index=ev_index,
        ev_gate=load_gate(EVIDENCE_GATE_JSON),
    )
    return RT


def _rt() -> Runtime:
    if RT is None:
        raise RuntimeError("Runtime is not loaded. Call load_runtime() first.")
    return RT


def _probs_row(name: str, probs) -> dict:
    return {"model": name, **{lab: round(float(p), 3) for lab, p in zip(LABELS, probs)}}


def _checks_md(checks) -> str:
    if not checks:
        return "সতর্কতা নেই।"
    tag = {"warn": "[সতর্কতা]", "info": "[তথ্য]"}
    return "\n".join(f"- {tag[level]} {msg}" for level, msg in checks)


# 1 + 2: Predict and modality check ---------------------------------------------------------------

def predict_handler(text, image):
    rt = _rt()
    text = (text or "").strip()
    mode = input_mode(text, image is not None)

    if mode == "empty":
        return "Text লিখুন অথবা ছবি upload করুন।", None, None, ""

    if mode == "image_only":
        img_vec = rt.img_enc.embed(image)
        probs = image_only_probs(rt.bundle, img_vec)
        pred = int(np.argmax(probs))
        predicted = LABELS[pred]
        p = float(probs[pred])
        verdict = (
            f"### {LABEL_MEANING[predicted]}\n\n"
            f"- Mode: **image only** (কোনো text নেই)\n"
            f"- Probability: **{p:.3f}**. Confidence band: **{confidence_band(p)}** "
            "(display band only, not calibrated).\n"
            "- এটি trained CLIP image model-এর ফল। Fused model কখনো ফাঁকা text দিয়ে train হয়নি, "
            "তাই সেটা এখানে ব্যবহার হয়নি।"
        )
        labels = {lab: float(v) for lab, v in zip(LABELS, probs)}
        df = pd.DataFrame([_probs_row("Image only: CLIP head", probs)])
        note = ("- [তথ্য] ছবিতে লেখা text থাকলে 'ছবি থেকে text বের করুন (OCR)' চাপুন। "
                "তারপর text সহ fused prediction পাবেন।\n- [সতর্কতা] " + OCR_TRAINING_NOTE)
        return verdict, labels, df, note

    out = predict_post(rt.bundle, rt.text_enc, rt.img_enc, text, image)
    pred = int(np.argmax(out["fused"]))
    predicted = LABELS[pred]
    p = float(out["fused"][pred])
    n_agree = sum(int(np.argmax(p)) == pred for p in out["fused_per_seed"])

    verdict = (
        f"### {LABEL_MEANING[predicted]}\n\n"
        f"- Fused probability: **{p:.3f}**. Confidence band: **{confidence_band(p)}** "
        "(display band only, not calibrated).\n"
        f"- Seed vote: {n_agree} out of {len(out['fused_per_seed'])} seeds agree on this class.\n"
        f"- Image: {'given' if out['has_image'] else 'not given (zero vector, as in training)'}."
    )
    labels = {lab: float(v) for lab, v in zip(LABELS, out["fused"])}

    rows = [_probs_row("Fused: BanglaBERT + CLIP (3-seed mean)", out["fused"]),
            _probs_row("Text only: BanglaBERT head", out["text_only"])]
    if out["image_only"] is not None:
        rows.append(_probs_row("Image only: CLIP head", out["image_only"]))
        text_pred = LABELS[int(np.argmax(out["text_only"]))]
        if text_pred != predicted:
            change = f"ছবি যোগ করার পরে prediction **বদলে গেছে**: text-only `{text_pred}` → fused `{predicted}`।"
        else:
            change = f"ছবি যোগ করার পরেও prediction একই থেকেছে (`{predicted}`)।"
        modality_note = change + (" Text-only head হলো BanglaBERT head। Grid-এর text winner "
                                  "XLM-R আলাদা, কারণ দুটোর validation মান প্রায় সমান।")
    else:
        modality_note = "ছবি না থাকায় fused prediction প্রায় শুধু text-এর উপর নির্ভর করছে।"
    modality_df = pd.DataFrame(rows)
    modality_md = modality_note

    checks = input_checks(text, out["has_image"], out["n_tokens"])
    checks_block = _checks_md(checks) + f"\n\nModel-এ যাওয়া text ({out['n_chars']} character): " \
        f"`{out['cleaned_text'][:200]}`" + (" …" if out["n_chars"] > 200 else "")

    return verdict, labels, modality_df, checks_block + "\n\n" + modality_md


# OCR helper (optional, fills the text box; the model input stays the typed text) ---------------

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        from src.ocr import load_ocr_reader

        _OCR_READER = load_ocr_reader(OCR_LANGS, use_gpu=(_rt().device == "cuda"))
    return _OCR_READER


def ocr_handler(image):
    if image is None:
        return gr.update(), "আগে একটা ছবি upload করুন।"
    if importlib.util.find_spec("easyocr") is None:
        return gr.update(), ("EasyOCR install করা নেই। Colab-এ `!pip install easyocr` চালিয়ে "
                             "app আবার চালু করুন।")
    from src.ocr import ocr_pil

    text = ocr_pil(_get_ocr_reader(), image)
    if not text:
        return gr.update(), "ছবিতে text পাওয়া যায়নি। নিজে text লিখুন।"
    return text, (f"OCR থেকে {len(text)} character পাওয়া গেছে এবং text box-এ বসানো হয়েছে। "
                  "পড়ে দেখে দরকার হলে বদলান। " + OCR_TRAINING_NOTE)


# 3: Explain --------------------------------------------------------------------------------------

def explain_handler(text, image, max_units):
    rt = _rt()
    text = (text or "").strip()
    if not text:
        return "", NEED_TEXT_MSG
    img_vec = rt.img_enc.embed(image)
    probs, _ = fused_probs_for_text(rt.bundle, rt.text_enc, text, img_vec)
    target = int(np.argmax(probs))

    def prob_fn(t):
        return fused_probs_for_text(rt.bundle, rt.text_enc, t, img_vec)[0]

    res = occlusion_attribution(text, prob_fn, target, max_units=int(max_units))
    support, oppose = top_drivers(res["spans"], k=5)

    def fmt(rows):
        if not rows:
            return "- (কোনোটি নেই)"
        return "\n".join(f"- `{r['text']}` : {r['delta']:+.3f}" for r in rows)

    md = (
        f"**Target class:** `{LABELS[target]}`, full-text probability **{res['full_prob']:.3f}**. "
        f"Text কে {len(res['spans'])} টি অংশে ভাগ করে প্রতিটি অংশ আলাদা করে সরানো হয়েছে।\n\n"
        "**বেশি সমর্থন করা অংশ (সরালে probability কমে):**\n" + fmt(support) + "\n\n"
        "**বেশি বিপক্ষে যাওয়া অংশ (সরালে probability বাড়ে):**\n" + fmt(oppose) + "\n\n"
        "_এটা model-এর sensitivity দেখায়, সত্য-মিথ্যার প্রমাণ নয়। একটি অংশ সরালে probability বদলালে "
        "সেটা শুধু বোঝায় যে model ওই অংশের উপর নির্ভর করেছে।_"
    )
    return highlight_html(res["spans"]), md


# 4: Robustness -----------------------------------------------------------------------------------

def robustness_handler(text, image):
    rt = _rt()
    text = (text or "").strip()
    if not text:
        return None, NEED_TEXT_MSG
    out = predict_post(rt.bundle, rt.text_enc, rt.img_enc, text, image)
    vote = seed_agreement(out["fused_per_seed"])
    per_seed = ", ".join(
        f"seed {s}: `{LABELS[int(np.argmax(p))]}`" for s, p in zip(rt.bundle.seeds, out["fused_per_seed"])
    )
    img_vec = out["image_embedding"]

    def prob_fn(t):
        return fused_probs_for_text(rt.bundle, rt.text_enc, t, img_vec)[0]

    rows = []
    for r in truncation_curve(text, prob_fn):
        probs = r["probs"]
        rows.append({"first share of words": f"{int(round(r['fraction'] * 100))}%",
                     "words used": r["words"],
                     "predicted": LABELS[int(np.argmax(probs))],
                     **{lab: round(float(v), 3) for lab, v in zip(LABELS, probs)}})
    df = pd.DataFrame(rows)
    preds = [row["predicted"] for row in rows]
    stable = len(set(preds)) == 1
    md = (
        f"**Seed vote:** {per_seed}. {round(vote['agreement'] * len(out['fused_per_seed']))} out of "
        f"{len(out['fused_per_seed'])} seeds একই class দিয়েছে; majority-class probability-র seed-spread "
        f"**{vote['spread_majority_prob']:.3f}**।\n\n"
        + ("**Truncation:** সব অংশে prediction একই আছে।" if stable else
           "**Truncation:** post-এর অংশ বদলালে prediction বদলায়। অর্থাৎ শেষ অংশের তথ্য prediction-এর জন্য "
           "গুরুত্বপূর্ণ, কিংবা prediction অস্থিতিশীল।")
        + "\n\n_Truncation test-এ শেষের অংশ বাদ দেওয়া হয়। 100% সারিটি পূর্ণ post-এর ফল, যা Predict tab-এর সাথে মেলা উচিত।_"
    )
    return df, md


# 5: Similar known posts --------------------------------------------------------------------------

def similar_handler(text, image, k):
    rt = _rt()
    text = (text or "").strip()
    if not text:
        return None, NEED_TEXT_MSG
    out = predict_post(rt.bundle, rt.text_enc, rt.img_enc, text, None)
    hits = nearest_neighbours(out["text_embedding"], rt.ref["embeddings"], k=int(k))
    rows = []
    for rank, (idx, sim) in enumerate(hits, start=1):
        rows.append({"rank": rank,
                     "similarity": round(sim, 3),
                     "known label (train)": LABELS[int(rt.ref["labels"][idx])],
                     "train post (preview)": rt.ref["texts"][idx][:160]})
    df = pd.DataFrame(rows)
    counts = pd.Series([r["known label (train)"] for r in rows]).value_counts()
    summary = ", ".join(f"{lab}: {n}" for lab, n in counts.items())
    md = (
        f"**পাশের {len(hits)} টি known train post-এর label:** {summary}।\n\n"
        "_এখানে শুধু train split থেকে নেওয়া post দেখানো হয়, validation বা test থেকে নয়। "
        "Similarity বলে শব্দ ও লেখার ধরনের মিল, সত্যতার মিল নয়। একই label থাকা মানে সত্য হওয়া নয়।_"
    )
    return df, md


# 6: Batch CSV ------------------------------------------------------------------------------------

def batch_handler(file_path):
    rt = _rt()
    if file_path is None:
        return "CSV upload korun। Column 'text' লাগবে। 'label' column থাকলে metrics দেখাবে।", None, None
    df = pd.read_csv(file_path)
    if "text" not in df.columns:
        return "CSV-তে 'text' column নেই।", None, None
    if len(df) > MAX_BATCH_ROWS:
        return f"একবারে সর্বোচ্চ {MAX_BATCH_ROWS} টি row। আপনার file-এ {len(df)} টি আছে।", None, None

    zero_image = rt.img_enc.embed(None)
    preds, probs_rows = [], []
    for t in df["text"].fillna("").astype(str):
        if not t.strip():
            preds.append(None)
            probs_rows.append([np.nan] * len(LABELS))
            continue
        p, _ = fused_probs_for_text(rt.bundle, rt.text_enc, t, zero_image)
        preds.append(LABELS[int(np.argmax(p))])
        probs_rows.append(list(p))

    out_df = pd.DataFrame({"text_preview": df["text"].fillna("").astype(str).str.slice(0, 120),
                           "predicted": preds})
    for j, lab in enumerate(LABELS):
        out_df[f"p_{lab}"] = [round(r[j], 4) if r[j] == r[j] else None for r in probs_rows]

    out_path = Path(tempfile.gettempdir()) / "bfnd_batch_predictions.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    message = (f"{len(df)} টি row প্রক্রিয়া হয়েছে। সব post-এ image নেই, তাই fused model zero image vector "
               "দিয়ে চলেছে। এটা training-এর no-image row-এর মতো।")
    if "label" in df.columns:
        valid = [(str(y), p) for y, p in zip(df["label"], preds) if p is not None and str(y) in LABELS]
        if valid:
            y_true = [LABELS.index(y) for y, _ in valid]
            y_pred = [LABELS.index(p) for _, p in valid]
            acc, f1 = accuracy_macro_f1(y_true, y_pred, n_classes=len(LABELS))
            message += (f"\n\n**Metrics ({len(valid)} টি labelled row):** accuracy **{acc:.4f}**, "
                        f"macro-F1 **{f1:.4f}**। এটি নতুন data হলে external check। এই row-গুলো test set "
                        "থেকে হলে report-এর test number-এর সাথে মেলান, কিন্তু এখান থেকে model বদলাবেন না।")
        else:
            message += "\n\n'label' column আছে, কিন্তু মান Authentic / Misleading / Satire নয়, তাই metrics দেখানো হয়নি।"
    return message, out_df, str(out_path)


# 10: Evidence (display only, decision D-5) ------------------------------------------------------

def _evidence_banner(status: str, detail: str) -> str:
    if status == GATE_PASS:
        head = "Relevance gate: **PASS**"
    elif status == GATE_FAIL:
        head = "Relevance gate: **FAIL**"
    else:
        head = f"Relevance gate: **{status}**"
    return (
        f"{head} ({detail}).\n\n"
        "এই passage-গুলো prediction-এ ব্যবহার হয়নি। Prediction শুধু text ও image থেকে আসে। "
        "Passage গুলো Bangla Wikipedia থেকে নেওয়া, fact-check নয়, এবং এগুলো post সত্য না মিথ্যা তা প্রমাণ করে না।"
    )


def evidence_handler(text):
    rt = _rt()
    text = (text or "").strip()
    if not text:
        return None, NEED_TEXT_MSG
    if rt.ev_index is None:
        return None, ("Evidence database পাওয়া যায়নি (data/evidence/evidence.csv নেই)। "
                      "Script 06 চালান, তারপর app আবার চালু করুন।")
    status, detail = gate_status(rt.ev_gate, EVIDENCE_RELEVANCE_MIN)
    rows = top_k_passages(rt.ev_index, rt.ev_df, text, EVIDENCE_TOP_K)
    banner = _evidence_banner(status, detail)
    if not rows:
        return None, banner + "\n\nএই text-এর সাথে মিলে এমন কোনো Wikipedia passage পাওয়া যায়নি।"
    df = pd.DataFrame(rows)
    return df, (banner + f"\n\n{len(rows)} টি passage দেখানো হচ্ছে (BM25, post text দিয়ে query)। "
                "Score বেশি মানে শব্দের মিল বেশি, সত্যের প্রমাণ নয়।")


# 7 + 8 + 9: Static tabs built from the saved results ----------------------------------------------

def leaderboard_info(bundle) -> str:
    return (
        f"**Winner (validation only):** `{bundle.winner}`.\n\n"
        "Rule: highest mean validation macro-F1 over seeds; ties go to the smaller std; the test set is "
        "reported but never used to choose. Std uses ddof=0 over 3 seeds.\n\n"
        "**Near tie:** text winner `text:xlmr` and `text:banglabert` differ by only about 0.001 on validation, "
        "which is inside the std. Treat them as tied."
    )


def error_tables(bundle):
    cm = np.asarray(bundle.confusion, dtype=float)
    per_class = pd.DataFrame(per_class_from_confusion(cm, LABELS))
    cm_df = pd.DataFrame(np.round(cm, 1), index=[f"true: {l}" for l in LABELS],
                         columns=[f"pred: {l}" for l in LABELS])
    note = (
        "**Satire** test support মাত্র কয়েকটি post, তাই Satire-এর F1 অস্থির। "
        "**Length-only control** দুই জায়গায় আলাদা: grid-এর neural control head test macro-F1 "
        f"{bundle.summary['control:length_only']['test_macro_f1_mean']:.3f}, আর notebook 02-এর "
        "logistic-regression length-only baseline macro-F1 0.385। এগুলো দুটি আলাদা model, তাই report-এ "
        "কোনটি কোন নামে লিখেছেন তা স্পষ্ট করুন।"
    )
    return per_class, cm_df, note


def model_card_md(bundle) -> str:
    w = bundle.summary[bundle.winner]
    lw = bundle.summary["control:length_only"]
    return f"""
## What this is
- Bangla post-এর 3-class classifier: **Authentic / Misleading / Satire**।
- Input: post text এবং ঐচ্ছিক একটি ছবি।
- Model: BanglaBERT (frozen, mean pooling) + CLIP ViT-B/32 image embedding, early-fusion head (CCEF), 3 seed-এর গড়।
- Winner `{bundle.winner}` নির্বাচিত হয়েছে validation macro-F1 দিয়ে: **{w['val_macro_f1_mean']:.4f} ± {w['val_macro_f1_std']:.4f}**।

## What it is not
- **Fact-checker নয়।** Wikipedia evidence এই prediction-এ নেই (relevance gate FAIL, D-4)। Evidence tab শুধু passage দেখায়, D-5।
- **Cultural feature নেই** (D-2: validation FAIL, flag off)।
- **Clickbait নেই** (576 row বাদ, আলাদাভাবে report করার কথা)।
- **OCR training-এ ছিল না** (USE_OCR = False)। App-এ OCR button আছে শুধু input সহজ করার জন্য। OCR text বসালে সেটা একটা নতুন input হয়, যার মান এখনো মাপা হয়নি।
- **শুধু ছবি দিলে** trained CLIP image model ব্যবহার হয়, fused model নয়, কারণ fused model ফাঁকা text দিয়ে train হয়নি।
- BTMD-এর label মানে dataset-এর label। প্রতিটি label স্বাধীনভাবে যাচাই করা নয়।

## Test results (reported once, never used for selection)
- Test accuracy **{w['test_accuracy_mean']:.4f}**, test macro-F1 **{w['test_macro_f1_mean']:.4f}** (mean over 3 seeds)।
- Length-only control (grid): test macro-F1 **{lw['test_macro_f1_mean']:.4f}**। Length-এর শুধু সংকেতেও কিছুটা ফল আসে।

## Known issues (read before quoting numbers)
1. **Length shortcut:** Authentic post-এর median ১১৩৬ character, Misleading-এর median ১০০ character। Model length-এর উপর কিছুটা নির্ভর করতে পারে।
2. **Satire ছোট class:** test-এ ৩২ টি, তাই Satire-এর F1 অস্থির।
3. **Data loss:** raw ৬০৫৮ row থেকে ১৫৬৬ টি বাদ গেছে কারণ text ফাঁকা। এটি selection bias আনতে পারে।
4. **Truncation:** BanglaBERT মাত্র প্রথম ১২৮ token দেখে।
5. **Two different "main" models:** notebook 04-এর E3 (4 feature block: bert, clip_img, clip_txt, sim) আর grid-এর winner (2 block: banglabert, clip_img) আলাদা model। Report-এ কোনটি final, তা একবার পরিষ্কার লিখতে হবে।
6. **Single dataset**, group split by heading। বাইরের dataset-এ যাচাই করা হয়নি।

## Decision log
- **D-1:** evidence source = Bangla Wikipedia, claim text থেকে query।
- **D-2:** cultural feature = LLM silver label। Human check-এ agreement FAIL, তাই flag off।
- **D-3:** model selection = validation only, test শুধু report।
- **D-4:** evidence relevance FAIL (০.৩২৫, threshold ০.৬)। E5 এখন ablation হিসেবে, threshold post-hoc হিসেবে লেখা।
- **D-5:** Evidence tab শুধু দেখায়; BM25 দিয়ে post text থেকে query; fact-check domain guard; prediction অপরিবর্তিত।
"""


def build_ui():
    rt = _rt()
    bundle = rt.bundle
    leaderboard_df = pd.DataFrame(leaderboard_rows(bundle.summary, bundle.winner))
    per_class_df, cm_df, error_note = error_tables(bundle)

    with gr.Blocks(title="BFND: Bangla multimodal fake-news detector") as demo:
        gr.Markdown(
            "# Bangla multimodal fake-news detector\n"
            "Bangla post আর (ঐচ্ছিক) ছবি দিন। Model তিনটি label-এর একটি দেবে: "
            "Authentic, Misleading বা Satire। এটি fact-checker নয়; Model card দেখুন।"
        )
        with gr.Row():
            text_in = gr.Textbox(label="Bangla post text", lines=5,
                                 placeholder="এখানে Bangla post লিখুন বা paste করুন")
            image_in = gr.Image(type="pil", label="ছবি (ঐচ্ছিক)")
        with gr.Row():
            btn_ocr = gr.Button("ছবি থেকে text বের করুন (OCR)")
        ocr_md = gr.Markdown()
        btn_ocr.click(ocr_handler, inputs=[image_in], outputs=[text_in, ocr_md])

        with gr.Tabs():
            with gr.Tab("1. Predict"):
                btn_predict = gr.Button("Predict", variant="primary")
                verdict_md = gr.Markdown()
                label_out = gr.Label(num_top_classes=3, label="Fused probabilities")
                with gr.Accordion("2. Modality check and input checks", open=True):
                    modality_df = gr.Dataframe(label="Text-only vs image-only vs fused")
                    checks_md = gr.Markdown()
                btn_predict.click(predict_handler, inputs=[text_in, image_in],
                                  outputs=[verdict_md, label_out, modality_df, checks_md])

            with gr.Tab("3. Explain (word level)"):
                max_units = gr.Slider(10, 60, value=30, step=5,
                                      label="সর্বোচ্চ কতটি অংশে ভাগ হবে (বেশি = ধীর, সূক্ষ্ম)")
                btn_explain = gr.Button("Explain")
                highlight_out = gr.HTML()
                drivers_md = gr.Markdown()
                btn_explain.click(explain_handler, inputs=[text_in, image_in, max_units],
                                  outputs=[highlight_out, drivers_md])

            with gr.Tab("4. Robustness"):
                btn_robust = gr.Button("Check robustness")
                trunc_df = gr.Dataframe(label="Prediction when only the first share of words is used")
                robust_md = gr.Markdown()
                btn_robust.click(robustness_handler, inputs=[text_in, image_in],
                                 outputs=[trunc_df, robust_md])

            with gr.Tab("5. Similar known posts"):
                k_slider = gr.Slider(3, 10, value=5, step=1, label="কতটি similar post")
                btn_similar = gr.Button("Find similar training posts")
                similar_df = gr.Dataframe(label="Nearest TRAIN posts")
                similar_md = gr.Markdown()
                btn_similar.click(similar_handler, inputs=[text_in, image_in, k_slider],
                                  outputs=[similar_df, similar_md])

            with gr.Tab("6. Batch CSV"):
                gr.Markdown(f"CSV-তে `text` column লাগবে (সর্বোচ্চ {MAX_BATCH_ROWS} row)। "
                            "`label` column থাকলে accuracy ও macro-F1 দেখাবে।")
                file_in = gr.File(file_types=[".csv"], type="filepath", label="CSV upload")
                btn_batch = gr.Button("Run batch")
                batch_md = gr.Markdown()
                batch_df = gr.Dataframe(label="Predictions")
                batch_file = gr.File(label="Download predictions CSV")
                btn_batch.click(batch_handler, inputs=[file_in],
                                outputs=[batch_md, batch_df, batch_file])

            with gr.Tab("7. Leaderboard"):
                gr.Markdown(leaderboard_info(bundle))
                gr.Dataframe(value=leaderboard_df, label="All configurations (mean over 3 seeds)")

            with gr.Tab("8. Error analysis"):
                gr.Markdown("Winner-এর test confusion matrix (3 seed-এর গড়)। Rows = true, columns = predicted।")
                gr.Dataframe(value=per_class_df, label="Per-class metrics (test)")
                gr.Dataframe(value=cm_df, label="Confusion matrix (test)")
                gr.Markdown(error_note)

            with gr.Tab("9. Model card"):
                gr.Markdown(model_card_md(bundle))

            with gr.Tab("10. Evidence (Wikipedia)"):
                gr.Markdown("Post text দিয়ে Bangla Wikipedia-র passage খোঁজা হয়। Relevance gate FAIL বা "
                            "সম্পন্ন না হলে এগুলো শুধু দেখার জন্য; prediction বদলায় না।")
                btn_evidence = gr.Button("Evidence খুঁজুন")
                evidence_md = gr.Markdown()
                evidence_df = gr.Dataframe(label="Retrieved Wikipedia passages")
                btn_evidence.click(evidence_handler, inputs=[text_in], outputs=[evidence_df, evidence_md])

    return demo


if __name__ == "__main__":
    load_runtime()
    build_ui().launch(share=os.environ.get("BFND_PUBLIC_LINK") == "1")
