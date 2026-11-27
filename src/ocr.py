"""EasyOCR (Bengali + English): extract text from meme and news images.

Two ways to read an image:
- ocr_text / ocr_pil: the old reading, all text merged into one paragraph (kept for compatibility).
- ocr_pil_boxes + clean_ocr_boxes: each text box keeps its confidence and height, so the main text
  (usually the largest box, e.g. the headline) can be kept and watermark or button text dropped.

clean_ocr_boxes is pure Python and runs without easyocr. This is an input-side heuristic: the
model was not trained on OCR text, and the accuracy of this filter has not been measured.
"""
import re

# Words that usually come from buttons, watermarks or channel branding, not from the news itself.
# Edit this list after looking at real images; it is a heuristic, not a list of facts.
NOISE_TERMS = [
    "লিংক", "কমেন্ট", "বিস্তারিত", "ফলো", "সাবস্ক্রাইব", "লাইক", "শেয়ার", "ক্লিক",
    "follow", "subscribe", "like", "share", "click", "link", "comment",
]
_URL = re.compile(r"https?://|www\.|\.com\b|\.net\b|\.org\b|\.bd\b", re.IGNORECASE)


def load_ocr_reader(langs, use_gpu: bool):
    import easyocr

    return easyocr.Reader(langs, gpu=use_gpu)


def ocr_text(reader, image_path: str) -> str:
    return " ".join(reader.readtext(str(image_path), detail=0, paragraph=True)).strip()


def ocr_pil(reader, pil_image) -> str:
    """Same reading as ocr_text(), for an image already loaded in memory (the demo app upload)."""
    import numpy as np

    arr = np.array(pil_image.convert("RGB"))
    return " ".join(reader.readtext(arr, detail=0, paragraph=True)).strip()


def ocr_pil_boxes(reader, pil_image) -> list:
    """Return one dict per text box: {text, conf, top, left, height}. Boxes are not merged."""
    import numpy as np

    arr = np.array(pil_image.convert("RGB"))
    out = []
    for bbox, text, conf in reader.readtext(arr, detail=1, paragraph=False):
        out.append(box_record(bbox, text, conf))
    return out


def box_record(bbox, text, conf) -> dict:
    """bbox is EasyOCR's 4 corner points [[x, y], ...]. Returns the geometry we need."""
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    return {
        "text": str(text).strip(),
        "conf": float(conf),
        "top": min(ys),
        "left": min(xs),
        "height": max(ys) - min(ys),
    }


def _has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def clean_ocr_boxes(boxes, min_conf: float = 0.4, min_rel_height: float = 0.6,
                    noise_terms=NOISE_TERMS):
    """Keep the main text and drop noise.

    Rules, applied to each box:
      low_conf      confidence below min_conf
      too_short     fewer than 2 characters after stripping
      no_letters    only digits and symbols (for example '45E01' style junk without words)
      url           contains a web address
      noise_term    contains a word from noise_terms (case-insensitive)
      small_text    box height below min_rel_height times the tallest box (watermarks, small captions)

    Returns (kept_text, dropped), where dropped is a list of {text, reason}. Kept boxes are joined
    in reading order (top to bottom, then left to right).
    """
    if not boxes:
        return "", []
    max_height = max(b["height"] for b in boxes) or 1.0
    terms = [t.lower() for t in noise_terms]
    kept, dropped = [], []
    for b in boxes:
        text = b["text"]
        reason = None
        if b["conf"] < min_conf:
            reason = "low_conf"
        elif len(text) < 2:
            reason = "too_short"
        elif not _has_letters(text):
            reason = "no_letters"
        elif _URL.search(text):
            reason = "url"
        elif any(t in text.lower() for t in terms):
            reason = "noise_term"
        elif b["height"] / max_height < min_rel_height:
            reason = "small_text"
        if reason is None:
            kept.append(b)
        else:
            dropped.append({"text": text, "reason": reason})
    kept.sort(key=lambda b: (round(b["top"] / max_height * 10), b["left"]))
    return " ".join(b["text"] for b in kept).strip(), dropped
