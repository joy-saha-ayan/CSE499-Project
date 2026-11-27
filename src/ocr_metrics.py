"""Character and word error rate for OCR output against a hand-typed reference.

Pure Python. No torch, no network, no model. The reference text must be typed by a person.
The OCR output is never used as its own reference.
"""
import unicodedata


def normalise(text: str) -> str:
    """Unicode NFC, collapse runs of whitespace, strip the ends. Bangla vowel signs are kept."""
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.split())


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    a, b = list(a), list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate = edit distance / length of the reference. Empty reference gives 0 only if hypothesis is empty too."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else float("inf")
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on whitespace-separated tokens."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else float("inf")
    return edit_distance(ref, hyp) / len(ref)


def corpus_cer(pairs) -> dict:
    """Pooled CER over (reference, hypothesis) pairs: total edits divided by total reference characters.

    Pooling weights long images more than short ones. The per-image mean is returned too.
    """
    total_edits, total_chars, per_image = 0, 0, []
    for ref, hyp in pairs:
        r, h = normalise(ref), normalise(hyp)
        total_edits += edit_distance(r, h)
        total_chars += len(r)
        per_image.append(cer(r, h))
    finite = [x for x in per_image if x != float("inf")]
    return {
        "images": len(per_image),
        "reference_chars": total_chars,
        "pooled_cer": (total_edits / total_chars) if total_chars else float("nan"),
        "mean_per_image_cer": (sum(finite) / len(finite)) if finite else float("nan"),
    }
