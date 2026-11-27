"""Text-report explanation of a prediction.

The CLIP similarity is not shown as an image-region explanation. A region-level explanation
needs a separate technique (Grad-CAM or patch attribution).
"""

REASON_BY_DECISION = {
    "Authentic": "The classifier predicted Authentic and the retrieved evidence supports the claim.",
    "Misleading": "The classifier predicted Misleading and the retrieved evidence refutes the claim.",
    "Satire": "The classifier predicted Satire; the evidence gate does not apply to this class.",
    "Insufficient": "Confidence is below the threshold, or the classifier and the evidence disagree. Human review is needed.",
}


def build_explanation(detected_text, ocr_text, cls_label, cls_conf, evidence_verdict,
                      decision, evidences) -> str:
    """evidences: [{'source', 'text', 'entail', 'contra'}, ...]"""
    lines = [
        f"## Final verdict: {decision}",
        f"- Classifier: {cls_label} (confidence {cls_conf:.2f})",
        f"- Evidence verdict: {evidence_verdict}",
        "",
        "### Detected text",
        detected_text or "(none)",
    ]
    if ocr_text:
        lines += ["", "### Text from image (OCR)", ocr_text]
    lines += ["", "### Retrieved evidence"]
    if not evidences:
        lines.append("- No evidence found.")
    for e in evidences:
        lines.append(
            f"- [{e['source']}] {e['text'][:200]} "
            f"(entail={e['entail']:.2f}, contra={e['contra']:.2f})"
        )
    lines += ["", "### Reason", REASON_BY_DECISION[decision]]
    return "\n".join(lines)
