"""Four-class final decision rule. Does not require torch.

Classifier (Authentic / Misleading / Satire) + evidence verdict (supports / refutes / insufficient)
-> Authentic / Misleading / Satire / Insufficient.
"""
import numpy as np


def final_decision(cls_label: str, evidence_verdict: str, cls_confidence: float,
                   min_confidence: float) -> str:
    if cls_confidence < min_confidence:
        return "Insufficient"
    if cls_label == "Satire":
        return "Satire"                      # satire is not gated by the fact-check evidence
    if cls_label == "Authentic" and evidence_verdict == "supports":
        return "Authentic"
    if cls_label == "Misleading" and evidence_verdict == "refutes":
        return "Misleading"
    return "Insufficient"                    # classifier and evidence disagree -> human review


def choose_min_confidence(val_confidences, target_coverage=0.85) -> float:
    """Pick the threshold from validation confidences so that about target_coverage of
    the samples receive an answer.

    The threshold is never chosen by looking at the test set.
    """
    conf = np.asarray(val_confidences, dtype=float)
    return float(np.quantile(conf, 1.0 - target_coverage))
