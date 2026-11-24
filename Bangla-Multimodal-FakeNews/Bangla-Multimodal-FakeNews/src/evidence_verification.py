"""NLI-based evidence verification. Premise = evidence, hypothesis = claim.

The pure helpers (map_nli_labels, aggregate_evidence, verdict_from_scores) run without torch.
"""
import numpy as np

NLI_KEYS = ("entailment", "contradiction", "neutral")


def map_nli_labels(id2label: dict) -> dict:
    """Find the entailment, contradiction and neutral indices from the model's id2label."""
    mapping = {}
    for idx, name in id2label.items():
        low = str(name).lower()
        for key in NLI_KEYS:
            if key[:5] in low:  # 'entai', 'contr', 'neutr'
                mapping[key] = int(idx)
    missing = set(NLI_KEYS) - set(mapping)
    if missing:
        raise ValueError(f"NLI model has no label for: {missing}; id2label={id2label}")
    return mapping


def aggregate_evidence(nli_probs) -> np.ndarray:
    """nli_probs: (K, 3) with columns [entail, contra, neutral].

    Output: [max_entail, max_contra, mean_neutral]. Returns [0, 0, 1] when there is no evidence.
    """
    p = np.asarray(nli_probs, dtype=float).reshape(-1, 3)
    if len(p) == 0:
        return np.array([0.0, 0.0, 1.0])
    return np.array([p[:, 0].max(), p[:, 1].max(), p[:, 2].mean()])


def verdict_from_scores(max_entail: float, max_contra: float, threshold: float = 0.5) -> str:
    if max_entail >= threshold and max_entail >= max_contra:
        return "supports"
    if max_contra >= threshold and max_contra > max_entail:
        return "refutes"
    return "insufficient"


class NLIVerifier:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.idx = map_nli_labels(self.model.config.id2label)

    def probs(self, premises, hypotheses, batch_size=16) -> np.ndarray:
        """Output (N, 3). Columns: [entailment, contradiction, neutral]."""
        cols = [self.idx["entailment"], self.idx["contradiction"], self.idx["neutral"]]
        premises, hypotheses = list(premises), list(hypotheses)
        out = []
        with self.torch.no_grad():
            for s in range(0, len(premises), batch_size):
                enc = self.tok(
                    premises[s:s + batch_size], hypotheses[s:s + batch_size],
                    truncation=True, max_length=256, padding=True, return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                p = self.torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(p[:, cols])
        return np.concatenate(out) if out else np.zeros((0, 3))


def evidence_features(verifier: NLIVerifier, claims, hits, evidence_texts) -> np.ndarray:
    """Build a 3-dim feature per claim from its top-k evidence."""
    feats = []
    for claim, hit_list in zip(claims, hits):
        if not hit_list:
            feats.append(aggregate_evidence([]))
            continue
        premises = [evidence_texts[j] for j, _ in hit_list]
        hypotheses = [claim] * len(hit_list)
        feats.append(aggregate_evidence(verifier.probs(premises, hypotheses)))
    return np.vstack(feats)
