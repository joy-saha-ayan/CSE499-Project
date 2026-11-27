"""Fusion heads for the model-selection grid. Three fusion kinds, one definition each:

- early:        concatenate the text and image vectors, then one MLP (CCEF from src/fusion_model.py)
- intermediate: each modality gets its own projection, then a joint layer over
                [h_text, h_image, h_text * h_image]
- late:         one MLP per modality; the two class-probability vectors are averaged
                with equal weight (no weight is tuned, so no validation leakage)

All heads use the same training rule as CCEF: class-weighted cross-entropy, AdamW,
early stopping on validation macro-F1, and standardisation fitted on the train split only.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.fusion_model import predict_proba, standardize, train_ccef


class IntermediateHead(nn.Module):
    def __init__(self, d_text: int, d_image: int, n_classes: int = 3, hidden: int = 128):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(d_text, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.image_proj = nn.Sequential(nn.Linear(d_image, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.joint = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, n_classes),
        )

    def forward(self, x_text, x_image):
        h_t = self.text_proj(x_text)
        h_i = self.image_proj(x_image)
        return self.joint(torch.cat([h_t, h_i, h_t * h_i], dim=1))


@torch.no_grad()
def predict_intermediate(model, Xt, Xi, device="cpu") -> np.ndarray:
    model.eval()
    logits = model(
        torch.tensor(Xt, dtype=torch.float32, device=device),
        torch.tensor(Xi, dtype=torch.float32, device=device),
    )
    return torch.softmax(logits, dim=1).cpu().numpy()


def train_intermediate(Xt_tr, Xi_tr, y_tr, Xt_va, Xi_va, y_va, n_classes=3, epochs=30, lr=1e-4,
                       batch_size=64, patience=5, seed=42, device="cpu"):
    """Same rule as train_ccef: keep the epoch with the best validation macro-F1."""
    torch.manual_seed(seed)
    model = IntermediateHead(Xt_tr.shape[1], Xi_tr.shape[1], n_classes).to(device)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(
        len(y_tr) / (n_classes * np.maximum(counts, 1.0)), dtype=torch.float32, device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xt_tr, dtype=torch.float32),
                      torch.tensor(Xi_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True,
    )

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xt, xi, yb in loader:
            xt, xi, yb = xt.to(device), xi.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xt, xi), yb)
            loss.backward()
            opt.step()
        val_pred = predict_intermediate(model, Xt_va, Xi_va, device).argmax(axis=1)
        f1 = f1_score(y_va, val_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, float(best_f1)


def fit_and_predict(kind, blocks, y, split, seed=42, device="cpu", epochs=30, lr=1e-4, patience=5):
    """Train one configuration and return test-set and validation-set probabilities.

    blocks: {"text": (N, d_t) or None, "image": (N, d_i) or None, "extra": (N, k) or None}
    split:  {"train": mask, "val": mask, "test": mask} (boolean arrays of length N)
    Only the train rows are used to fit the standardisation and the weights.
    """
    tr, va, te = split["train"], split["val"], split["test"]
    y = np.asarray(y, dtype=int)

    def std_block(block):
        if block is None:
            return None
        (tr_s, va_s, te_s), _, _ = standardize(block[tr], block[va], block[te])
        full = np.zeros((len(block), block.shape[1]), dtype=np.float32)
        full[tr], full[va], full[te] = tr_s, va_s, te_s
        return full

    if kind == "early":
        parts = [b for b in [blocks.get("text"), blocks.get("image"), blocks.get("extra")] if b is not None]
        X = np.concatenate([std_block(b) for b in parts], axis=1).astype(np.float32)
        model, val_f1 = train_ccef(X[tr], y[tr], X[va], y[va], epochs=epochs, lr=lr,
                                   patience=patience, seed=seed, device=device)
        return {"val": predict_proba(model, X[va], device), "test": predict_proba(model, X[te], device),
                "val_macro_f1": val_f1, "model": model}

    if kind == "intermediate":
        Xt, Xi = std_block(blocks["text"]), std_block(blocks["image"])
        model, val_f1 = train_intermediate(Xt[tr], Xi[tr], y[tr], Xt[va], Xi[va], y[va],
                                           epochs=epochs, lr=lr, patience=patience, seed=seed, device=device)
        return {"val": predict_intermediate(model, Xt[va], Xi[va], device),
                "test": predict_intermediate(model, Xt[te], Xi[te], device),
                "val_macro_f1": val_f1, "model": model}

    if kind == "late":
        outs, models = {"val": [], "test": []}, []
        val_f1s = []
        for name in ["text", "image"]:
            X = std_block(blocks[name])
            model, val_f1 = train_ccef(X[tr], y[tr], X[va], y[va], epochs=epochs, lr=lr,
                                       patience=patience, seed=seed, device=device)
            outs["val"].append(predict_proba(model, X[va], device))
            outs["test"].append(predict_proba(model, X[te], device))
            models.append(model)
            val_f1s.append(val_f1)
        probs_val = np.mean(outs["val"], axis=0)
        probs_test = np.mean(outs["test"], axis=0)
        combined_val_f1 = f1_score(y[va], probs_val.argmax(axis=1), labels=[0, 1, 2],
                                   average="macro", zero_division=0)
        return {"val": probs_val, "test": probs_test, "val_macro_f1": float(combined_val_f1),
                "model": models}

    raise ValueError(f"unknown fusion kind {kind!r}")
