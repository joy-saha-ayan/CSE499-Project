"""CCEF decision head (MLP) and its training loop."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


class CCEF(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def standardize(train, *others):
    """Normalise every split with the mean and std of the train split.

    No information from the other splits is used.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return [(train - mu) / sd] + [(o - mu) / sd for o in others], mu, sd


@torch.no_grad()
def predict_proba(model, X, device="cpu") -> np.ndarray:
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32, device=device))
    return torch.softmax(logits, dim=1).cpu().numpy()


def train_ccef(X_tr, y_tr, X_va, y_va, n_classes=3, epochs=30, lr=1e-4, batch_size=64,
               patience=5, seed=42, device="cpu"):
    """Keep the best epoch by validation macro-F1 (early stopping).

    Inverse-frequency class weights are used to handle class imbalance.
    """
    torch.manual_seed(seed)
    model = CCEF(X_tr.shape[1], n_classes).to(device)
    counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(
        len(y_tr) / (n_classes * np.maximum(counts, 1.0)), dtype=torch.float32, device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size, shuffle=True,
    )

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        val_pred = predict_proba(model, X_va, device).argmax(axis=1)
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
