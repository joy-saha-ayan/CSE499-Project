"""Cultural context module.

Multi-label logistic regression on the manually annotated subset, over frozen BanglaBERT embeddings.
- Fitting uses only annotated rows from the train split.
- Train-row predictions are out-of-fold (a row's own label is never used to predict that row).
- Validation, test and unannotated rows are predicted by the train-only model.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def fit_cultural_classifiers(emb, df, annot_df, cols, seed=42, max_folds=5):
    """
    emb: (N, d) frozen text embedding for each row of df (same order)
    df: contains sample_id and split columns
    annot_df: sample_id + cols (0/1)
    Returns: probs (N, len(cols)), list of fitted models
    """
    if annot_df["sample_id"].duplicated().any():
        raise ValueError("Annotation CSV has duplicate sample_id values")

    base = df[["sample_id", "split"]].copy()
    base["sample_id"] = base["sample_id"].astype(str)
    annot = annot_df[["sample_id"] + cols].copy()
    annot["sample_id"] = annot["sample_id"].astype(str)
    merged = base.reset_index(drop=True).merge(annot, on="sample_id", how="left")
    if len(merged) != len(df):
        raise ValueError("Row count changed after merge")

    annotated = merged[cols].notna().all(axis=1).to_numpy()
    fit_mask = (merged["split"] == "train").to_numpy() & annotated
    if fit_mask.sum() < 20:
        raise ValueError(
            f"Only {fit_mask.sum()} annotated training samples; hundreds are needed"
        )

    probs = np.zeros((len(df), len(cols)))
    fit_idx = np.where(fit_mask)[0]
    models = []
    for j, col in enumerate(cols):
        y_fit = merged[col].to_numpy()[fit_mask].astype(int)
        counts = np.bincount(y_fit, minlength=2)
        if (counts == 0).any():
            raise ValueError(f"'{col}': train has no samples of one class (counts={counts.tolist()})")
        n_splits = min(max_folds, int(counts.min()))
        if n_splits < 2:
            raise ValueError(f"'{col}': not enough samples for out-of-fold prediction")

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = cross_val_predict(clf, emb[fit_mask], y_fit, cv=cv, method="predict_proba")[:, 1]
        clf.fit(emb[fit_mask], y_fit)
        probs[:, j] = clf.predict_proba(emb)[:, 1]
        probs[fit_idx, j] = oof
        models.append(clf)
    return probs, models
