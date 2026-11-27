"""Controls for the two shortcuts found in the progress audit: image presence and text length.

Pure numpy / pandas / scikit-learn. No torch, no network, no label is used to build a control.

- shuffle_images_within_present: keeps which rows have an image, breaks which image belongs to which
  post. If the image model still beats the presence-only control after this, the gain is not only
  "has an image".
- length_quartile_labels: splits rows into length bins, so that results can be read per bin.
- cross_split_near_duplicates: finds posts in a query split whose text is almost the same as a post
  in the reference split (TF-IDF over character n-grams, cosine similarity). This tests the claim
  that the heading-based split leaves no near-copies across splits.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def shuffle_images_within_present(image_feats, has_image, groups=None, seed=42):
    """Return a copy of image_feats where the vectors of rows with an image are permuted.

    has_image: boolean array, True for rows that have an image.
    groups:    optional array (for example the split name). Permutation happens inside each group,
               so each split keeps its own presence count and its own image vectors.
    Rows without an image keep their (zero) vector. The returned permutation maps each output row
    to the source row it received its vector from.
    """
    feats = np.asarray(image_feats)
    mask = np.asarray(has_image, dtype=bool)
    if len(mask) != len(feats):
        raise ValueError("has_image must have one entry per row")
    if groups is None:
        groups = np.zeros(len(feats), dtype=int)
    groups = np.asarray(groups)

    rng = np.random.default_rng(seed)
    perm = np.arange(len(feats))
    for g in np.unique(groups):
        idx = np.where((groups == g) & mask)[0]
        if len(idx) > 1:
            perm[idx] = rng.permutation(idx)
    out = feats[perm].copy()
    out[~mask] = feats[~mask]
    return out, perm


def length_quartile_labels(char_lengths, n_bins=4):
    """Bin index 0..n_bins-1 by quantiles of the character length. Ties may give fewer bins."""
    s = pd.Series(np.asarray(char_lengths, dtype=float))
    return pd.qcut(s, q=n_bins, labels=False, duplicates="drop").to_numpy()


def exact_text_overlap(reference_texts, query_texts):
    """Count query texts that match a reference text exactly after whitespace normalisation."""
    norm = lambda t: " ".join(str(t).split())
    ref = {norm(t) for t in reference_texts if norm(t)}
    return [i for i, t in enumerate(query_texts) if norm(t) and norm(t) in ref]


def cross_split_near_duplicates(reference_texts, query_texts, threshold=0.8,
                                ngram_range=(2, 4), chunk=512):
    """Find (query_index, reference_index, similarity) pairs with cosine similarity >= threshold.

    The vectoriser is fitted on the reference texts only. Similarity is computed on TF-IDF vectors
    of character n-grams, which tolerates small edits such as an added full stop or emoji.
    """
    ref = [str(t) for t in reference_texts]
    qry = [str(t) for t in query_texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
    R = vec.fit_transform(ref)
    Q = vec.transform(qry)

    hits = []
    for start in range(0, Q.shape[0], chunk):
        block = (Q[start:start + chunk] @ R.T).toarray()
        rows, cols = np.where(block >= threshold)
        for r, c in zip(rows, cols):
            hits.append((int(start + r), int(c), float(block[r, c])))
    return hits


def near_duplicate_pair_table(hits, split_name, query_global_idx, ref_global_idx, sample_ids, labels):
    """Turn cross_split_near_duplicates hits into one row per pair, with both labels.

    hits: (query_local_index, reference_local_index, similarity) from cross_split_near_duplicates.
    query_global_idx, ref_global_idx: row numbers in meta for the query and the reference rows.
    sample_ids, labels: the meta columns (arrays indexed by meta row number).
    Labels are only reported, never used to build a control.
    """
    rows = []
    for qi, ci, sim in hits:
        q = int(query_global_idx[qi])
        r = int(ref_global_idx[ci])
        rows.append({
            "split": split_name,
            "query_sample_id": sample_ids[q],
            "ref_sample_id": sample_ids[r],
            "similarity": round(float(sim), 4),
            "query_y": int(labels[q]),
            "ref_y": int(labels[r]),
            "same_label": int(labels[q] == labels[r]),
        })
    return rows

