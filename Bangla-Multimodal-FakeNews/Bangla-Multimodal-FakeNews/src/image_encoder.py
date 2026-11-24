"""CLIP image and Bangla text embeddings in the same 512-dim space.

Two different models are needed:
- image: sentence-transformers/clip-ViT-B-32 (original CLIP, encodes images)
- Bangla text: sentence-transformers/clip-ViT-B-32-multilingual-v1 (text only, aligned to the
  original CLIP text space)
Passing an image to the multilingual model raises a ValueError.

Embeddings are normalised, so the dot product equals cosine similarity.
A missing image gives a zero vector, and its similarity is 0.
"""
import numpy as np
from PIL import Image


def load_clip(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def encode_images(model, image_paths, batch_size=64) -> np.ndarray:
    dim = model.get_sentence_embedding_dimension()
    out = np.zeros((len(image_paths), dim), dtype=np.float32)
    valid = [i for i, p in enumerate(image_paths) if isinstance(p, str)]
    for s in range(0, len(valid), batch_size):
        idx = valid[s:s + batch_size]
        imgs = [_load_rgb(image_paths[i]) for i in idx]
        out[idx] = model.encode(
            imgs, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
    return out


def encode_texts(model, texts, batch_size=64) -> np.ndarray:
    emb = model.encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True
    )
    return emb.astype(np.float32)


def image_text_similarity(img_emb, txt_emb) -> np.ndarray:
    return (img_emb * txt_emb).sum(axis=1)
