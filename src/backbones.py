"""Frozen backbone embeddings for the model-selection grid.

Text: attention-masked mean of the last hidden layer (the same pooling as the BanglaBERT features).
Image: pooled output of a vision encoder. A missing image gives a zero vector, as in CLIP.

Frozen embeddings give every backbone the same protocol and make the grid cheap. They are not
fine-tuned, so a backbone's fine-tuned score can differ from its frozen score; the report says so.
torch and transformers are imported only when an extraction function is called.
"""
import numpy as np


def masked_mean_pool(last_hidden, attention_mask):
    """last_hidden: (B, T, D); attention_mask: (B, T). Returns (B, D), ignoring padding."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def text_embeddings(model_name, texts, max_len=128, batch_size=32, device="cpu") -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    texts = [str(t) for t in texts]
    chunks = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            chunks.append(masked_mean_pool(hidden, enc["attention_mask"]).float().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def image_embeddings(model_name, image_paths, batch_size=32, device="cpu") -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    valid = [i for i, p in enumerate(image_paths) if isinstance(p, str)]
    if not valid:
        raise ValueError("no image paths to encode")

    out = None
    with torch.no_grad():
        for s in range(0, len(valid), batch_size):
            idx = valid[s:s + batch_size]
            imgs = []
            for i in idx:
                with Image.open(image_paths[i]) as im:
                    imgs.append(im.convert("RGB"))
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            result = model(**inputs)
            pooled = result.pooler_output if result.pooler_output is not None else result.last_hidden_state[:, 0]
            pooled = pooled.reshape(len(idx), -1).float().cpu().numpy()
            if out is None:
                out = np.zeros((len(image_paths), pooled.shape[1]), dtype=np.float32)
            out[idx] = pooled
    return out

