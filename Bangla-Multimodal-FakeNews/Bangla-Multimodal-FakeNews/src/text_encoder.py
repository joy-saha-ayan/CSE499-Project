"""BanglaBERT: (E1) fine-tuned classifier and frozen embeddings for fusion.

Fusion uses frozen embeddings (pretrained, not fine-tuned). The train-set embeddings of a
fine-tuned model are over-fitted, which would teach the fusion model the wrong thing.
"""
from src.checkpoints import last_complete_checkpoint
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from configs.config import (
    BATCH_SIZE,
    EPOCHS,
    LABEL2ID,
    LABELS_3,
    LR,
    MAX_LEN,
    TEXT_MODEL,
    WEIGHT_DECAY,
)
from src.dataset import TextClassificationDataset


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def finetune_text_classifier(train_df, val_df, test_df, out_dir, seed=42):
    """train/val/test DataFrames must contain 'model_text' and 'y' (0/1/2) columns.

    Returns (predictions, probabilities) for the test rows. Epoch selection uses validation only.

    Resumable: if out_dir already holds a finished 'best' model, only the test prediction runs.
    If it holds an epoch checkpoint, training resumes from the last one.
    """
    out_dir = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    test_ds = TextClassificationDataset(test_df["model_text"], test_df["y"], tok, MAX_LEN)

    best_dir = out_dir / "best"
    if best_dir.exists():
        print(f"Found finished model for seed {seed}; predicting only.")
        best_model = AutoModelForSequenceClassification.from_pretrained(str(best_dir))
        predictor = Trainer(
            model=best_model,
            args=TrainingArguments(
                output_dir=str(out_dir),
                per_device_eval_batch_size=BATCH_SIZE * 2,
                report_to="none",
            ),
        )
        logits = predictor.predict(test_ds).predictions
        return np.argmax(logits, axis=-1), _softmax(logits)

    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_MODEL,
        num_labels=len(LABELS_3),
        id2label=dict(enumerate(LABELS_3)),
        label2id=LABEL2ID,
    )
    train_ds = TextClassificationDataset(train_df["model_text"], train_df["y"], tok, MAX_LEN)
    val_ds = TextClassificationDataset(val_df["model_text"], val_df["y"], tok, MAX_LEN)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, labels=[0, 1, 2], average="macro", zero_division=0),
        }

    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # resume_from_checkpoint=None starts a fresh run when no checkpoint exists yet
    last_checkpoint = last_complete_checkpoint(out_dir)
    if last_checkpoint:
        print(f"Resuming seed {seed} from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(best_dir))
    tok.save_pretrained(str(best_dir))
    logits = trainer.predict(test_ds).predictions
    return np.argmax(logits, axis=-1), _softmax(logits)


@torch.no_grad()
def frozen_text_embeddings(texts, batch_size=64, max_len=MAX_LEN) -> np.ndarray:
    """Mean-pooled embeddings from pretrained BanglaBERT (no fine-tuning)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    texts = list(texts)
    out = []
    for s in range(0, len(texts), batch_size):
        enc = tok(
            texts[s:s + batch_size], truncation=True, max_length=max_len,
            padding=True, return_tensors="pt",
        ).to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out)
