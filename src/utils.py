"""Common helpers: random seed, device selection and JSON saving."""
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # allow the pure-logic part to run without torch
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
