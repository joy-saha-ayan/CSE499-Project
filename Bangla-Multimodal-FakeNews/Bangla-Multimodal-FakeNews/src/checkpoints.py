"""Pick a resumable Trainer checkpoint. Pure file logic: no torch or transformers import."""
from pathlib import Path

# Files Trainer writes for a complete checkpoint (weights, optimizer, scheduler, RNG state, progress).
REQUIRED_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def is_complete_checkpoint(path: Path) -> bool:
    path = Path(path)
    has_weights = any((path / name).is_file() for name in WEIGHT_FILES)
    return has_weights and all((path / name).is_file() for name in REQUIRED_FILES)


def last_complete_checkpoint(out_dir: Path):
    """Return the newest complete checkpoint-N folder under out_dir, or None.

    An interrupted save (for example a Colab disconnect while writing) leaves a checkpoint folder
    without its files. Such folders are skipped, and the next newest complete one is used.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    candidates = []
    for p in out_dir.glob("checkpoint-*"):
        suffix = p.name.split("-")[-1]
        if p.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), p))
    for step, p in sorted(candidates, reverse=True):
        if is_complete_checkpoint(p):
            return str(p)
        print(f"Skipping incomplete checkpoint {p.name} (missing files, probably an interrupted save)")
    return None
