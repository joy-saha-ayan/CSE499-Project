from src.checkpoints import last_complete_checkpoint, is_complete_checkpoint, REQUIRED_FILES


def _make(root, step, files):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    for name in files:
        (d / name).write_bytes(b"x")
    return d


def test_skips_incomplete_newest_checkpoint(tmp_path):
    complete = ("model.safetensors",) + REQUIRED_FILES
    _make(tmp_path, 228, complete)
    _make(tmp_path, 342, ("model.safetensors",))  # interrupted save: no optimizer/scheduler/state
    assert last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-228")


def test_returns_none_when_nothing_is_complete(tmp_path):
    _make(tmp_path, 342, ("trainer_state.json",))
    assert last_complete_checkpoint(tmp_path) is None


def test_returns_none_for_missing_folder(tmp_path):
    assert last_complete_checkpoint(tmp_path / "absent") is None


def test_accepts_pytorch_bin_weights(tmp_path):
    d = _make(tmp_path, 10, ("pytorch_model.bin",) + REQUIRED_FILES)
    assert is_complete_checkpoint(d)


def test_ignores_non_numeric_folders(tmp_path):
    _make(tmp_path, 5, ("model.safetensors",) + REQUIRED_FILES)
    (tmp_path / "checkpoint-best").mkdir()
    assert last_complete_checkpoint(tmp_path) == str(tmp_path / "checkpoint-5")
