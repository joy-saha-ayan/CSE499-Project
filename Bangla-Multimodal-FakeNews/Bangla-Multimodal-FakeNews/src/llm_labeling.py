"""LLM labelling: cultural silver labels and Wikipedia search queries.

Rules that every prompt in this file follows:
- The prompt contains the post text only. The class label (Authentic / Misleading / Satire)
  is never sent to the model, so the model cannot copy the label.
- Answers are parsed strictly. A malformed answer raises an error; it is never guessed.
- Every answer is cached per id in a JSONL file, so an interrupted run resumes where it stopped.

Pure helpers (prompts, parsing, cache, agreement, sampling) run without torch or anthropic.
LocalLLMClient imports torch and transformers, and LLMClient imports anthropic, only when created.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

CULTURAL_DEFINITIONS = {
    "sarcasm": "the post uses sarcasm or irony, so the intended meaning differs from the literal words",
    "humor": "the post is meant to be funny (a joke, a pun, meme wording)",
    "local_reference": "the post refers to a Bangladesh-specific place, institution, person, event or everyday local practice",
    "historical_reference": "the post refers to a specific past event, date or historical figure",
}


# ---- prompts (text only, no label) ----------------------------------------------

def _cut(text, max_chars: int) -> str:
    return str(text if text is not None else "")[:max_chars]


def cultural_prompt(text, cols, max_chars: int = 2000) -> str:
    unknown = [c for c in cols if c not in CULTURAL_DEFINITIONS]
    if unknown:
        raise KeyError(f"no definition for cultural feature(s): {unknown}")
    feature_lines = "\n".join(f'- "{c}": {CULTURAL_DEFINITIONS[c]}' for c in cols)
    keys = ", ".join(f'"{c}": 0 or 1' for c in cols)
    return (
        "You label a Bangla social-media post for cultural features.\n"
        "For each feature answer 1 if it is clearly present in the post, otherwise 0.\n\n"
        f"Features:\n{feature_lines}\n\n"
        "Return only one JSON object with exactly these keys: " + "{" + keys + "}\n\n"
        "Post:\n<<<\n" + _cut(text, max_chars) + "\n>>>"
    )


def query_prompt(text, max_chars: int = 2000) -> str:
    return (
        "Read the Bangla social-media post below.\n"
        "Write ONE search query in Bangla, at most 6 words, that would find the Bangla Wikipedia "
        "article about the main factual subject of the post (a person, place, organisation or event).\n"
        "If the post has no checkable factual subject, write an empty string.\n\n"
        'Return only one JSON object: {"query": "..."}\n\n'
        "Post:\n<<<\n" + _cut(text, max_chars) + "\n>>>"
    )


# ---- strict parsing ----------------------------------------------------------------

def parse_json_object(answer: str) -> dict:
    match = re.search(r"\{.*\}", str(answer), flags=re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON object in model answer: {str(answer)[:200]!r}")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model answer: {str(answer)[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise ValueError("model answer is not a JSON object")
    return obj


def parse_cultural_answer(answer: str, cols) -> dict:
    """Return {col: 0 or 1}. Missing keys, other values or non-binary values raise.

    If the JSON is malformed (small models sometimes add a stray character), each key is read
    with a pattern instead. Only true, false, 0 or 1 are accepted as values; anything else raises.
    """
    try:
        obj = parse_json_object(answer)
    except ValueError:
        obj = {}
        for c in cols:
            match = re.search(r'"%s"\s*:\s*(true|false|0|1)\b' % re.escape(c), str(answer))
            if match is not None:
                obj[c] = 1 if match.group(1) in ("true", "1") else 0
    out = {}
    for c in cols:
        if c not in obj:
            raise ValueError(f"model answer has no key '{c}'")
        value = obj[c]
        if isinstance(value, bool):
            out[c] = int(value)
        elif isinstance(value, int) and value in (0, 1):
            out[c] = value
        else:
            raise ValueError(f"'{c}' must be 0 or 1, got {value!r}")
    return out


def parse_query_answer(answer: str) -> str:
    """Return the Bangla search query, or '' when the model found no checkable subject.

    If the JSON is malformed, the value of "query" is read with a pattern. If that fails too,
    the error is raised, so the caller can count it. Nothing is guessed.
    """
    try:
        obj = parse_json_object(answer)
        query = obj.get("query")
    except ValueError:
        match = re.search(r'"query"\s*:\s*"([^"\n]*)"', str(answer))
        if match is None:
            raise
        query = match.group(1)
    if query is None:
        raise ValueError("model answer has no value for 'query'")
    return str(query).strip()[:120]


# ---- retries ----------------------------------------------------------------------

def call_with_retries(fn, max_retries: int = 5, base_delay: float = 2.0, sleep=time.sleep):
    """Call fn(); retry with exponential back-off. After the last attempt the last error is raised."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # transient API / network / parse errors; re-raised below
            last_error = exc
            if attempt < max_retries - 1:
                sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"failed after {max_retries} attempts: {last_error}") from last_error


# ---- resumable cache ----------------------------------------------------------------

def load_cache(path: Path):
    """Return ({key: value}, number_of_unreadable_lines). A line cut by a disconnect is skipped."""
    path = Path(path)
    if not path.exists():
        return {}, 0
    cache, bad = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            cache[rec["key"]] = rec["value"]
        except (json.JSONDecodeError, KeyError, TypeError):
            bad += 1
    return cache, bad


def append_cache(path: Path, key, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def run_cached(items, fn, cache_path: Path, workers: int = 8, log_every: int = 200, on_error=None) -> dict:
    """Compute fn(payload) for every (key, payload) not yet in the cache.

    Results are written to the cache in the main thread as they finish, so a crash loses
    at most the calls in flight. If one call fails after its retries, the error propagates,
    unless on_error(key, exc) is given: then the failure is reported to it, nothing is cached
    for that key (so the next run retries it), and the run continues.
    """
    cache, bad = load_cache(cache_path)
    if bad:
        print(f"[WARN] {bad} unreadable line(s) in {cache_path.name} skipped (interrupted write)")
    pending = [(k, p) for k, p in items if k not in cache]
    print(f"{cache_path.name}: {len(cache)} cached, {len(pending)} to compute")
    if not pending:
        return cache
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, payload): key for key, payload in pending}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                value = fut.result()
            except Exception as exc:
                if on_error is None:
                    raise
                on_error(key, exc)
                done += 1
                continue
            cache[key] = value
            append_cache(cache_path, key, value)
            done += 1
            if done % log_every == 0:
                print(f"  {done}/{len(pending)}")
    return cache


# ---- clients ----------------------------------------------------------------------

class LocalLLMClient:
    """Open-weights instruct model on the local GPU. Free, no API key. Greedy decoding (deterministic)."""

    def __init__(self, model_name: str, max_new_tokens: int = 80):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # device_map loads the weights straight onto the GPU, so the full model is not kept in CPU RAM first.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map={"": self.device}
        ).eval()
        self.max_new_tokens = max_new_tokens

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True)


class LLMClient:
    def __init__(self, model: str, max_tokens: int = 300):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. In Colab, add it under Secrets and export it.")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


# ---- validation of silver labels ------------------------------------------------------

def cohen_kappa_binary(a, b) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("kappa needs two non-empty arrays of the same length")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def agreement_report(silver_df: pd.DataFrame, human_df: pd.DataFrame, cols) -> dict:
    """Agreement between silver (LLM) and human labels on the rows that the human has filled in."""
    silver = silver_df[["sample_id"] + list(cols)].copy()
    silver["sample_id"] = silver["sample_id"].astype(str)
    human = human_df[["sample_id"] + list(cols)].copy()
    human["sample_id"] = human["sample_id"].astype(str)
    merged = human.merge(silver, on="sample_id", how="inner", suffixes=("_human", "_silver"))
    if merged.empty:
        raise ValueError("no overlap between silver labels and the human check file")
    report = {}
    for c in cols:
        pair = merged[[c + "_human", c + "_silver"]].dropna()
        if pair.empty:
            report[c] = {"n": 0, "note": "human has not labelled this feature yet"}
            continue
        h = pair[c + "_human"].astype(int).to_numpy()
        s = pair[c + "_silver"].astype(int).to_numpy()
        report[c] = {
            "n": int(len(pair)),
            "agreement": round(float((h == s).mean()), 4),
            "kappa": round(cohen_kappa_binary(h, s), 4),
            "human_positive_rate": round(float(h.mean()), 4),
            "silver_positive_rate": round(float(s.mean()), 4),
        }
    return report


def stratified_check_sample(df: pd.DataFrame, label_col: str, per_label: dict, seed: int = 42) -> pd.DataFrame:
    """Sample rows per class. The label is used only to choose rows, not written to the output."""
    parts = []
    for label, n in per_label.items():
        pool = df[df[label_col] == label]
        if len(pool) < n:
            raise ValueError(f"class '{label}' has {len(pool)} rows, fewer than the {n} requested")
        parts.append(pool.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_client(max_tokens: int, backend: str, local_model: str, anthropic_model: str,
                local_max_new_tokens: int, anthropic_workers: int):
    """Return (client, workers). The local model runs one prompt at a time on one GPU."""
    if backend == "local":
        return LocalLLMClient(local_model, max_new_tokens=local_max_new_tokens), 1
    if backend == "anthropic":
        return LLMClient(anthropic_model, max_tokens=max_tokens), anthropic_workers
    raise ValueError(f"unknown LLM_BACKEND {backend!r}; use 'local' or 'anthropic'")
