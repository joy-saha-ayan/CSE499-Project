"""Bangla Wikipedia as the evidence source (decision D-1).

Evidence is retrieved from the claim text only: an LLM writes a search query from the post
text (src/llm_labeling.py), the MediaWiki API returns matching article leads, and the leads
are split into passages. No class label is used at any step.

Pure helpers (parameters, parsers, chunking, row building) run without network access.
http_get_json needs requests and is only called by scripts/06_build_evidence_wiki.py.
"""
import re
import time

import pandas as pd

USER_AGENT = "BFND-capstone-research/1.0 (NSU CSE499A student project)"


# ---- request parameters -----------------------------------------------------------

def build_search_params(query: str, limit: int) -> dict:
    return {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": int(limit),
        "srnamespace": 0,
        "format": "json",
        "formatversion": 2,
    }


def build_extract_params(pageids) -> dict:
    return {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "pageids": "|".join(str(p) for p in pageids),
        "format": "json",
        "formatversion": 2,
    }


# ---- response parsers ---------------------------------------------------------------

def parse_search_ids(payload: dict) -> list:
    """Return the page ids of the search hits, in rank order (as strings)."""
    hits = (payload.get("query") or {}).get("search") or []
    return [str(h["pageid"]) for h in hits if "pageid" in h]


def parse_extracts(payload: dict) -> dict:
    """Return {pageid: {"title", "extract"}} for pages that have a non-empty extract."""
    pages = (payload.get("query") or {}).get("pages") or []
    out = {}
    for page in pages:
        if "missing" in page or "pageid" not in page:
            continue
        extract = str(page.get("extract") or "").strip()
        if extract:
            out[str(page["pageid"])] = {"title": str(page.get("title", "")), "extract": extract}
    return out


# ---- passages -----------------------------------------------------------------------

def chunk_text(text, max_chars: int = 500) -> list:
    """Split text into passages of at most max_chars, at sentence ends where possible.

    Sentences are split after Bangla danda (।), '.', '!' or '?'. A sentence longer than
    max_chars is cut hard, so every passage respects the limit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = re.sub(r"\s+", " ", str(text if text is not None else "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[।.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks


def evidence_rows(pages: dict, max_chars: int = 500) -> pd.DataFrame:
    """Build evidence.csv rows (evidence_id, group_id, evidence_text, source, url).

    The article title is put in front of every passage, so BM25 can match on the title.
    group_id is the article id, so one article's passages share a group.
    """
    rows = []
    for pid in sorted(pages, key=int):
        page = pages[pid]
        for k, passage in enumerate(chunk_text(page["extract"], max_chars)):
            rows.append({
                "evidence_id": f"W{pid}_{k}",
                "group_id": f"wiki_{pid}",
                "evidence_text": f"{page['title']}. {passage}",
                "source": "Bangla Wikipedia",
                "url": f"https://bn.wikipedia.org/?curid={pid}",
            })
    columns = ["evidence_id", "group_id", "evidence_text", "source", "url"]
    return pd.DataFrame(rows, columns=columns)


# ---- network ------------------------------------------------------------------------

def http_get_json(url: str, params: dict, timeout: float = 30.0, pause: float = 0.1) -> dict:
    import requests

    resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    time.sleep(pause)  # polite pause between requests to the public API
    return resp.json()
