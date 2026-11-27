"""Step 6: build the evidence database from Bangla Wikipedia (decision D-1).

Pipeline (text only, no class label is used at any step):
1. LLM writes one Bangla search query per post (cached).
2. MediaWiki search returns up to WIKI_SEARCH_LIMIT article ids per query (cached).
3. MediaWiki extracts return the lead section of each article (cached).
4. Leads are split into passages; the output is written to EVIDENCE_CSV in the format
   that script 02 reads (evidence_id, group_id, evidence_text, source, url).
5. A report with coverage numbers, and a blind 40-row relevance check sheet for a human.

Resumable: every LLM and Wikipedia answer is cached in CACHE_DIR, so rerunning after a
disconnect does not repeat finished calls.

Needs ANTHROPIC_API_KEY and internet access to bn.wikipedia.org (run it in Colab).
Run: python scripts/06_build_evidence_wiki.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from configs.config import (  # noqa: E402
    ANNOTATION_DIR,
    CACHE_DIR,
    EVIDENCE_CSV,
    EVIDENCE_RELEVANCE_CHECK_CSV,
    LLM_MAX_INPUT_CHARS,
    LLM_BACKEND,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_WORKERS,
    LOCAL_LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_MODEL,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    WIKI_API,
    WIKI_BATCH,
    WIKI_CHUNK_CHARS,
    WIKI_SEARCH_LIMIT,
)
from src.llm_labeling import (  # noqa: E402
    load_cache,
    make_client,
    call_with_retries,
    parse_query_answer,
    query_prompt,
    run_cached,
)
from src.utils import save_json  # noqa: E402
from src.wiki_evidence import (  # noqa: E402
    build_extract_params,
    build_search_params,
    evidence_rows,
    http_get_json,
    parse_extracts,
    parse_search_ids,
)

CHECK_N = 40


def main():
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "unified_split.csv")
    df["sample_id"] = df["sample_id"].astype(str)
    df["text"] = df["text"].fillna("")

    # 1. LLM search query per post (text only)
    # The model is loaded only when some query is still missing from the cache (saves GPU time and RAM).
    cached_queries, _ = load_cache(CACHE_DIR / "query_cache.jsonl")
    if all(sid in cached_queries for sid in df["sample_id"]):
        client, llm_workers = None, 1
        print("All queries are cached: LLM model not loaded.")
    else:
        client, llm_workers = make_client(
            max_tokens=80, backend=LLM_BACKEND, local_model=LOCAL_LLM_MODEL,
            anthropic_model=LLM_MODEL, local_max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
            anthropic_workers=LLM_WORKERS,
        )

    # Local (greedy) answers are deterministic, so a retry would repeat the same output: one attempt.
    query_retries = 1 if LLM_BACKEND == "local" else LLM_MAX_RETRIES
    query_failures = []  # posts whose answer could not be parsed: they get no search (counted in the report)

    WIKI_TIMEOUT_S = 10  # seconds per request
    WIKI_RETRIES = 3     # attempts per request; a failure is counted, not fatal

    def make_query(text):
        try:
            return call_with_retries(
                lambda: parse_query_answer(client.complete(query_prompt(text, LLM_MAX_INPUT_CHARS))),
                max_retries=query_retries,
            )
        except RuntimeError as exc:
            query_failures.append(str(exc)[:200])
            return ""

    queries = run_cached(
        list(zip(df["sample_id"], df["text"])), make_query,
        CACHE_DIR / "query_cache.jsonl", workers=llm_workers,
    )
    queries = {sid: queries[sid] for sid in df["sample_id"]}
    n_no_query = sum(1 for q in queries.values() if not q)

    # 2. Wikipedia search per unique query (empty query = no search)
    unique_queries = sorted({q for q in queries.values() if q})

    # Sequential requests (one at a time) with a short timeout: the public API is fast, and a stuck
    # parallel request previously held the whole run for over an hour.
    # A failed search is counted and stored as no hits, so one bad request does not stop the run.
    wiki_failures = []  # failed requests are not cached, so the next run retries them

    def search(query):
        return call_with_retries(
            lambda: parse_search_ids(http_get_json(WIKI_API, build_search_params(query, WIKI_SEARCH_LIMIT),
                                                   timeout=WIKI_TIMEOUT_S)),
            max_retries=WIKI_RETRIES,
        )

    def on_search_error(query, exc):
        wiki_failures.append(f"search [{query[:30]}]: {str(exc)[:200]}")

    hits = run_cached([(q, q) for q in unique_queries], search,
                      CACHE_DIR / "wiki_search_cache.jsonl", workers=1, log_every=20,
                      on_error=on_search_error)

    # 3. Wikipedia lead sections for every article found
    all_ids = sorted({pid for ids in hits.values() for pid in ids}, key=int)
    batches = [all_ids[i:i + WIKI_BATCH] for i in range(0, len(all_ids), WIKI_BATCH)]

    def fetch(batch):
        return call_with_retries(
            lambda: parse_extracts(http_get_json(WIKI_API, build_extract_params(batch), timeout=WIKI_TIMEOUT_S)),
            max_retries=WIKI_RETRIES,
        )

    def on_fetch_error(key, exc):
        wiki_failures.append(f"extract [{key[:30]}]: {str(exc)[:200]}")

    batch_results = run_cached([(",".join(b), b) for b in batches], fetch,
                               CACHE_DIR / "wiki_extract_cache.jsonl", workers=1, log_every=20,
                               on_error=on_fetch_error)
    pages = {}
    for result in batch_results.values():
        pages.update(result)

    # 4. Passages -> evidence.csv (format read by script 02)
    ev = evidence_rows(pages, WIKI_CHUNK_CHARS)
    if ev.empty:
        raise RuntimeError("No Wikipedia passages were built. Check the queries and the network.")
    ev.to_csv(EVIDENCE_CSV, index=False)

    # 5. Per-post first hit, coverage report, and blind relevance check sheet
    first_hit = {}
    for sid in df["sample_id"]:
        ids = hits.get(queries[sid], []) if queries[sid] else []
        ids = [p for p in ids if p in pages]
        first_hit[sid] = ids[0] if ids else None
    covered = sum(1 for v in first_hit.values() if v is not None)
    report = {
        "posts": int(len(df)),
        "posts_without_query": int(n_no_query),
        "query_parse_failures": len(query_failures),
        "wiki_request_failures": len(wiki_failures),
        "wiki_failure_examples": wiki_failures[:5],
        "unique_queries": len(unique_queries),
        "posts_with_at_least_one_article": int(covered),
        "coverage_share": round(covered / len(df), 4),
        "articles": len(pages),
        "passages": int(len(ev)),
        "llm_model": LLM_MODEL,
        "evidence_source": "Bangla Wikipedia lead sections; query from post text only",
        "chunk_chars": WIKI_CHUNK_CHARS,
        "search_limit": WIKI_SEARCH_LIMIT,
    }
    save_json(report, RESULTS_DIR / "evidence_build_report.json")
    print("Evidence build:", report)

    if not EVIDENCE_RELEVANCE_CHECK_CSV.exists():
        candidates = df[df["sample_id"].map(lambda s: first_hit[s] is not None)]
        sample = candidates.sample(n=min(CHECK_N, len(candidates)), random_state=SEED)
        sheet = pd.DataFrame({
            "sample_id": sample["sample_id"],
            "text": sample["text"].str.slice(0, 600),
            "query": sample["sample_id"].map(lambda s: queries[s]),
            "wiki_title": sample["sample_id"].map(lambda s: pages[first_hit[s]]["title"]),
            "wiki_url": sample["sample_id"].map(lambda s: f"https://bn.wikipedia.org/?curid={first_hit[s]}"),
            "relevant_0_or_1": "",
        })
        sheet.to_csv(EVIDENCE_RELEVANCE_CHECK_CSV, index=False)
        print(f"Relevance check sheet written ({len(sheet)} rows): {EVIDENCE_RELEVANCE_CHECK_CSV}")


if __name__ == "__main__":
    main()
