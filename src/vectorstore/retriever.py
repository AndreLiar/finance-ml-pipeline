"""
src/vectorstore/retriever.py — High-level semantic search interface

Provides `semantic_search(query, collection, top_k)` for agents to call.
Also provides `build_context(query, top_k)` which searches ALL collections
and formats results as a concise natural-language context block.
"""

from __future__ import annotations

import logging
from typing import Literal

from src.vectorstore.store import (
    get_store,
    COLL_TRANSACTIONS, COLL_SUMMARIES, COLL_ANOMALIES,
)

log = logging.getLogger(__name__)

Collection = Literal["transactions", "summaries", "anomalies", "all"]


def semantic_search(
    query: str,
    collection: Collection = "all",
    top_k: int = 8,
) -> list[dict]:
    """
    Search the vector store for the most relevant documents.

    Args:
        query:      Natural language search query.
        collection: Which collection to search ("all" searches all three).
        top_k:      Number of results to return per collection.

    Returns:
        List of result dicts sorted by score (highest first).
        Each dict has at minimum: {"text": str, "score": float}.
    """
    store = get_store()
    results = []

    if collection == "all":
        for coll in [COLL_TRANSACTIONS, COLL_SUMMARIES, COLL_ANOMALIES]:
            if store.is_indexed(coll):
                hits = store.search(coll, query, top_k=top_k // 2 or 4)
                for h in hits:
                    h["_collection"] = coll
                results.extend(hits)
        # Re-rank by score across collections
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
    else:
        results = store.search(collection, query, top_k=top_k)
        for r in results:
            r["_collection"] = collection

    log.debug("semantic_search('%s', %s) -> %d results", query[:60], collection, len(results))
    return results


def build_context(query: str, top_k: int = 8) -> str:
    """
    Build a concise context block for the LLM by searching all collections.

    Returns a formatted string ready to inject into a prompt.
    Returns empty string if the store is not indexed.
    """
    results = semantic_search(query, collection="all", top_k=top_k)
    if not results:
        return ""

    lines = [f"=== Relevant financial data (semantic search: '{query[:60]}') ==="]
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        coll  = r.get("_collection", "?")
        text  = r.get("text", "").strip()
        lines.append(f"[{i}] ({coll}, score={score:.2f}) {text}")

    return "\n".join(lines)


def store_status() -> dict:
    """Return indexing status for all collections."""
    store = get_store()
    return {
        c: {"indexed": store.is_indexed(c), "size": store.collection_size(c)}
        for c in [COLL_TRANSACTIONS, COLL_SUMMARIES, COLL_ANOMALIES]
    }
