"""Backend-safe adapters for FurnaceMind memory search tools."""

from __future__ import annotations

from typing import Any

_shift_store: Any = None
_knowledge_store: Any = None


def configure_memory_stores(*, shift_store: Any = None, knowledge_store: Any = None) -> None:
    """Configure stores used by memory search tools."""
    global _shift_store, _knowledge_store
    _shift_store = shift_store
    _knowledge_store = knowledge_store


def search_shift_history(query: str) -> str:
    """Search past shift summaries using the configured vector store."""
    if _shift_store is None:
        return "Shift store not initialized."

    results = _shift_store.search_similar_windows(query_text=query, top_k=5)
    if not results:
        return "No shift summaries found for this query."

    parts = []
    for idx, result in enumerate(results, 1):
        payload = result.get("payload", {})
        text = payload.get("summary_text", "No summary.")
        window_id = payload.get("window_id", "unknown")
        parts.append(f"[{idx}] Shift: {window_id}\n{text}")
    return "\n\n".join(parts)


def search_knowledge_docs(query: str) -> str:
    """Search uploaded knowledge documents using the configured vector store."""
    if _knowledge_store is None:
        return "Knowledge store not initialized."

    results = _knowledge_store.search(query, top_k=5)
    if not results:
        return "No knowledge documents found for this query."

    parts = []
    for idx, result in enumerate(results, 1):
        payload = result.get("payload", {})
        content = payload.get("content", "No content.")
        source = payload.get("source", "unknown")
        parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts)
