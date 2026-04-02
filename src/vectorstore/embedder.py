"""
src/vectorstore/embedder.py — Embedding backend with layered fallbacks

Priority order:
  1. Ollama nomic-embed-text  (best quality, local, privacy-safe)
  2. sentence-transformers    (all-MiniLM-L6-v2, no network after first download)
  3. TF-IDF bag-of-words      (offline fallback, no ML model required)

All backends return List[float] of consistent dimensionality per backend.
The store tracks which backend was used so retrieval uses the same one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import requests

log = logging.getLogger(__name__)

EmbedBackend = Literal["nomic", "minilm", "tfidf"]

NOMIC_MODEL  = "nomic-embed-text"
MINILM_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL   = "http://localhost:11434/api/embeddings"
OLLAMA_TIMEOUT = 5.0


# ── Ollama nomic-embed-text ────────────────────────────────────────────────────

def _nomic_available() -> bool:
    """Check if Ollama is up and nomic-embed-text is loaded."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any("nomic-embed-text" in m for m in models)
    except Exception:
        return False


def _nomic_embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Ollama nomic-embed-text."""
    embeddings = []
    for text in texts:
        r = requests.post(
            OLLAMA_URL,
            json={"model": NOMIC_MODEL, "prompt": text},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        embeddings.append(r.json()["embedding"])
    return embeddings


# ── sentence-transformers (MiniLM) ─────────────────────────────────────────────

_minilm_model = None

def _minilm_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _get_minilm():
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer
        _minilm_model = SentenceTransformer(MINILM_MODEL)
        log.info("Loaded sentence-transformers model: %s", MINILM_MODEL)
    return _minilm_model


def _minilm_embed(texts: list[str]) -> list[list[float]]:
    model = _get_minilm()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vecs.tolist()


# ── TF-IDF fallback ────────────────────────────────────────────────────────────

_tfidf_vectorizer = None
_tfidf_dim = 512


def _get_tfidf():
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        _tfidf_vectorizer = TfidfVectorizer(
            max_features=_tfidf_dim,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        # Fit lazily on first batch
    return _tfidf_vectorizer


def _tfidf_embed(texts: list[str]) -> list[list[float]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = _get_tfidf()
    if not hasattr(vec, "vocabulary_"):
        # Not fitted yet — fit on this batch
        mat = vec.fit_transform(texts)
    else:
        mat = vec.transform(texts)
    # Pad / truncate to _tfidf_dim
    arr = mat.toarray()
    if arr.shape[1] < _tfidf_dim:
        arr = np.pad(arr, ((0, 0), (0, _tfidf_dim - arr.shape[1])))
    return arr[:, :_tfidf_dim].tolist()


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_backend() -> EmbedBackend:
    """Return the best available embedding backend."""
    if _nomic_available():
        log.info("Embedding backend: nomic-embed-text (Ollama)")
        return "nomic"
    if _minilm_available():
        log.info("Embedding backend: sentence-transformers (MiniLM)")
        return "minilm"
    log.warning("Embedding backend: TF-IDF (offline fallback)")
    return "tfidf"


def embed_texts(texts: list[str], backend: EmbedBackend | None = None) -> tuple[list[list[float]], EmbedBackend]:
    """
    Embed a list of texts.

    Args:
        texts:   List of strings to embed.
        backend: Force a specific backend; auto-detect if None.

    Returns:
        (embeddings, backend_used) — embeddings is List[List[float]].
    """
    if not texts:
        return [], backend or "tfidf"

    chosen = backend or detect_backend()

    try:
        if chosen == "nomic":
            return _nomic_embed(texts), "nomic"
        elif chosen == "minilm":
            return _minilm_embed(texts), "minilm"
        else:
            return _tfidf_embed(texts), "tfidf"
    except Exception as exc:
        log.warning("Backend '%s' failed (%s) — falling back", chosen, exc)
        # Cascade fallback
        if chosen == "nomic":
            try:
                return _minilm_embed(texts), "minilm"
            except Exception:
                pass
        return _tfidf_embed(texts), "tfidf"


def embed_query(query: str, backend: EmbedBackend) -> list[float]:
    """Embed a single query string with the given backend."""
    vecs, _ = embed_texts([query], backend=backend)
    return vecs[0]
