"""
src/vectorstore/store.py — Qdrant-based local vector store

Persists embeddings to data/vectorstore/ (no server required — local file mode).
Supports three collections:
  - "transactions"  : all 1226 raw transactions
  - "summaries"     : monthly income/spend summaries
  - "anomalies"     : flagged anomaly verdicts

Schema per document:
  {
    "id":       int,
    "text":     str,    # the embedded string
    "payload":  dict,   # arbitrary metadata (date, category, amount, ...)
  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DATA_DIR
from src.vectorstore.embedder import EmbedBackend, embed_texts, embed_query

log = logging.getLogger(__name__)

VECTORSTORE_DIR = DATA_DIR / "vectorstore"
STATE_FILE = VECTORSTORE_DIR / "store_state.json"

# Collection names
COLL_TRANSACTIONS = "transactions"
COLL_SUMMARIES    = "summaries"
COLL_ANOMALIES    = "anomalies"

_COLLECTIONS = [COLL_TRANSACTIONS, COLL_SUMMARIES, COLL_ANOMALIES]


# ── In-memory + numpy store ────────────────────────────────────────────────────

class _VectorStore:
    """
    Vector store with Qdrant (local file) as primary and numpy cosine sim as fallback.
    All numpy vectors are saved as .npy files; metadata as JSON — no pickle used.
    """

    def __init__(self):
        self._client = None
        self._numpy_store: dict[str, dict] = {c: {"vecs": None, "payloads": [], "texts": []} for c in _COLLECTIONS}
        self._backend: EmbedBackend | None = None
        self._use_qdrant = False
        self._loaded = False

    def _init_qdrant(self) -> bool:
        try:
            from qdrant_client import QdrantClient
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(VECTORSTORE_DIR / "qdrant"))
            self._use_qdrant = True
            log.info("Qdrant vector store initialised at %s", VECTORSTORE_DIR / "qdrant")
            return True
        except Exception as exc:
            log.warning("Qdrant unavailable (%s) — using numpy fallback", exc)
            return False

    def _ensure_collection(self, name: str, dim: int):
        if not self._use_qdrant:
            return
        from qdrant_client.models import Distance, VectorParams
        existing = [c.name for c in self._client.get_collections().collections]
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            log.info("Created Qdrant collection '%s' (dim=%d)", name, dim)

    def index(self, collection: str, texts: list[str], payloads: list[dict], backend: EmbedBackend | None = None):
        """
        Embed and index a list of texts into the named collection.

        Args:
            collection: One of the _COLLECTIONS constants.
            texts:      List of strings to embed.
            payloads:   Metadata dicts (one per text) stored alongside vectors.
            backend:    Force embedding backend; auto-detect if None.
        """
        if not texts:
            log.warning("index() called with empty texts for '%s'", collection)
            return

        log.info("Indexing %d documents into '%s'...", len(texts), collection)
        vecs, used_backend = embed_texts(texts, backend=backend)
        self._backend = used_backend
        log.info("Embedded with backend='%s'", used_backend)

        dim = len(vecs[0])
        vecs_np = np.array(vecs, dtype=np.float32)

        # Always store in numpy for fast local fallback
        self._numpy_store[collection]["vecs"]     = vecs_np
        self._numpy_store[collection]["texts"]    = texts
        self._numpy_store[collection]["payloads"] = payloads
        self._save_collection(collection, vecs_np, texts, payloads)

        if self._use_qdrant:
            from qdrant_client.models import PointStruct
            self._ensure_collection(collection, dim)
            # Re-create collection to replace existing data
            try:
                info = self._client.get_collection(collection)
                if info.points_count > 0:
                    self._client.delete_collection(collection)
                    self._ensure_collection(collection, dim)
            except Exception:
                pass
            points = [
                PointStruct(id=i, vector=v.tolist(), payload={"text": t, **p})
                for i, (v, t, p) in enumerate(zip(vecs_np, texts, payloads))
            ]
            self._client.upsert(collection_name=collection, points=points)
            log.info("Qdrant: upserted %d points into '%s'", len(points), collection)
        else:
            log.info("Numpy: stored %d vectors for '%s'", len(vecs), collection)

    def search(self, collection: str, query: str, top_k: int = 8) -> list[dict]:
        """
        Semantic search for top_k most relevant documents.

        Returns:
            List of dicts: {"text": str, "score": float, **payload}
        """
        if not self._loaded:
            self._load_state()

        if self._backend is None:
            # State file missing but .npy files exist — detect backend and continue
            from src.vectorstore.embedder import detect_backend
            self._backend = detect_backend()
            log.info("Backend auto-detected for search: %s", self._backend)

        q_vec = embed_query(query, self._backend)

        if self._use_qdrant and self._client:
            try:
                hits = self._client.search(
                    collection_name=collection,
                    query_vector=q_vec,
                    limit=top_k,
                )
                return [
                    {"text": h.payload.get("text", ""), "score": h.score,
                     **{k: v for k, v in h.payload.items() if k != "text"}}
                    for h in hits
                ]
            except Exception as exc:
                log.warning("Qdrant search failed (%s) — falling back to numpy", exc)

        # Numpy cosine similarity fallback
        store = self._numpy_store.get(collection, {})
        vecs = store.get("vecs")
        if vecs is None or len(vecs) == 0:
            # Try loading from disk
            self._load_collection(collection)
            vecs = self._numpy_store[collection].get("vecs")
            if vecs is None or len(vecs) == 0:
                return []

        q = np.array(q_vec, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        sims = (vecs / norms) @ q_norm

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        texts    = self._numpy_store[collection].get("texts", [])
        payloads = self._numpy_store[collection].get("payloads", [])
        for i in top_idx:
            payload = payloads[i] if payloads else {}
            results.append({
                "text":  texts[i] if texts else "",
                "score": float(sims[i]),
                **payload,
            })
        return results

    def collection_size(self, collection: str) -> int:
        if self._use_qdrant and self._client:
            try:
                return self._client.get_collection(collection).points_count
            except Exception:
                pass
        vecs = self._numpy_store.get(collection, {}).get("vecs")
        return len(vecs) if vecs is not None else 0

    def is_indexed(self, collection: str) -> bool:
        return self.collection_size(collection) > 0

    # ── Persistence (JSON + .npy — no pickle) ──────────────────────────────────

    def _save_collection(self, name: str, vecs: np.ndarray, texts: list[str], payloads: list[dict]):
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(VECTORSTORE_DIR / f"{name}_vecs.npy", vecs)
        meta = {"texts": texts, "payloads": payloads}
        (VECTORSTORE_DIR / f"{name}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _load_collection(self, name: str):
        vecs_path = VECTORSTORE_DIR / f"{name}_vecs.npy"
        meta_path = VECTORSTORE_DIR / f"{name}_meta.json"
        if not vecs_path.exists() or not meta_path.exists():
            return
        try:
            vecs = np.load(vecs_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._numpy_store[name]["vecs"]     = vecs
            self._numpy_store[name]["texts"]    = meta.get("texts", [])
            self._numpy_store[name]["payloads"] = meta.get("payloads", [])
            log.info("Loaded collection '%s' from disk (%d vectors)", name, len(vecs))
        except Exception as exc:
            log.warning("Could not load collection '%s': %s", name, exc)

    def _save_state(self):
        try:
            VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
            state = {"backend": self._backend, "use_qdrant": self._use_qdrant}
            STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not save vector store state: %s", exc)

    def _load_state(self):
        self._loaded = True
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            self._backend    = state.get("backend")
            self._use_qdrant = state.get("use_qdrant", False)
            # Restore qdrant client if we had it
            if self._use_qdrant and self._client is None:
                self._init_qdrant()
            # Load numpy fallback data
            for c in _COLLECTIONS:
                self._load_collection(c)
            log.info("Vector store state loaded (backend=%s)", self._backend)
        except Exception as exc:
            log.warning("Could not load vector store state: %s", exc)

    def initialise(self):
        """Initialise Qdrant (or numpy) and load persisted state."""
        self._init_qdrant()
        self._load_state()


# ── Module-level singleton ─────────────────────────────────────────────────────

_store: _VectorStore | None = None


def get_store() -> _VectorStore:
    global _store
    if _store is None:
        _store = _VectorStore()
        _store.initialise()
    return _store
