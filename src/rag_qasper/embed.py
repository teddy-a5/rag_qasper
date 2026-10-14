"""BGE embeddings with on-disk caching (documents and queries)."""
import hashlib
import pickle
import re

import numpy as np

from .config import CACHE_DIR, EMBED_MODEL, QUERY_PREFIX


def _digest(parts):
    h = hashlib.sha256()
    for s in parts:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class Embedder:
    """Document and query embedder with disk caching and L2 normalization."""

    def __init__(self, model_name=EMBED_MODEL, device="cpu", cache_dir=CACHE_DIR, use_query_cache=True):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_query_cache = use_query_cache
        self.cache_dir.mkdir(exist_ok=True)
        self._model = None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
        self._qpath = self.cache_dir / f"queries_{safe}.pkl"
        self._qcache = pickle.loads(self._qpath.read_bytes()) if self._qpath.exists() else {}
        self._dirty = False

    @property
    def model(self):
        if self._model is None:                    
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts, batch_size=64, progress=False):
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=progress)
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts, batch_size=64):
        """Matrix (n, dim). Cached on disk, keyed by model name + exact chunk texts."""
        path = self.cache_dir / f"docs_{_digest([self.model_name, *texts])}.npy"
        if path.exists():
            return np.load(path)
        vecs = self._encode(list(texts), batch_size=batch_size, progress=True)
        np.save(path, vecs)
        return vecs

    def embed_queries(self, texts, use_prefix=True, batch_size=64):
        """Matrix (n, dim). Per-query disk cache; only uncached queries hit the model."""
        prefix = QUERY_PREFIX if use_prefix else ""
        if not self.use_query_cache:
            return self._encode([prefix + t for t in texts], batch_size=batch_size)
        keys = [_digest([self.model_name, prefix + t]) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._qcache]
        if missing:
            vecs = self._encode([prefix + texts[i] for i in missing], batch_size=batch_size)
            for i, v in zip(missing, vecs):
                self._qcache[keys[i]] = v
            self._dirty = True
        return np.stack([self._qcache[k] for k in keys])

    def embed_query(self, text, use_prefix=True):
        return self.embed_queries([text], use_prefix=use_prefix)[0]

    def save(self):
        if self._dirty:
            self._qpath.write_bytes(pickle.dumps(self._qcache))
            self._dirty = False

