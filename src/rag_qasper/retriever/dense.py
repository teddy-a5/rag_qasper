import numpy as np

from ..embed import Embedder


class DenseRetriever:
    """Brute-force cosine search over a chunk-embedding matrix (fine for a few thousand chunks)."""
    def __init__(self, chunks, embedder=None):
        self.chunks = chunks
        self.embedder = embedder or Embedder()
        self.matrix = self.embedder.embed_documents([c.get("index_text", c["text"]) for c in chunks])  # (n, dim)

    def search(self, query, top_k=10, query_vec=None):
        if not query.strip() or top_k <= 0 or len(self.chunks) == 0:
            return []
        q = query_vec if query_vec is not None else self.embedder.embed_query(query)
        scores = self.matrix @ q                       # cosine similarity (vectors are normalized)
        # sort by score descending, ties broken by chunk index -> deterministic
        order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
        return [(int(i), float(scores[i])) for i in order]

