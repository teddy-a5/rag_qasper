from .bm25 import BM25
from .dense import DenseRetriever


class HybridRetriever:
    """Dense + BM25 over the same chunks, merged with Reciprocal Rank Fusion (ranks start at 1)."""

    def __init__(self, chunks, embedder=None, pool=100, rrf_k=60):
        self.dense = DenseRetriever(chunks, embedder=embedder)
        self.sparse = BM25([c.get("index_text", c["text"]) for c in chunks])
        self.pool, self.rrf_k = pool, rrf_k

    def search(self, query, top_k=10, query_vec=None):
        if not query.strip() or top_k <= 0:
            return []
        scores = {}
        for hits in (self.dense.search(query, self.pool, query_vec), self.sparse.search(query, self.pool)):
            for rank, (i, _) in enumerate(hits, start=1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

