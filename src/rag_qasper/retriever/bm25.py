"""BM25 from scratch: tokenizer, document frequencies, IDF, scoring (no rank_bm25)."""
import math
import re
from collections import Counter, defaultdict

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_PLACEHOLDER = re.compile(r"^(bibref|tabref|figref|secref|floatref)\d*$")   # QASPER citation/table markers


def tokenize(text):
    """Lowercase alphanumeric tokens; QASPER placeholders like BIBREF19 / TABREF3 are dropped."""
    return [t for t in _TOKEN.findall(text.lower()) if not _PLACEHOLDER.match(t)]


class BM25:
    """In-memory BM25 Okapi retriever built from scratch.

    Parameters
    ----------
    texts : Iterable[str]
        Corpus documents to index.
    k1 : float, default=1.5
        Term frequency saturation parameter.
    b : float, default=0.75
        Document length normalization parameter (0 = no length penalty, 1 = full penalty).
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b, self.N = k1, b, len(texts)
        # Inverted index mapping: term -> list of (doc_index, term_frequency)
        self.postings = defaultdict(list)
        self.doc_len = np.zeros(self.N)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if self.N else 1.0
        # Lucene-style smoothed IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        # Ensures non-negative weights for high-frequency terms appearing across most documents.
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query):
        s = np.zeros(self.N)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                s[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return s

    def search(self, query, top_k=10):
        s = self.scores(query)
        order = np.lexsort((np.arange(self.N), -s))[:top_k]       # ties -> lower index first
        return [(int(i), float(s[i])) for i in order if s[i] > 0]

