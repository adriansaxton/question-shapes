"""Embedding search: encode documents once, encode the question, rank by cosine similarity.

The store here is an in-memory matrix with exact search. Swap in pgvector (or any vector store) behind
the same two calls; at a few thousand documents exact search is what an index approximates anyway.
The question is passed as written: no plan, which is the point of this method.
"""
import hashlib
import os
import sys
import time
import numpy as np
from .base import Method, Result


class Embeddings(Method):
    name = 'embeddings'

    def __init__(self, encoder, query_prefix='', cache=None):
        self.encoder = encoder
        self.query_prefix = query_prefix
        self.cache = cache  # optional .npz path; reused when the corpus texts are unchanged

    def setup(self, corpus):
        t = time.perf_counter()
        self.ids = [d.id for d in corpus]
        texts = [d.text or d.title for d in corpus]
        key = hashlib.sha256('\n'.join(texts).encode()).hexdigest()
        cached = False
        if self.cache and os.path.exists(self.cache):
            z = np.load(self.cache)
            if str(z['key']) == key:
                self.matrix, cached = z['matrix'], True
        if not cached:
            self.matrix = self.encoder(texts)
            if self.cache:
                try:
                    np.savez(self.cache, matrix=self.matrix, key=key)
                except OSError as e:  # an unwritable cache costs a re-encode next start, not the service
                    print(f'embeddings: cache not saved ({e})', file=sys.stderr)
        return {'documents': len(self.ids), 'dimensions': int(self.matrix.shape[1]), 'cached': cached,
                'build_seconds': round(time.perf_counter() - t, 1)}

    def ordered(self, text, limit=None):
        """Document ids, most similar to `text` first (ties broken by id, as the Postgres store does)."""
        q = self.encoder([self.query_prefix + text])[0]
        scores = self.matrix @ q
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))
        return [self.ids[i] for i in (order if limit is None else order[:limit])]

    def retrieve(self, question, plan, k=10):
        exclude = set(plan.get('exclude', [])) if plan else set()
        return Result([i for i in self.ordered(question['text']) if i not in exclude][:max(k * 5, 50)])
