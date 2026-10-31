"""Hybrid: embeddings for closeness, field filters for the hard rules.

Plan: {"like": "text describing what the dish should be like", "filters": [field-lookup conditions],
       "exclude_seed": "<document id>"}
Documents are ranked by similarity to `like` (the question text when `like` is empty), and only those
passing every filter are kept. It shares the vector store of the Embeddings method (in memory or Postgres).
"""
from .base import Method, Result
from .field_lookup import _match


class Hybrid(Method):
    name = 'hybrid'

    def __init__(self, embeddings):
        self.emb = embeddings

    def setup(self, corpus):
        self.by_id = {d.id: d for d in corpus}
        return {'documents': len(corpus), 'shares': self.emb.name}

    def retrieve(self, question, plan, k=10):
        like = plan.get('like') or question['text']
        filters, skip = plan.get('filters', []), plan.get('exclude_seed')
        out = []
        for doc_id in self.emb.ordered(like):
            d = self.by_id[doc_id]
            if d.id != skip and all(_match(d, c) for c in filters):
                out.append(d.id)
                if len(out) >= max(k * 5, 50):
                    break
        return Result(out)
