"""Postgres storage for the embeddings, hybrid and graph methods (pgvector for vectors, plain tables for links).

Same answers as the in-memory versions: vector search is an exact cosine scan (no approximate index; at a few
thousand documents a scan is fast), filters are applied in the same order, graph queries are the same SQL.
Tables are (re)loaded when the corpus changes, keyed by a hash of the document texts.
"""
import hashlib
import json
import time
import numpy as np
import psycopg
from .embeddings import Embeddings
from .graph import Graph

SCHEMA = '''
create extension if not exists vector;
create table if not exists qs_meta (k text primary key, v text);
create table if not exists qs_doc (id text primary key, grp text, title text, protein text, fields jsonb, embedding vector({dim}));
create table if not exists qs_link (doc text references qs_doc(id) on delete cascade, item text, fresh boolean);
create index if not exists qs_link_item on qs_link(item);
create index if not exists qs_link_doc on qs_link(doc);
'''


def connect(dsn):
    return psycopg.connect(dsn, autocommit=True)


def load(conn, corpus, encoder, cache=None):
    """Load documents, embeddings and links unless the stored corpus key already matches."""
    texts = [d.text or d.title for d in corpus]
    key = hashlib.sha256('\n'.join(texts).encode()).hexdigest()
    conn.execute(SCHEMA.format(dim=encoder(['dimension probe']).shape[1]))
    row = conn.execute("select v from qs_meta where k='corpus'").fetchone()
    if row and row[0] == key:
        return {'loaded': False, 'documents': len(corpus)}
    t = time.perf_counter()
    emb = Embeddings(encoder, cache=cache)
    emb.setup(corpus)
    with conn.transaction():
        conn.execute('truncate qs_link, qs_doc')
        with conn.cursor().copy('copy qs_doc (id, grp, title, protein, fields, embedding) from stdin') as cp:
            for d, v in zip(corpus, emb.matrix):
                cp.write_row((d.id, d.group, d.title, '|'.join(d.fields.get('protein', [])), json.dumps(d.fields),
                              '[' + ','.join(f'{x:.7f}' for x in v) + ']'))
        with conn.cursor().copy('copy qs_link (doc, item, fresh) from stdin') as cp:
            for d in corpus:
                fresh = set(d.fields.get('fresh', []))
                for i in set(d.ingredients):
                    cp.write_row((d.id, i, i in fresh))
        conn.execute("insert into qs_meta values ('corpus', %s) on conflict (k) do update set v = excluded.v", (key,))
    return {'loaded': True, 'documents': len(corpus), 'load_seconds': round(time.perf_counter() - t, 1)}


class PgEmbeddings(Embeddings):
    """Embeddings with the vectors in Postgres. Retrieval orders by cosine distance in the database."""

    def __init__(self, encoder, conn, query_prefix='', cache=None):
        super().__init__(encoder, query_prefix, cache)
        self.conn = conn

    def setup(self, corpus):
        info = load(self.conn, corpus, self.encoder, self.cache)
        self.ids = [d.id for d in corpus]
        return {**info, 'store': 'postgres'}

    def ordered(self, text, limit=None):
        q = self.encoder([self.query_prefix + text])[0]
        vec = '[' + ','.join(f'{x:.7f}' for x in q) + ']'
        sql = 'select id from qs_doc order by embedding <=> %s::vector, id' + (' limit %s' if limit else '')
        rows = self.conn.execute(sql, (vec, limit) if limit else (vec,)).fetchall()
        return [r[0] for r in rows]

    def retrieve(self, question, plan, k=10):
        from .base import Result
        exclude = set(plan.get('exclude', [])) if plan else set()
        return Result([i for i in self.ordered(question['text'], max(k * 5, 50) + len(exclude)) if i not in exclude][:max(k * 5, 50)])


class PgGraph(Graph):
    """The graph method over the Postgres link table: the same queries as the SQLite version."""
    DOC, LINK, P, TRUE = 'qs_doc', 'qs_link', '%s', 'true'

    def __init__(self, conn):
        self.db = conn

    def setup(self, corpus):
        n = self.db.execute('select count(*) from qs_link').fetchone()[0]
        return {'documents': len(corpus), 'links': n, 'store': 'postgres'}
