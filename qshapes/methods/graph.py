"""Graph search over document-ingredient links, held as plain tables (SQLite in memory, or Postgres via pg.py).

Plan: {"rule": ..., ...}. Rules:
  shares_with_seed: {"seed": id, "min_shared": 3, "fresh_only": true, "protein_in"/"protein_not_in": [...]}
  uses_all:         {"items": ["feta", "kale"], "exclude_seed": id, "protein_in"/"protein_not_in": [...]}
  basket:           {"items": [...], "min_used": 2, "max_extra": 2}
Results are ranked by how many ingredients they share or use, most first.
"""
import sqlite3
from collections import defaultdict
from .base import Method, Result


class Graph(Method):
    name = 'graph'
    # Table names and the parameter placeholder, so the same queries run on SQLite and Postgres.
    DOC, LINK, P, TRUE = 'doc', 'link', '?', '1'

    def setup(self, corpus):
        self.db = sqlite3.connect(':memory:')
        c = self.db
        c.execute('create table doc (id text primary key, grp text, protein text)')
        c.execute('create table link (doc text, item text, fresh integer)')
        for d in corpus:
            c.execute('insert into doc values (?,?,?)', (d.id, d.group, '|'.join(d.fields.get('protein', []))))
            fresh = set(d.fields.get('fresh', []))
            c.executemany('insert into link values (?,?,?)', [(d.id, i, int(i in fresh)) for i in set(d.ingredients)])
        c.execute('create index li on link(item)')
        c.execute('create index ld on link(doc)')
        n = c.execute('select count(*) from link').fetchone()[0]
        return {'documents': len(corpus), 'links': n}

    def q(self, sql, params=()):
        sql = sql.format(doc=self.DOC, link=self.LINK, p=self.P, true=self.TRUE)
        return self.db.execute(sql, params).fetchall()

    def _proteins(self):
        if not hasattr(self, '_prot'):
            self._prot = {i: set(filter(None, (p or '').split('|'))) for i, p in self.q('select id, protein from {doc}')}
        return self._prot

    def _protein_ok(self, doc_id, plan):
        p = self._proteins().get(doc_id, set())
        if 'protein_in' in plan and not any(any(w in x for x in p) for w in plan['protein_in']):
            return False
        if 'protein_not_in' in plan and any(any(w in x for x in p) for w in plan['protein_not_in']):
            return False
        return True

    def retrieve(self, question, plan, k=10):
        rule = plan.get('rule')
        if rule == 'shares_with_seed':
            fresh = 'and a.fresh={true} and b.fresh={true}' if plan.get('fresh_only', True) else ''
            rows = self.q('select b.doc, count(*) from {link} a join {link} b on a.item=b.item '
                          'where a.doc={p} and b.doc<>{p} ' + fresh + ' group by b.doc having count(*)>={p} '
                          'order by count(*) desc, b.doc', (plan['seed'], plan['seed'], plan.get('min_shared', 1)))
        elif rule == 'uses_all':
            items = [i.lower() for i in plan['items']]
            cond = ' and '.join(['exists (select 1 from {link} l where l.doc=d.id and l.item like {p})'] * len(items)) or '1=1'
            rows = self.q('select d.id, 0 from {doc} d where ' + cond + ' order by d.id', [f'%{i}%' for i in items])
            rows = [r for r in rows if r[0] != plan.get('exclude_seed')]
        elif rule == 'basket':
            items = [i.lower() for i in plan['items']]
            fresh = defaultdict(list)
            for doc, item in self.q('select doc, item from {link} where fresh={true}'):
                fresh[doc].append(item)
            rows = []
            for (doc,) in self.q('select id from {doc}'):
                f = fresh.get(doc, [])
                used = sum(any(i in x for x in f) for i in items)
                extra = sum(not any(i in x for i in items) for x in f)
                if used >= plan.get('min_used', 1) and extra <= plan.get('max_extra', 2):
                    rows.append((doc, used * 10 - extra))
            rows.sort(key=lambda r: (-r[1], r[0]))
        else:
            return Result([], notes={'error': f'unknown rule {rule}'})
        return Result([d for d, _ in rows if self._protein_ok(d, plan)])
