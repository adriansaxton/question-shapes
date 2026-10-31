"""Field lookup: filter documents on their fields and ingredient lists.

Plan: {"all": [condition, ...], "sort": ["-rating"]}. Conditions:
  {"ingredient": "chicken breast"}           any ingredient contains the text
  {"title_any": ["stew", "curry"]}           title contains any of the words
  {"field": "prep", "op": "<=", "value": 40} numeric comparison
  {"field": "categories", "any": ["Vegetarian"]}  list field contains any value (substring, case-insensitive)
  {"not": condition}
"""
import operator
from .base import Method, Result

OPS = {'<': operator.lt, '<=': operator.le, '>': operator.gt, '>=': operator.ge, '==': operator.eq, '=': operator.eq}


def _match(doc, c):
    if 'not' in c:
        return not _match(doc, c['not'])
    if 'ingredient' in c:
        w = c['ingredient'].lower()
        return any(w in i for i in doc.ingredients)
    if 'title_any' in c:
        t = doc.title.lower()
        return any(w.lower() in t for w in c['title_any'])
    v = doc.fields.get(c['field'])
    if 'any' in c:
        vals = [str(x).lower() for x in (v if isinstance(v, list) else [v] if v is not None else [])]
        return any(a.lower() in x for a in c['any'] for x in vals)
    return v is not None and OPS[c['op']](v, c['value'])


class FieldLookup(Method):
    name = 'field-lookup'

    def setup(self, corpus):
        self.docs = corpus
        return {'documents': len(corpus)}

    def retrieve(self, question, plan, k=10):
        hits = [d for d in self.docs if all(_match(d, c) for c in plan.get('all', []))]
        for key in reversed(plan.get('sort', ['-rating'])):
            desc = key.startswith('-')
            name = key.lstrip('-')
            hits.sort(key=lambda d: (d.fields.get(name) is None, d.fields.get(name) or 0), reverse=desc)
        return Result([d.id for d in hits])
