"""Corpus format: one JSON object per line.

Each document has an `id`, a `group` (documents that are versions of the same thing share a group, and
scoring counts groups, not documents), a `title`, free `text` for embedding, a `fields` dict for field
lookups (numbers or lists of strings), and an `ingredients` list of normalised items used by the graph.
"""
import json
from dataclasses import dataclass, field


@dataclass
class Doc:
    id: str
    group: str
    title: str
    text: str = ''
    fields: dict = field(default_factory=dict)
    ingredients: list = field(default_factory=list)


def load_corpus(path):
    docs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                docs.append(Doc(d['id'], d.get('group', d['id']), d.get('title', ''), d.get('text', ''),
                                d.get('fields', {}), d.get('ingredients', [])))
    return docs
