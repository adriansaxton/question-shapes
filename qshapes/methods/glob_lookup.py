"""File-name lookup: match shell-style patterns against document ids (file names).

This is what a planner does when all it can do is list files. Plan: {"globs": ["*chicken-breast*stew*", ...]}.
Matches come back in plan order, then name order, as a file listing would return them.
"""
from fnmatch import fnmatch
from .base import Method, Result


class GlobLookup(Method):
    name = 'glob-lookup'

    def setup(self, corpus):
        self.ids = sorted(d.id for d in corpus)
        return {'documents': len(self.ids)}

    def retrieve(self, question, plan, k=10):
        out, seen = [], set()
        for pattern in plan.get('globs', []):
            for i in self.ids:
                if i not in seen and fnmatch(i, pattern.removesuffix('.md')):
                    seen.add(i)
                    out.append(i)
        return Result(out)
