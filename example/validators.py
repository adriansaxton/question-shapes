"""Validity checks for questions whose answer is "any valid set" (see `check` in qshapes/questions.py).

`load(corpus)` returns {check name: function}. Each function gets the top group ids and returns True or False.
"""


def load(corpus):
    fresh = {}
    for d in corpus:
        fresh.setdefault(d.group, set()).update(d.fields.get('fresh', []))

    def shares_fresh(groups):
        # Valid when there are three dinners and each shares at least two fresh ingredients with another of them.
        if len(groups) < 3:
            return False
        return all(any(len(fresh[g] & fresh[h]) >= 2 for h in groups if h != g) for g in groups)

    return {'shares_fresh': shares_fresh}
