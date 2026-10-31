"""A method turns a question into a ranked list of document ids.

Methods that need a structured query (a filter, a seed and a rule) read it from a plan: the plan is what
the calling model would pass as tool arguments. Plans are produced once per question, stored, and
reused, so every run of a method sees the same input and the cost of producing them is recorded once.
"""
import time
from dataclasses import dataclass, field


@dataclass
class Result:
    doc_ids: list
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)


class Method:
    name = 'base'

    def setup(self, corpus):
        """Build whatever the method needs. Returns a dict of setup facts (time, sizes)."""
        return {}

    def retrieve(self, question, plan, k=10):
        raise NotImplementedError

    def timed(self, question, plan, k=10):
        t = time.perf_counter()
        try:
            r = self.retrieve(question, plan, k)
        except (KeyError, TypeError, ValueError) as e:
            # A plan the tool cannot execute counts as an empty answer, recorded, not a crash.
            r = Result([], notes={'invalid_plan': repr(e)})
        r.seconds = time.perf_counter() - t
        return r
