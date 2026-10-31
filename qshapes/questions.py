"""Question set format (JSON): a list of questions.

Each question has an `id`, a `shape` (lookup, similarity or relationship), the natural-language `text`,
optional `seed` (a document id the question refers to), optional `narrow` (true when the answer set is
small enough for recall to mean something), and ground truth in one of three forms:
  `groups`: the correct group ids;
  `judged`: none up front, marked by a person after the runs (similarity);
  `check`:  the name of a validity check for questions whose answer is "any valid set".
"""
import json


def load_questions(path):
    with open(path) as f:
        return json.load(f)
