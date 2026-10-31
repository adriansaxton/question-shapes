# Question Shapes

Question Shapes is a small harness for testing which retrieval method suits which kind of question. The idea is that a question has a shape (a lookup, a similarity match, or a relationship between documents) and that the shape should choose the method, rather than every question going to a vector store because one is there. It was built to test that on 40 labelled questions from a meal-planning agent, over about nine thousand recipe files that group into about six thousand distinct dishes. That data is not included; the `example/` folder holds an invented corpus so the harness runs as is.

## What it tests

Each question has a shape:

- **Lookup:** hard rules. "Vegetarian dinners in 30 minutes or less."
- **Similarity:** closeness in meaning. "Something like this dish, but not fish."
- **Relationship:** links between documents. "Dishes that share at least two fresh ingredients with this one."

The harness runs every question through five retrieval methods and scores the top 10:

| Method | What it does |
|---|---|
| `glob-lookup` | Matches shell-style patterns against file names. What an agent does when all it can do is list files. |
| `field-lookup` | Filters on structured fields (numbers, lists, ingredients). |
| `embeddings` | Ranks by cosine similarity to the question as written. No plan. |
| `graph` | Queries document-ingredient links held as plain tables. |
| `hybrid` | Ranks by similarity to a short description, keeping only documents that pass field filters. |

Every method except `embeddings` needs a query plan: the arguments a calling model would pass to that tool. Plans
are written once per question by a model that sees the tool description and the questions but never the answers,
then stored and reused, so every run of a method gets the same input.

Scoring counts groups, not documents: several versions of the same dish count once. Lookup and relationship
questions are scored against known answers (precision at 10, and recall when the answer set is small). Similarity
questions are left for a person to mark. Questions whose answer is "any valid set" are scored by a check function.

## Results

The figures below are from the private run on the recipe data. Each cell is the share of the top 10 that was correct, averaged over the questions of that shape; similarity was marked blind by one person as good or fine. No single method wins. The hybrid (embeddings for ranking, field filters for the rules) wins lookups and similarity, the graph wins relationships, and embeddings alone and file-name search win nothing. Embeddings alone lost on similarity because a question that excludes something puts the excluded word into the query and pulls those documents closer. Two questions that ask for a set with rules across it (a best pair, a trio) were not answered by any method in one call.

| Method | Lookups (20) | Similarity (10) | Relationships (8 scored) |
|---|---|---|---|
| Hybrid | **0.99** | **0.89** | 0.41 |
| Field lookup | 0.98 | 0.68 | 0.39 |
| Graph | 0.54 | 0.30 | **0.91** |
| Embeddings | 0.68 | 0.65 | 0.14 |
| File-name lookup | 0.79 | 0.50 | 0.44 |

How much to trust these:

- 40 questions and one person's similarity marks. The numbers are directionally right, not precise. Some lookup answers are defined by title words, which penalises plans that filtered on categories instead.
- Plan noise is about ±5 points. A fresh set of hybrid plans, written from identical tool wording, scored 0.94 on lookups against the 0.99 above. One plan set per question is a sample, not a constant.
- Tool descriptions change the scores a lot. With example values only, field lookup scored 0.85 on lookups; listing the corpus's real values for each field took it to 0.98. Tightening the hybrid's description so that every rule became a filter fixed two similarity plans and dropped lookups to 0.84, so it was reverted.

## Run the example

The `example/` folder holds an invented corpus of 21 dishes, 8 questions and hand-written plans, so the harness
runs without any private data.

```sh
pip install -r requirements.txt
hf download BAAI/bge-small-en-v1.5 --cache-dir .models
python -m qshapes.run example/config.yaml
```

The run prints the path of a results file in `example/results/`. Each row is one question and one method, with the
top 10 groups and the score. To run without embeddings (and skip the model download), delete the four embedding
lines from `example/config.yaml`.

To write fresh plans with a model instead of using the hand-written ones (needs the `claude` CLI):

```sh
python -m qshapes.plan example/config.yaml --model sonnet
```

## Use your own data

You need four files and a config. Paths in the config are relative to the config file.

**Corpus** (`corpus.jsonl`): one JSON object per line.

```json
{"id": "cod-with-tomato-and-olive-sauce", "group": "G05", "title": "Cod with tomato and olive sauce",
 "text": "Text to embed.", "fields": {"prep": 25, "kcal": 390, "categories": ["Healthy"], "protein": ["cod"],
 "fresh": ["cod", "tomato", "spinach"]}, "ingredients": ["cod", "tomato", "olives", "spinach"]}
```

`group` ties versions of the same thing together. The graph method uses `ingredients` for links, `fields.fresh` to
mark which links count as fresh, and `fields.protein` for protein rules. Any other field can be filtered.

**Questions** (`questions.json`): a list. Each has an `id`, a `shape`, the `text`, an optional `seed` (a document id
the question refers to), `narrow: true` when recall is worth measuring, and one form of answer: `groups` (the
correct group ids), `check` (the name of a check function) or neither (marked by a person later).

**Plans** (`plans.json`): `{question id: {method name: plan}}`. The plan format for each method is in its docstring
under `qshapes/methods/`. A method with no plan for a question is recorded as skipped.

**Checks** (`validators.py`, optional): a `load(corpus)` function returning `{name: function}`. Each function gets
the top group ids and returns true or false.

**Config** (`config.yaml`): see `example/config.yaml`. Optional keys: `tool_descriptions` (a YAML file of tool
descriptions keyed by method name, used by `qshapes.plan`; listing your corpus's real field values here makes a
large difference), `methods` (run only these), `k` (default 10).

The tool descriptions built into `qshapes/plan.py` are written for recipes. For another kind of corpus, supply your
own with `tool_descriptions`.

## Postgres

Set `store: postgres` and a `dsn` in the config to keep vectors in pgvector and links in plain Postgres tables. The
answers are the same as the in-memory stores: vector search is an exact scan, and the graph runs the same SQL.
Tables reload only when the corpus text changes.

## Licence

MIT. See `LICENSE`.
