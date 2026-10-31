"""Ask a model to write each method's query plan for every question, as a calling agent would.

Usage: python -m qshapes.plan CONFIG.yaml [--model sonnet] [--methods glob-lookup,field-lookup,graph]
A config may override each tool description (`tool_descriptions`, YAML keyed by method name), for
example to list a corpus's real field values. The model sees the method's query format, the questions and any seed ids. It never sees ground truth.
Plans are written to the plans file and token use to plans-usage.json, once, so every run reuses them.
"""
import json
import os
import subprocess
import sys
import tempfile
import yaml

FORMATS = {
 'glob-lookup': '''You can only list recipe files by name. Files are kebab-case recipe titles, for example
"spiced-chickpea-and-spinach-curry" or "garlic-butter-salmon-with-greens". You choose shell-style glob
patterns and get back every file whose name matches any pattern. You cannot search inside files.
Plan: {"globs": ["*chicken-breast*stew*", "*chicken-breast*curry*"]}''',
 'field-lookup': '''You can filter recipes on their fields. Fields: prep (minutes, number), kcal (per serving, number),
rating (0 to 5, number), cuisine (text, e.g. Greek, Indian, Italian), categories (list of text, e.g. Vegetarian,
Healthy, Quick), protein (list, e.g. chicken breast, chicken thigh, cod, tofu,
beef), title (text). Ingredient conditions match any ingredient line containing the text.
Plan: {"all": [conditions], "sort": ["-rating"]} where every condition must hold. Conditions:
  {"ingredient": "chicken breast"}
  {"title_any": ["stew", "curry", "casserole"]}
  {"field": "prep", "op": "<=", "value": 40}      (ops: <, <=, >, >=, ==)
  {"field": "categories", "any": ["Vegetarian"]}  (list or text field contains any value)
  {"not": <condition>}''',
 'graph': '''You can query a graph of recipes linked to their ingredients. "Fresh" ingredients are produce, chilled
dairy, meat, fish and pulses (store-cupboard items like oil, spices, sauces, grains and tins are not fresh).
Each recipe also has a protein list (e.g. chicken breast, chicken thigh, cod, tofu, turkey, beef).
Plan, one of:
  {"rule": "shares_with_seed", "seed": "<recipe id>", "min_shared": 3, "fresh_only": true}
  {"rule": "uses_all", "items": ["feta", "kale"], "exclude_seed": "<recipe id>"}
  {"rule": "basket", "items": ["chicken breast", "spinach"], "min_used": 2, "max_extra": 2}
Any rule may add "protein_in": [...] or "protein_not_in": [...] (matched as text within protein names).
Results are ranked by ingredients shared or used.''',
 'hybrid': '''You can search recipes by meaning and filter them on their fields at the same time. "like" is a short
description of what the dish should be like; recipes are ranked by how close they are in meaning to it. "filters" are
hard rules every result must pass, in the same form as a field lookup. Put anything the dish must or must not be
(an ingredient, a diet, a time limit, a cuisine to avoid) in filters, not in "like": "like" should describe only what
you want, never what you don't. Fields: prep (minutes), kcal, rating (0 to 5), cuisine, categories (list), protein
(list), title. Conditions: {"ingredient": "..."}, {"title_any": [...]}, {"field": "prep", "op": "<=", "value": 30},
{"field": "categories", "any": [...]}, {"not": <condition>}.
Plan: {"like": "light, fresh, zesty", "filters": [{"not": {"title_any": ["curry"]}}], "exclude_seed": "<recipe id>"}'''}

PROMPT = '''You are writing search plans for a recipe-retrieval tool. For each question below, write the plan you
would pass to this tool to answer it as well as the tool allows. If the tool cannot express the question at all,
use {"none": true}. Do not explain. Reply with one JSON object mapping each question id to its plan, nothing else.

TOOL
{fmt}

QUESTIONS
{questions}'''


def main(config_path, model='sonnet', methods=None):
    cfg = yaml.safe_load(open(config_path))
    base = os.path.dirname(os.path.abspath(config_path))
    questions = json.load(open(os.path.join(base, cfg['questions'])))
    plans_path = os.path.join(base, cfg['plans'])
    plans = json.load(open(plans_path)) if os.path.exists(plans_path) else {}
    usage_path = os.path.join(os.path.dirname(plans_path), 'plans-usage.json')
    usage = json.load(open(usage_path)) if os.path.exists(usage_path) else {}
    formats = dict(FORMATS)
    if cfg.get('tool_descriptions'):
        formats.update(yaml.safe_load(open(os.path.join(base, cfg['tool_descriptions']))))
    qtext = '\n'.join(f"{q['id']}: {q['text']}" + (f" (recipe id of the dish named: {q['seed']})" if q.get('seed') else '')
                      for q in questions)
    for name in methods or formats:
        prompt = PROMPT.replace("{fmt}", formats[name]).replace("{questions}", qtext)
        with tempfile.TemporaryDirectory() as empty:
            r = subprocess.run(['claude', '-p', prompt, '--model', model, '--tools', '', '--setting-sources', '',
                                '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json'],
                               cwd=empty, capture_output=True, text=True, timeout=600)
        out = json.loads(r.stdout)
        text = out['result'].strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        got = json.loads(text)
        for qid, plan in got.items():
            plans.setdefault(qid, {})[name] = plan
        u = out.get('usage', {})
        usage[name] = {'model': model, 'questions': len(got), 'input_tokens': u.get('input_tokens'),
                       'cache_creation_input_tokens': u.get('cache_creation_input_tokens'),
                       'cache_read_input_tokens': u.get('cache_read_input_tokens'),
                       'output_tokens': u.get('output_tokens'), 'cost_usd': out.get('total_cost_usd'),
                       'seconds': round(out.get('duration_ms', 0) / 1000, 1)}
        print(name, usage[name])
    json.dump(plans, open(plans_path, 'w'), indent=1)
    json.dump(usage, open(usage_path, 'w'), indent=1)


if __name__ == '__main__':
    args = sys.argv[1:]
    model = args[args.index('--model') + 1] if '--model' in args else 'sonnet'
    meth = args[args.index('--methods') + 1].split(',') if '--methods' in args else None
    main(args[0], model, meth)
