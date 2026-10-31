"""Run every method on every question and write a results file.

Usage: python -m qshapes.run CONFIG.yaml
The config names the corpus, questions, plans file, model and output directory. See example/config.yaml.
"""
import importlib.util
import json
import os
import sys
import time
import yaml
from .corpus import load_corpus
from .questions import load_questions
from .scoring import score, top_groups
from .methods.glob_lookup import GlobLookup
from .methods.field_lookup import FieldLookup
from .methods.graph import Graph


def build_methods(cfg, base):
    pg = None
    if cfg.get('store') == 'postgres':
        from .methods.pg import connect, PgEmbeddings, PgGraph
        pg = connect(cfg['dsn'])
    methods = [GlobLookup(), FieldLookup(), PgGraph(pg) if pg else Graph()]
    if cfg.get('embedding_model'):
        from .encoder import OnnxEncoder, hf_snapshot
        from .methods.embeddings import Embeddings
        enc = OnnxEncoder(hf_snapshot(os.path.join(base, cfg['model_cache']), cfg['embedding_model']))
        cache = os.path.join(base, cfg['embedding_cache']) if cfg.get('embedding_cache') else None
        emb = PgEmbeddings(enc, pg, cfg.get('query_prefix', ''), cache) if pg else Embeddings(enc, cfg.get('query_prefix', ''), cache)
        methods.insert(2, emb)
        from .methods.hybrid import Hybrid
        methods.append(Hybrid(emb))
    only = cfg.get('methods')
    if only and 'hybrid' in only and 'embeddings' not in only:
        only = list(only) + ['embeddings']
    return [m for m in methods if not only or m.name in only]


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    base = os.path.dirname(os.path.abspath(config_path))
    p = lambda key: os.path.join(base, cfg[key])
    corpus = load_corpus(p('corpus'))
    group_of = {d.id: d.group for d in corpus}
    questions = load_questions(p('questions'))
    plans = json.load(open(p('plans'))) if cfg.get('plans') and os.path.exists(p('plans')) else {}
    validators = {}
    if cfg.get('validators'):
        spec = importlib.util.spec_from_file_location('validators', p('validators'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        validators = mod.load(corpus)
    k = cfg.get('k', 10)
    os.makedirs(p('out_dir'), exist_ok=True)
    out ={'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'k': k, 'setup': {}, 'runs': []}
    for m in build_methods(cfg, base):
        out['setup'][m.name] = m.setup(corpus)
        for q in questions:
            plan = plans.get(q['id'], {}).get(m.name)
            if plan is None and m.name != 'embeddings':
                out['runs'].append({'q': q['id'], 'shape': q['shape'], 'method': m.name, 'skipped': 'no plan'})
                continue
            r = m.timed(q, plan or {}, k)
            out['runs'].append({'q': q['id'], 'shape': q['shape'], 'method': m.name, 'seconds': round(r.seconds, 4),
                                'top': top_groups(r.doc_ids, group_of, k), **score(q, r.doc_ids, group_of, k, validators),
                                **({'notes': r.notes} if r.notes else {})})
    path = os.path.join(p('out_dir'), f"results-{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump(out, open(path, 'w'), indent=1)
    print(path)
    return out


if __name__ == '__main__':
    main(sys.argv[1])
