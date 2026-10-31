"""Score ranked results by group: versions of the same thing count once."""


def top_groups(doc_ids, group_of, k):
    out, seen = [], set()
    for d in doc_ids:
        g = group_of.get(d)
        if g is not None and g not in seen:
            seen.add(g)
            out.append(g)
            if len(out) == k:
                break
    return out


def score(question, doc_ids, group_of, k=10, validators=None):
    s = {'returned_groups': len(top_groups(doc_ids, group_of, 10 ** 9))}
    if 'groups' in question:
        truth = set(question['groups'])
        top = top_groups(doc_ids, group_of, k)
        s['precision_at_k'] = round(len([g for g in top if g in truth]) / len(top), 3) if top else 0.0
        s['k'] = k
        if question.get('narrow'):
            wide = top_groups(doc_ids, group_of, max(k, len(truth)))
            s['recall'] = round(len(truth & set(wide)) / len(truth), 3) if truth else None
    elif 'check' in question and validators:
        s['valid'] = bool(validators[question['check']](top_groups(doc_ids, group_of, question.get('n', 3))))
    else:
        s['judged'] = 'pending'
    return s
