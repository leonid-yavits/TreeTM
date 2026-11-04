"""Expected dense literal tests per input of confidence-gated routing
(all seeds of a variant), relative to the flat TM. Review round 2, m-10.
Usage: python gated_cost.py <dataset> <variant>   -> results/<ds>/<variant>/gated_cost.json"""
import glob, json, sys
import numpy as np
from taxonomy import Taxonomy
from evaluate import Router

ds, var = sys.argv[1], sys.argv[2]
tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
M = 2 * json.load(open(f'work/{ds}/meta.json'))['n_bits']
out = []
for run in sorted(d for d in glob.glob(f'results/{ds}/{var}/seed*') if not d.endswith('.out')):
    ev = json.load(open(f'{run}/treetm_eval.json'))
    nodes = tax.internal + ['flat']
    T = {u: json.load(open(f'{run}/{u}_hist.json'))['cfg']['T'] for u in nodes}
    C = {u: json.load(open(f'{run}/{u}_hist.json'))['cfg']['clauses'] for u in nodes}
    S = {u: np.load(f'{run}/{u}_scores.npz')['test'][-1].astype(float) for u in nodes}
    n = len(S['flat']); tests = np.zeros(n)
    r = Router(tax, S, T)
    delta = ev['delta']

    def go(u, mask):
        if not mask.any():
            return
        tests[mask] += len(tax.children[u]) * C[u] * M
        s = S[u]; srt = np.sort(s, 1)
        conf = (srt[:, -1] - srt[:, -2]) / T[u] >= delta
        o = np.argsort(-s, 1)
        for k, c in enumerate(tax.children[u]):
            take = mask & ((o[:, 0] == k) | (~conf & (o[:, 1] == k)))
            if c in tax.children:
                go(c, take)
    go('root', np.ones(n, bool))
    flat = len(tax.leaves) * C['flat'] * M
    out.append({'run': run, 'gated_dense_tests': float(tests.mean()), 'flat_dense_tests': flat,
                'reduction': float(1 - tests.mean() / flat)})
json.dump(out, open(f'results/{ds}/{var}/gated_cost.json', 'w'), indent=1)
print(ds, var, [round(o['reduction'], 3) for o in out], round(float(np.mean([o['reduction'] for o in out])), 3))
