"""Clause length statistics (included literals per clause, non-empty clauses)
for TreeTM experts and the flat TM. Usage: python clause_len.py <dataset> [variant]
Writes results/<dataset>/clause_length[_<variant>].json"""
import glob, json, sys
import numpy as np
from taxonomy import Taxonomy
ds = sys.argv[1]; var = sys.argv[2] if len(sys.argv) > 2 else 'main'
tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
out = {'tree': [], 'flat': [], 'per_seed': []}
for run in sorted(glob.glob(f'results/{ds}/{var}/seed*/')):
    def lens(u):
        z = np.load(f'{run}{u}_final_masks.npz'); nl = int(z['n_literals'])
        m = np.unpackbits(z['masks'], axis=2)[:, :, :nl]
        L = m.sum(2).ravel(); return L[L > 0]
    t = np.concatenate([lens(u) for u in tax.internal]); f = lens('flat')
    out['per_seed'].append({'run': run.rstrip('/'), 'tree_median': float(np.median(t)), 'tree_mean': float(t.mean()),
                            'flat_median': float(np.median(f)), 'flat_mean': float(f.mean())})
r = out['per_seed']
for k in ('tree_median', 'tree_mean', 'flat_median', 'flat_mean'):
    out[k] = float(np.mean([x[k] for x in r]))
del out['tree'], out['flat']
json.dump(out, open(f'results/{ds}/clause_length' + ('' if var == 'main' else f'_{var}') + '.json', 'w'), indent=1)
print(ds, {k: round(v, 1) for k, v in out.items() if k != 'per_seed'})
