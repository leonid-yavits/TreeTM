"""Decode the best positive clauses of one class of one expert into readable
feature conditions. Clauses are ranked by validation TPR minus FPR among the
validation samples of the expert's subtree.
Usage: python interpret.py <dataset> <run_dir> <expert> <class_name> [top]
Writes <run_dir>/clauses_<expert>_<class>.json"""
import json
import re
import sys
import numpy as np
from taxonomy import Taxonomy

ds, run, u, cname = sys.argv[1:5]
top = int(sys.argv[5]) if len(sys.argv) > 5 else 3
meta = json.load(open(f'work/{ds}/meta.json'))
tax = Taxonomy(meta['taxonomy'])
names = meta['bit_names']
P = np.load(f'work/{ds}/prepared.npz')
z = np.load(f'{run}/{u}_final_masks.npz')
nl = int(z['n_literals'])
m = np.unpackbits(z['masks'], axis=2)[:, :, :nl].astype(bool)
k = tax.children[u].index(cname)
keep, lab = tax.node_labels(u, P['yva'])
X = P['Bva'][keep].astype(bool)
L = np.hstack([X, ~X])
y = lab[keep] == k
res = []
for j in range(0, m.shape[1], 2):          # even index: positive polarity
    inc = m[k, j]
    if not inc.any():
        continue
    fire = (L[:, inc]).all(1)
    tpr, fpr = fire[y].mean(), fire[~y].mean()
    res.append((tpr - fpr, j, tpr, fpr, inc))
res.sort(key=lambda r: -r[0])


def simplify(lits):
    """Merge thermometer literals per feature into one interval condition."""
    lo, hi, other = {}, {}, []
    for i in lits:
        neg = i >= len(names)
        n = names[i % len(names)]
        mt = re.match(r'(.+) >= (.+)$', n)
        if mt:
            f, t = mt.group(1), float(mt.group(2))
            if neg:
                hi[f] = min(hi.get(f, np.inf), t)
            else:
                lo[f] = max(lo.get(f, -np.inf), t)
        else:
            other.append(('NOT ' if neg else '') + n)
    conds = []
    for f in sorted(set(lo) | set(hi)):
        a, b = lo.get(f), hi.get(f)
        if a is not None and b is not None:
            conds.append(f'{a:.4g} <= {f} < {b:.4g}')
        elif a is not None:
            conds.append(f'{f} >= {a:.4g}')
        else:
            conds.append(f'{f} < {b:.4g}')
    return conds + other


out = []
for score, j, tpr, fpr, inc in res[:top]:
    lits = np.flatnonzero(inc).tolist()
    out.append({'clause': int(j), 'tpr': float(tpr), 'fpr': float(fpr), 'n_literals': len(lits),
                'simplified': simplify(lits)})
    print(f'clause {j}: TPR {tpr:.3f} FPR {fpr:.4f} literals {len(lits)}')
    for c in simplify(lits):
        print('   ', c)
json.dump(out, open(f'{run}/clauses_{u}_{cname.replace("/", "")}.json', 'w'), indent=1)
