"""Decode trained clauses of an expert into readable feature conditions.

Rebuilds the thermometer thresholds deterministically (same split and seed as
prepare.py), maps literal indices to feature conditions, and reports, for a
chosen expert and class, the positive-polarity clauses that best separate the
class on the validation set (true-positive rate minus false-positive rate of
the clause output). Output is used for the interpretability example.
"""
import json
import sys
import numpy as np
from data import load_raw, preprocess, split, Thermometer, LEAVES, TOP, TREE, PARENT

import os
RUN = os.environ.get('RUN', 'results/main')
expert, cls = sys.argv[1], sys.argv[2]
meta = json.load(open('work/meta.json'))
df = load_raw('raw')
X, y = preprocess(df)
(Xtr, ytr), (Xva, yva), _ = split(X, y, seed=meta['seed'])
enc = Thermometer(meta['bins']).fit(Xtr.values)
names = list(X.columns)
lit = []
for j, th in enumerate(enc.thresholds):
    for t in th:
        lit.append((names[j], '>=', float(t)))
    if enc.has_missing[j]:
        lit.append((names[j], 'missing', None))
nb = len(lit)


def describe(i):
    f, op, t = lit[i % nb]
    neg = i >= nb
    if op == 'missing':
        return f'{f} is {"present" if neg else "missing"}'
    return f'{f} {"<" if neg else ">="} {t:.4g}'


z = np.load(f'{RUN}/{expert}_final_masks.npz')
masks = np.unpackbits(z['masks'], axis=2)[:, :, :int(z['n_literals'])].astype(bool)
classes = TOP if expert == 'root' else (LEAVES if expert == 'flat' else TREE[expert])
k = classes.index(cls)
Bva = enc.transform(Xva.values).astype(bool)
L = np.hstack([Bva, ~Bva])  # literals: x, then NOT x
if expert == 'root':
    lab = np.array([PARENT[c] == cls for c in yva])
    keep = np.ones(len(yva), bool)
elif expert == 'flat':
    lab = yva == cls
    keep = np.ones(len(yva), bool)
else:
    keep = np.array([PARENT[c] == expert for c in yva])
    lab = (yva == cls)[keep]
    L = L[keep]
rows = []
for c in range(0, masks.shape[1], 2):  # even index = positive polarity
    inc = np.flatnonzero(masks[k, c])
    if inc.size == 0:
        continue
    fire = L[:, inc].all(axis=1)
    tpr, fpr = fire[lab].mean(), fire[~lab].mean()
    rows.append((tpr - fpr, tpr, fpr, inc))
rows.sort(key=lambda r: -r[0])
out = []
for score, tpr, fpr, inc in rows[:3]:
    out.append({'tpr': round(float(tpr), 3), 'fpr': round(float(fpr), 4), 'n_literals': int(inc.size),
                'literals': [describe(i) for i in inc]})
    print(f'TPR={tpr:.3f} FPR={fpr:.4f} literals={inc.size}')
    for i in inc:
        print('   ', describe(i))
json.dump(out, open(f'results/clauses_{expert}_{cls.replace("/", "_")}.json', 'w'), indent=1)


def simplify(literals):
    """Collapse thermometer literals per feature to the tightest bounds.
    A negated thermometer literal 'f < t' is also true when f is missing."""
    import re
    feats = {}
    for s in literals:
        m = re.match(r'(.+?) (>=|<) (\S+)$', s)
        if m:
            f, op, t = m.group(1), m.group(2), float(m.group(3))
            d = feats.setdefault(f, {})
            if op == '>=':
                d['ge'] = max(d.get('ge', -1e300), t)
            else:
                d['lt'] = min(d.get('lt', 1e300), t)
        else:
            f, st = s.rsplit(' is ', 1)
            feats.setdefault(f, {})[st] = True
    out = []
    for f, d in feats.items():
        parts = []
        if 'ge' in d:
            parts.append(f'>= {d["ge"]:.4g}')
        if 'lt' in d:
            parts.append(f'< {d["lt"]:.4g}')
        if d.get('present'):
            parts.append('present')
        if d.get('missing'):
            parts.append('missing')
        out.append(f + ': ' + ', '.join(parts))
    return out


for i, o in enumerate(out):
    s = simplify(o['literals'])
    o['simplified'] = s
    print(f'--- clause {i}: TPR={o["tpr"]} FPR={o["fpr"]} literals={o["n_literals"]} features={len(s)}')
    for line in s:
        print('   ', line)
json.dump(out, open(f'results/clauses_{expert}_{cls.replace("/", "_")}.json', 'w'), indent=1)
