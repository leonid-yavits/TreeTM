"""Prior-corrected TM-Tree, the same one-parameter correction as brf_prior.py
(response to review round 2, M-2): each expert's pseudo-probabilities
softmax(V/(alpha T)) are multiplied by the training class prior of its node
raised to the power g and renormalized. g = 0 is the uncorrected model.
g is selected on the VALIDATION set by macro F1 over the same grid as for the
BHRF. Hard routing: argmax of the corrected root, then argmax of the corrected
child. Soft routing: product of corrected probabilities.
Uses the stored clipped scores at epoch 50. Writes results/tm_prior.json."""
import json
import numpy as np
from sklearn.metrics import f1_score
from data import LEAVES, TOP, TREE, PARENT
from metrics import report

R = 'results/main'
P = np.load('work/prepared.npz')
ytr, yva, yte = P['ytr'], P['yva'], P['yte']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])
cl = {p: np.array([LEAVES.index(c) for c in TREE[p]]) for p in TOP}
T = {n: json.load(open(f'{R}/{n}_hist.json'))['cfg']['T'] for n in ['root'] + TOP}
alpha = json.load(open(f'{R}/evaluation.json'))['best_alpha']
prior_top = np.bincount(top_of[ytr], minlength=3) / len(ytr)
prior_child = {p: np.bincount(ytr[np.isin(ytr, cl[p])], minlength=15)[cl[p]]
               / np.isin(ytr, cl[p]).sum() for p in TOP}
S = {s: {n: np.load(f'{R}/{n}_scores.npz')[s][-1].astype(float) for n in ['root'] + TOP}
     for s in ('val', 'test')}


def sm(z, t):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def corrected(split, g):
    r = sm(S[split]['root'], alpha * T['root']) * prior_top ** g
    r = r / r.sum(1, keepdims=True)
    c = {}
    for p in TOP:
        b = sm(S[split][p], alpha * T[p]) * prior_child[p] ** g
        c[p] = b / b.sum(1, keepdims=True)
    return r, c


def predict(split, g, policy):
    r, c = corrected(split, g)
    if policy == 'soft':
        pr = np.zeros((len(r), 15))
        for i, p in enumerate(TOP):
            pr[:, cl[p]] = c[p] * r[:, [i]]
        return pr.argmax(1), top_of[pr.argmax(1)]
    top = r.argmax(1)
    out = np.empty(len(r), int)
    for i, p in enumerate(TOP):
        m = top == i
        out[m] = cl[p][c[p][m].argmax(1)]
    return out, top


grid = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
res = {}
for policy in ('hard', 'soft'):
    sweep = [{'g': g, 'val_macro_f1': float(f1_score(yva, predict('val', g, policy)[0], average='macro',
                                                      labels=range(15), zero_division=0))} for g in grid]
    best = max(s['val_macro_f1'] for s in sweep)
    g = min(s['g'] for s in sweep if s['val_macro_f1'] >= best - 1e-12)
    pred, top = predict('test', g, policy)
    res[policy] = {'sweep': sweep, 'chosen_g': g,
                   'test': {'top': report(top_of[yte], top, 3), 'leaf': report(yte, pred, 15)}}
    r = res[policy]['test']['leaf']
    print(policy, 'g', g, 'acc %.4f bal %.4f P %.4f F1 %.4f top %.4f' % (
        r['accuracy'], r['balanced_accuracy'], r['macro_precision'], r['macro_f1'],
        res[policy]['test']['top']['accuracy']))
json.dump(res, open('results/tm_prior.json', 'w'), indent=1)
