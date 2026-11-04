"""Final-epoch (epoch 50) metrics for the TM models, so that accuracy and the
measured inference cost refer to the same trained model. Also per-class
recall and precision against the reproduced BHRF."""
import json
import numpy as np
from data import LEAVES, TOP, TREE, PARENT
from metrics import report

import os
RUN = os.environ.get('RUN', 'results/main')
P = np.load('work/prepared.npz')
yte = P['yte']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])
child_leaf = {p: np.array([LEAVES.index(c) for c in TREE[p]]) for p in TOP}
S = {n: np.load(f'{RUN}/{n}_scores.npz')['test'][-1].astype(float)
     for n in ['root', 'Transient', 'Stochastic', 'Periodic', 'flat']}
T = {n: json.load(open(f'{RUN}/{n}_hist.json'))['cfg']['T'] for n in S}
alpha = json.load(open(f'{RUN}/evaluation.json'))['best_alpha']

top = S['root'].argmax(1)
hard = np.empty(len(yte), int)
for i, p in enumerate(TOP):
    m = top == i
    hard[m] = child_leaf[p][S[p][m].argmax(1)]


def sm(z, tau):
    z = z / tau; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


pr = sm(S['root'], alpha * T['root'])
probs = np.zeros((len(yte), 15))
for i, p in enumerate(TOP):
    probs[:, child_leaf[p]] = sm(S[p], alpha * T[p]) * pr[:, [i]]
soft = probs.argmax(1)
flat = S['flat'].argmax(1)
brf = np.load('results/brf/brf_pred_leaf.npy')

res = {}
for name, pred in [('BHRF', brf), ('tree_hard', hard), ('tree_soft', soft), ('flat', flat)]:
    res[name] = {'top': report(top_of[yte], top_of[pred], 3), 'leaf': report(yte, pred, 15)}
json.dump(res, open(os.environ.get('OUT', 'results/final_epoch_metrics.json'), 'w'), indent=1)
print(f"{'model':10s} top_acc top_bal leaf_acc leaf_bal macroP macroR macroF1")
for n, r in res.items():
    print(f"{n:10s} {r['top']['accuracy']:.4f} {r['top']['balanced_accuracy']:.4f} "
          f"{r['leaf']['accuracy']:.4f} {r['leaf']['balanced_accuracy']:.4f} "
          f"{r['leaf']['macro_precision']:.4f} {r['leaf']['macro_recall']:.4f} {r['leaf']['macro_f1']:.4f}")
print('\nper-class recall / precision (test support)')
for j, c in enumerate(LEAVES):
    sup = int((yte == j).sum())
    print(f"{c:15s} n={sup:5d}  " + '  '.join(
        f"{n}: R={res[n]['leaf']['per_class_recall'][j]:.2f} P={res[n]['leaf']['per_class_precision'][j]:.2f}"
        for n in ['BHRF', 'tree_hard', 'flat']))
