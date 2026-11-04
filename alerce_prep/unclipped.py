"""(1) AP(TM)-consistent evaluation: class votes recomputed from the saved
include masks of the epoch-50 models WITHOUT clipping to [-T, T] (AP(TM)
computes plain popcount differences), argmax with lowest-index ties.
(2) Validation-based choice of the gating threshold delta: the smallest delta
of the sweep that reaches the highest validation balanced accuracy.
Writes results/unclipped_metrics.json and results/gate_validation.json."""
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score as bal
from data import LEAVES, TOP, TREE, PARENT
from metrics import report

R = 'results/main'
P = np.load('work/prepared.npz')
yte, yva = P['yte'], P['yva']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])
child_leaf = {p: np.array([LEAVES.index(c) for c in TREE[p]]) for p in TOP}
Lte = np.hstack([P['Bte'], 1 - P['Bte']]).astype(bool)


def votes(name, L):
    z = np.load(f'{R}/{name}_final_masks.npz')
    m = np.unpackbits(z['masks'], axis=2)[:, :, :int(z['n_literals'])].astype(bool)
    k, C, _ = m.shape
    sign = np.where(np.arange(C) % 2 == 0, 1, -1)
    out = np.zeros((L.shape[0], k), dtype=np.int64)
    notL = (~L).astype(np.float32)
    for c in range(k):
        inc = m[c].astype(np.float32)               # clauses x literals
        viol = notL @ inc.T                          # samples x clauses
        fire = (viol == 0) & (inc.sum(1) > 0)[None, :]
        out[:, c] = (fire * sign[None, :]).sum(1)
    return out


V = {n: votes(n, Lte) for n in ['root', 'Transient', 'Stochastic', 'Periodic', 'flat']}
T = {n: json.load(open(f'{R}/{n}_hist.json'))['cfg']['T'] for n in V}
alpha = json.load(open(f'{R}/evaluation.json'))['best_alpha']
top = V['root'].argmax(1)
hard = np.empty(len(yte), int)
for i, p in enumerate(TOP):
    msk = top == i
    hard[msk] = child_leaf[p][V[p][msk].argmax(1)]


def sm(z, tau):
    z = z / tau; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


probs = np.zeros((len(yte), 15))
pr = sm(V['root'].astype(float), alpha * T['root'])
for i, p in enumerate(TOP):
    probs[:, child_leaf[p]] = sm(V[p].astype(float), alpha * T[p]) * pr[:, [i]]
soft = probs.argmax(1)
flat = V['flat'].argmax(1)
res = {}
for name, pred, tp in [('tree_hard', hard, top), ('tree_soft', soft, top_of[soft]), ('flat', flat, top_of[flat])]:
    res[name] = {'top': report(top_of[yte], tp, 3), 'leaf': report(yte, pred, 15)}
    print(name, round(res[name]['top']['accuracy'], 4), round(res[name]['leaf']['accuracy'], 4),
          round(res[name]['leaf']['balanced_accuracy'], 4), round(res[name]['leaf']['macro_precision'], 4),
          round(res[name]['leaf']['macro_f1'], 4))
clipped_frac = {n: float((np.abs(V[n]) > T[n]).any(1).mean()) for n in V}
res['fraction_samples_with_a_vote_beyond_T'] = clipped_frac
json.dump(res, open('results/unclipped_metrics.json', 'w'), indent=1)

# Validation-based delta (stored clipped validation scores at epoch 50, as in the main results)
S = {n: np.load(f'{R}/{n}_scores.npz')['val'][-1].astype(float) for n in ['root'] + TOP}
r = S['root']
pr = sm(r, alpha * T['root'])
pv = np.zeros((len(yva), 15))
for i, p in enumerate(TOP):
    pv[:, child_leaf[p]] = sm(S[p], alpha * T[p]) * pr[:, [i]]
srt = np.sort(r, 1); margin = (srt[:, -1] - srt[:, -2]) / T['root']
t1 = r.argmax(1); t2 = np.argsort(r, 1)[:, -2]
sweep = []
for d in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.01, 2.01]:
    conf = margin >= d
    keep = np.zeros_like(pv, bool)
    for i, p in enumerate(TOP):
        keep[:, child_leaf[p]] = ((t1 == i) | (~conf & (t2 == i)))[:, None]
    pred = np.where(keep, pv, -1).argmax(1)
    sweep.append({'delta': d, 'val_bal': float(bal(yva, pred)), 'mean_children': float(np.where(conf, 1, 2).mean())})
best = max(s['val_bal'] for s in sweep)
chosen = min(s['delta'] for s in sweep if s['val_bal'] >= best - 1e-12)
json.dump({'sweep': sweep, 'chosen_delta': chosen}, open('results/gate_validation.json', 'w'), indent=1)
print('validation sweep:', [(s['delta'], round(s['val_bal'], 4)) for s in sweep], 'chosen', chosen)
print('fraction of samples with |vote|>T:', clipped_frac)

# (3) Saturation of the clipped class sums on the test set (stored scores, epoch 50):
# 'sat' = fraction of sources whose largest class sum equals T,
# 'tie' = fraction whose largest class sum equals T for two or more classes.
cs = {}
for n in ['root', 'flat']:
    s = np.load(f'{R}/{n}_scores.npz')['test'][-1].astype(int)
    mx = s.max(1)
    sat = mx == T[n]
    cs[n] = {'sat': float(sat.mean()), 'tie': float((sat & ((s == mx[:, None]).sum(1) > 1)).mean())}
json.dump(cs, open('results/clip_stats.json', 'w'), indent=1)
print('clip stats:', cs)
