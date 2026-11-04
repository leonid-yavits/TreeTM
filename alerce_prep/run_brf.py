"""Reproduce the ALeRCE hierarchical balanced random forest (BHRF) on our split
and measure its inference cost.

Hyperparameters follow the authors' notebook:
n_estimators=500, max_features='sqrt' ('auto' in the 2020 sklearn release),
0.2 for the stochastic forest, max_depth=None, criterion='entropy',
class_weight='balanced_subsample', bootstrap=True.
imblearn defaults changed after 2020, so the 2020 sampling defaults
(sampling_strategy='auto', replacement=False) are set explicitly.
Final probabilities: P_top(parent) * P_child(leaf), as in the notebook.
"""
import json
import os
import pickle
import time
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from data import LEAVES, TOP, TREE, PARENT
from metrics import report

P = np.load('work/prepared.npz')
out = 'results/brf'
os.makedirs(out, exist_ok=True)
Ftr, Fte, ytr, yte = P['Ftr'], P['Fte'], P['ytr'], P['yte']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])


def brf(max_features='sqrt'):
    return BalancedRandomForestClassifier(
        n_estimators=500, max_features=max_features, max_depth=None, n_jobs=2,
        bootstrap=True, class_weight='balanced_subsample', criterion='entropy',
        min_samples_split=2, min_samples_leaf=1,
        sampling_strategy='auto', replacement=False, random_state=0)


def forest_cost(model, X):
    """Node counts, depth, and mean internal nodes visited per sample (comparisons)."""
    n_nodes = sum(e.tree_.node_count for e in model.estimators_)
    n_leaves = sum(e.tree_.n_leaves for e in model.estimators_)
    depth = [e.tree_.max_depth for e in model.estimators_]
    sub = X[:2000]
    ind, _ = model.decision_path(sub)
    visited = np.asarray(ind.sum(axis=1)).ravel()  # nodes on paths summed over trees
    comps = visited - len(model.estimators_)       # leaves are not comparisons
    return {'trees': len(model.estimators_), 'nodes': int(n_nodes), 'leaves': int(n_leaves),
            'internal_nodes': int(n_nodes - n_leaves), 'max_depth_max': int(max(depth)),
            'max_depth_mean': float(np.mean(depth)),
            'comparisons_per_sample_mean': float(comps.mean()),
            'n_classes': int(model.n_classes_),
            'pickle_bytes': len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))}


res = {}
t0 = time.time()
top = brf().fit(Ftr, top_of[ytr])
res['top_fit_s'] = time.time() - t0
p_top = top.predict_proba(Fte)
cost = {'top': forest_cost(top, Fte)}
probs = np.zeros((len(yte), len(LEAVES)))
for parent in TOP:
    leaves = [LEAVES.index(c) for c in TREE[parent]]
    m = np.isin(ytr, leaves)
    t0 = time.time()
    clf = brf(0.2 if parent == 'Stochastic' else 'sqrt').fit(Ftr[m], ytr[m])
    res[f'{parent}_fit_s'] = time.time() - t0
    pc = clf.predict_proba(Fte)
    pi = list(top.classes_).index(TOP.index(parent))
    for j, leaf in enumerate(clf.classes_):
        probs[:, leaf] = pc[:, j] * p_top[:, pi]
    cost[parent] = forest_cost(clf, Fte)

pred_top = top.classes_[p_top.argmax(1)]
pred_leaf = probs.argmax(1)
res['top'] = report(top_of[yte], pred_top, len(TOP))
res['leaf'] = report(yte, pred_leaf, len(LEAVES))
res['cost'] = cost
with open(os.path.join(out, 'brf_results.json'), 'w') as f:
    json.dump(res, f, indent=1)
np.save(os.path.join(out, 'brf_pred_leaf.npy'), pred_leaf)
print(json.dumps({k: res[k] for k in ('top', 'leaf')}, indent=1))
print(json.dumps(cost, indent=1))
