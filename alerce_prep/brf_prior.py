"""Prior-corrected BHRF (response to review round 2, M-2 and m-17).

Retrains the reproduced BHRF with exactly the settings of run_brf.py
(deterministic, random_state=0), checks that its leaf predictions equal the
saved ones, and stores validation and test probabilities.

Prior correction: each level's probabilities are multiplied by the training
class prior raised to the power g and renormalized; the leaf score is the
product of the corrected top-level and child-level probabilities. g = 0 is
the original BHRF. g is selected on the VALIDATION set by macro F1.

Also records the routing fractions of a hard-routed BHRF that routes by the
argmax of its top-level forest (m-17).
Writes results/brf/brf_prior.json and results/brf/brf_probs.npz.
"""
import json
import numpy as np
from sklearn.metrics import f1_score
from data import LEAVES, TOP, TREE, PARENT
from metrics import report
from imblearn.ensemble import BalancedRandomForestClassifier


def brf(max_features='sqrt'):
    # identical to run_brf.py
    return BalancedRandomForestClassifier(
        n_estimators=500, max_features=max_features, max_depth=None, n_jobs=2,
        bootstrap=True, class_weight='balanced_subsample', criterion='entropy',
        min_samples_split=2, min_samples_leaf=1,
        sampling_strategy='auto', replacement=False, random_state=0)

P = np.load('work/prepared.npz')
Ftr, Fva, Fte = P['Ftr'], P['Fva'], P['Fte']
ytr, yva, yte = P['ytr'], P['yva'], P['yte']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])
child_leaf = {p: np.array([LEAVES.index(c) for c in TREE[p]]) for p in TOP}

top = brf().fit(Ftr, top_of[ytr])
assert list(top.classes_) == [0, 1, 2]
pt = {'val': top.predict_proba(Fva), 'test': top.predict_proba(Fte)}
pc = {'val': {}, 'test': {}}
for p in TOP:
    m = np.isin(ytr, child_leaf[p])
    clf = brf(0.2 if p == 'Stochastic' else 'sqrt').fit(Ftr[m], ytr[m])
    assert list(clf.classes_) == list(child_leaf[p])
    pc['val'][p] = clf.predict_proba(Fva)
    pc['test'][p] = clf.predict_proba(Fte)

prior_top = np.bincount(top_of[ytr], minlength=3) / len(ytr)
prior_child = {p: np.bincount(ytr[np.isin(ytr, child_leaf[p])], minlength=15)[child_leaf[p]]
               / np.isin(ytr, child_leaf[p]).sum() for p in TOP}


def combine(split, g):
    a = pt[split] * prior_top ** g
    a = a / a.sum(1, keepdims=True)
    pr = np.zeros((len(a), 15))
    for i, p in enumerate(TOP):
        b = pc[split][p] * prior_child[p] ** g
        b = b / b.sum(1, keepdims=True)
        pr[:, child_leaf[p]] = b * a[:, [i]]
    return pr.argmax(1)


saved = np.load('results/brf/brf_pred_leaf.npy')
agree = float((combine('test', 0.0) == saved).mean())
print('agreement with saved BHRF predictions:', agree)
grid = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
sweep = [{'g': g, 'val_macro_f1': float(f1_score(yva, combine('val', g), average='macro',
                                                  labels=range(15), zero_division=0))} for g in grid]
best = max(s['val_macro_f1'] for s in sweep)
g = min(s['g'] for s in sweep if s['val_macro_f1'] >= best - 1e-12)
pred = combine('test', g)
top_route = pt['test'].argmax(1)
out = {'agreement_with_saved': agree, 'sweep': sweep, 'chosen_g': g,
       'test': {'top': report(top_of[yte], top_of[pred], 3), 'leaf': report(yte, pred, 15)},
       'top_forest_routing_fraction_test': {p: float((top_route == i).mean()) for i, p in enumerate(TOP)}}
json.dump(out, open('results/brf/brf_prior.json', 'w'), indent=1)
np.savez_compressed('results/brf/brf_probs.npz', top_val=pt['val'], top_test=pt['test'],
                    **{f'{p}_{s}': pc[s][p] for p in TOP for s in ('val', 'test')})
r = out['test']['leaf']
print('chosen g', g, 'test acc %.4f bal %.4f P %.4f F1 %.4f' % (
    r['accuracy'], r['balanced_accuracy'], r['macro_precision'], r['macro_f1']))
print('top-forest routing', out['top_forest_routing_fraction_test'])
