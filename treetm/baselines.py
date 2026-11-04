"""Non-TM baselines on a prepared dataset, one seed per call.

  flat_rf    random forest over all leaves (class_weight='balanced_subsample')
  lcpn_rf    the same forest as a local classifier per parent node, i.e. the
             TreeTM structure with random-forest experts (hard routing and
             soft probability product)
  flat_hgb   histogram gradient boosting over all leaves (class_weight='balanced')
  flat_mlp   one-hidden-layer MLP on standardized features, early stopping

Usage: python baselines.py <dataset> <seed> [--models flat_rf,lcpn_rf,flat_hgb,flat_mlp]
Writes results/<dataset>/baselines/seed<seed>/<model>.json (and predictions).
"""
import argparse
import json
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from taxonomy import Taxonomy
from evaluate import full_metrics

ap = argparse.ArgumentParser()
ap.add_argument('dataset')
ap.add_argument('seed', type=int)
ap.add_argument('--models', default='flat_rf,lcpn_rf,flat_hgb,flat_mlp')
ap.add_argument('--trees', type=int, default=300)
a = ap.parse_args()

meta = json.load(open(f'work/{a.dataset}/meta.json'))
tax = Taxonomy(meta['taxonomy'])
P = np.load(f'work/{a.dataset}/prepared.npz')
Ftr, Fva, Fte = P['Ftr'], P['Fva'], P['Fte']
ytr, yva, yte = P['ytr'], P['yva'], P['yte']
out_dir = f'results/{a.dataset}/baselines/seed{a.seed}'
os.makedirs(out_dir, exist_ok=True)


def rf():
    return RandomForestClassifier(n_estimators=a.trees, class_weight='balanced_subsample',
                                  n_jobs=2, random_state=a.seed)


def finish(name, pred, extra, t_fit, t_pred):
    r = full_metrics(tax, yte, pred)
    r.update(extra)
    r.update({'fit_s': t_fit, 'predict_s_per_input': t_pred / len(yte), 'rf_trees': a.trees})
    json.dump(r, open(f'{out_dir}/{name}.json', 'w'), indent=1)
    np.save(f'{out_dir}/{name}_pred.npy', pred)
    print(f"{a.dataset} seed{a.seed} {name:9s} acc {r['accuracy']:.4f} bal {r['balanced_accuracy']:.4f} "
          f"P {r['macro_precision']:.4f} F1 {r['macro_f1']:.4f} hF {r['hF']:.4f} fit {t_fit:.0f}s", flush=True)


for m in a.models.split(','):
    done_name = 'lcpn_rf_soft' if m == 'lcpn_rf' else m
    if os.path.exists(f'{out_dir}/{done_name}.json'):
        print('skip', m, flush=True)
        continue
    if m == 'flat_rf':
        t0 = time.time(); c = rf().fit(Ftr, ytr); t1 = time.time()
        p = c.classes_[c.predict_proba(Fte).argmax(1)]; t2 = time.time()
        finish(m, p, {}, t1 - t0, t2 - t1)
    elif m == 'flat_hgb':
        t0 = time.time()
        c = HistGradientBoostingClassifier(max_iter=200, class_weight='balanced', early_stopping=True,
                                           random_state=a.seed).fit(Ftr, ytr)
        t1 = time.time(); p = c.predict(Fte); t2 = time.time()
        finish(m, p, {'n_iter': int(c.n_iter_)}, t1 - t0, t2 - t1)
    elif m == 'flat_mlp':
        sc = StandardScaler().fit(Ftr)
        t0 = time.time()
        c = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200,
                          random_state=a.seed).fit(sc.transform(Ftr), ytr)
        t1 = time.time(); p = c.predict(sc.transform(Fte)); t2 = time.time()
        finish(m, p, {'n_iter': int(c.n_iter_)}, t1 - t0, t2 - t1)
    elif m == 'lcpn_rf':
        t0 = time.time()
        models = {}
        for u in tax.internal:
            keep, lab = tax.node_labels(u, ytr)
            models[u] = rf().fit(Ftr[keep], lab[keep])
        t1 = time.time()
        prob = {u: np.zeros((len(yte), len(tax.children[u]))) for u in tax.internal}
        for u, c in models.items():
            prob[u][:, c.classes_] = c.predict_proba(Fte)
        # hard routing
        node = np.array(['root'] * len(yte), dtype=object)
        active = np.ones(len(yte), bool)
        while active.any():
            for u in set(node[active]):
                mm = active & (node == u)
                node[mm] = np.array(tax.children[u], dtype=object)[prob[u][mm].argmax(1)]
            active = np.array([x in tax.children for x in node])
        ph = np.array([tax.leaf_index[x] for x in node])
        t2 = time.time()
        finish('lcpn_rf_hard', ph, {}, t1 - t0, t2 - t1)
        leaf = np.ones((len(yte), len(tax.leaves)))
        for li, path in enumerate(tax.paths):
            for u, k in path:
                leaf[:, li] *= prob[u][:, k]
        finish('lcpn_rf_soft', leaf.argmax(1), {}, t1 - t0, t2 - t1)
