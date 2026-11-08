"""Evaluate trained TreeTM and flat TM runs on a prepared dataset.

Usage: python evaluate.py <dataset> <run_dir> [<run_dir> ...]
For every run (one seed), at the final epoch:
  * TreeTM with hard routing (only when the run holds all experts, root included),
  * flat TM,
  * leaf metrics (accuracy, balanced accuracy, macro precision/recall/F1),
    per-level accuracy, hierarchical precision/recall/F-measure (ancestor
    sets, root excluded), and the mean number of experts evaluated per input.
Writes <run_dir>/treetm_eval.json and <run_dir>/treetm_predictions.npz.
"""
import json
import os
import sys
import zlib
import numpy as np
from taxonomy import Taxonomy
from metrics import report


def load_scores(run, u, sp):
    """Class sums of node u on split sp ('val' or 'test'): the unclipped sums
    of unclip.py when present, otherwise the clipped sums of the library.
    Exact ties are broken uniformly at random (fixed seed per node and split)
    by adding a perturbation far below one vote, which removes the bias of
    argmax toward the lowest class index."""
    if os.path.exists(f'{run}/{u}_uscores.npz'):
        s = np.load(f'{run}/{u}_uscores.npz')[sp]
    else:
        s = np.load(f'{run}/{u}_scores.npz')[sp][-1]
    rng = np.random.default_rng(zlib.crc32(f'{u}/{sp}'.encode()))
    return s.astype(float) + rng.uniform(0, 1e-3, s.shape)


class Router:
    """Hard top-down routing: at every internal node, follow the argmax child."""

    def __init__(self, tax, S, T=None):
        self.tax, self.S = tax, S   # S[node]: (n, K) class sums

    def hard(self):
        tax, n = self.tax, len(next(iter(self.S.values())))
        pred = np.empty(n, int)
        visits = np.zeros(n, int)
        node = np.array(['root'] * n, dtype=object)
        done = np.zeros(n, bool)
        while not done.all():
            for u in set(node[~done]):
                m = (~done) & (node == u)
                c = self.S[u][m].argmax(1)
                visits[m] += 1
                node[m] = np.array(tax.children[u], dtype=object)[c]
            for i in np.flatnonzero(~done):
                if node[i] in tax.leaf_index:
                    pred[i] = tax.leaf_index[node[i]]
                    done[i] = True
        return pred, visits


def hier_prf(tax, y, p):
    inter = tot_p = tot_t = 0
    for a, b in zip(y, p):
        A, B = set(tax.ancestors(int(a))), set(tax.ancestors(int(b)))
        inter += len(A & B); tot_p += len(B); tot_t += len(A)
    hp, hr = inter / tot_p, inter / tot_t
    return {'hP': hp, 'hR': hr, 'hF': 2 * hp * hr / (hp + hr)}


def level_acc(tax, y, p):
    return {f'level{d}': float(np.mean(np.array(tax.level_label(y, d)) == np.array(tax.level_label(p, d))))
            for d in range(1, tax.depth + 1)}


def full_metrics(tax, y, p):
    r = report(y, p, len(tax.leaves))
    r.update(level_acc(tax, y, p))
    r.update(hier_prf(tax, y, p))
    return r


def evaluate_run(ds, run):
    tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
    yte = np.load(f'work/{ds}/prepared.npz')['yte']
    out, pred = {}, {'y': yte}
    if all(os.path.exists(f'{run}/{u}_final_masks.npz') for u in tax.internal):
        S = {u: load_scores(run, u, 'test') for u in tax.internal}
        ph, vh = Router(tax, S).hard()
        out['tree_hard'] = full_metrics(tax, yte, ph); out['tree_hard']['experts_per_input'] = float(vh.mean())
        out['routing_visits_hard'] = {u: float(v) for u, v in visit_fractions(tax, S).items()}
        pred['tree_hard'] = ph
    pf = load_scores(run, 'flat', 'test').argmax(1)
    out['flat'] = full_metrics(tax, yte, pf); out['flat']['experts_per_input'] = 1.0
    pred['flat'] = pf
    np.savez_compressed(f'{run}/treetm_predictions.npz', **pred)
    json.dump(out, open(f'{run}/treetm_eval.json', 'w'), indent=1)
    return out


def visit_fractions(tax, S):
    """Fraction of test inputs that hard routing sends through each internal node."""
    n = len(next(iter(S.values())))
    frac = {u: np.zeros(n, bool) for u in tax.internal}
    node = np.array(['root'] * n, dtype=object)
    active = np.ones(n, bool)
    while active.any():
        nxt = node.copy()
        for u in set(node[active]):
            m = active & (node == u)
            frac[u] |= m
            nxt[m] = np.array(tax.children[u], dtype=object)[S[u][m].argmax(1)]
        node = nxt
        active = np.array([x in tax.children for x in node])
    return {u: v.mean() for u, v in frac.items()}


if __name__ == '__main__':
    ds = sys.argv[1]
    for run in sys.argv[2:]:
        o = evaluate_run(ds, run)
        for k in ('tree_hard', 'flat'):
            if k in o:
                r = o[k]
                print(f"{run} {k:9s} acc {r['accuracy']:.4f} bal {r['balanced_accuracy']:.4f} "
                      f"F1 {r['macro_f1']:.4f} hF {r['hF']:.4f} top {r['level1']:.4f} experts {r['experts_per_input']:.3f}")
