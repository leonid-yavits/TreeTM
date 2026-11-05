"""Evaluate trained TreeTM and flat TM runs on any prepared dataset.

Usage: python evaluate.py <dataset> <run_dir> [<run_dir> ...]
For every run (one seed), at the final epoch:
  * hard, soft and confidence-gated routing (alpha and delta chosen on the
    validation set by balanced accuracy; delta = smallest best value),
  * flat TM,
  * leaf metrics (accuracy, balanced accuracy, macro precision/recall/F1),
    per-level accuracy, hierarchical precision/recall/F1 (ancestor sets,
    root excluded), error decomposition by the depth of the first wrong
    routing decision, mean number of experts evaluated per input.
Writes <run_dir>/treetm_eval.json and treetm_predictions.npz.
"""
import json
import os
import sys
import zlib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from taxonomy import Taxonomy
from metrics import report

ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
DELTAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.01, 2.01]


def load_scores(run, u, sp):
    """Class sums of node u on split sp ('val', 'test' or 'novel'): the unclipped
    sums of unclip.py when present, otherwise the clipped sums of the library.
    Exact ties are broken uniformly at random (fixed seed per node and split)
    by adding a perturbation far below one vote, which removes the bias of
    argmax toward the lowest class index."""
    if sp == 'novel':
        f = f'{run}/{u}_novel_uscores.npy'
        s = np.load(f) if os.path.exists(f) else np.load(f'{run}/{u}_novel_scores.npy')
    elif os.path.exists(f'{run}/{u}_uscores.npz'):
        s = np.load(f'{run}/{u}_uscores.npz')[sp]
    else:
        s = np.load(f'{run}/{u}_scores.npz')[sp][-1]
    rng = np.random.default_rng(zlib.crc32(f'{u}/{sp}'.encode()))
    return s.astype(float) + rng.uniform(0, 1e-3, s.shape)


def softmax(z, t):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Router:
    def __init__(self, tax, S, T):
        self.tax, self.S, self.T = tax, S, T   # S[node]: (n, K) class sums

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
                kids = np.array(tax.children[u], dtype=object)[c]
                node[m] = kids
            for i in np.flatnonzero(~done):
                if node[i] in tax.leaf_index:
                    pred[i] = tax.leaf_index[node[i]]
                    done[i] = True
        return pred, visits

    def leaf_probs(self, alpha):
        tax = self.tax
        n = len(next(iter(self.S.values())))
        P = {u: softmax(self.S[u].astype(float), alpha * self.T[u]) for u in tax.internal}
        out = np.ones((n, len(tax.leaves)))
        for li, path in enumerate(tax.paths):
            for u, k in path:
                out[:, li] *= P[u][:, k]
        return out, P

    def gated(self, alpha, delta):
        """At every internal node, follow only the best child when the
        normalized margin of its two largest class sums is >= delta,
        otherwise the two best children; leaves reached are scored by the
        soft path product."""
        tax = self.tax
        n = len(next(iter(self.S.values())))
        probs, _ = self.leaf_probs(alpha)
        reach = {}
        visits = np.zeros(n, int)

        def go(u, mask):
            if not mask.any():
                return
            visits[mask] += 1
            s = self.S[u]
            srt = np.sort(s, 1)
            conf = (srt[:, -1] - srt[:, -2]) / self.T[u] >= delta
            o = np.argsort(-s, 1)
            for k, c in enumerate(tax.children[u]):
                take = mask & ((o[:, 0] == k) | (~conf & (o[:, 1] == k)))
                if c in tax.leaf_index:
                    reach[c] = reach.get(c, np.zeros(n, bool)) | take
                else:
                    go(c, take)
        go('root', np.ones(n, bool))
        keep = np.zeros_like(probs, bool)
        for c, m in reach.items():
            keep[:, tax.leaf_index[c]] = m
        return np.where(keep, probs, -1).argmax(1), visits


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


def first_error_depth(tax, y, p):
    """For wrong predictions: depth (1 = decision at the root) of the first
    routing decision that left the true path."""
    out = {}
    for a, b in zip(y, p):
        if a == b:
            continue
        A, B = tax.ancestors(int(a)), tax.ancestors(int(b))
        d = next(i for i in range(min(len(A), len(B))) if A[i] != B[i]) + 1
        out[d] = out.get(d, 0) + 1
    return {f'depth{d}': v for d, v in sorted(out.items())}


def full_metrics(tax, y, p):
    r = report(y, p, len(tax.leaves))
    r.update(level_acc(tax, y, p))
    r.update(hier_prf(tax, y, p))
    r['errors_by_first_wrong_depth'] = first_error_depth(tax, y, p)
    return r


def evaluate_run(ds, run):
    meta = json.load(open(f'work/{ds}/meta.json'))
    tax = Taxonomy(meta['taxonomy'])
    P = np.load(f'work/{ds}/prepared.npz')
    yva, yte = P['yva'], P['yte']
    T = {u: json.load(open(f'{run}/{u}_hist.json'))['cfg']['T'] for u in tax.internal + ['flat']}
    S = {sp: {u: load_scores(run, u, sp) for u in tax.internal + ['flat']} for sp in ('val', 'test')}
    rv, rt = Router(tax, S['val'], T), Router(tax, S['test'], T)
    alpha = max(ALPHAS, key=lambda a: balanced_accuracy_score(yva, rv.leaf_probs(a)[0].argmax(1)))
    sweep = [(d, balanced_accuracy_score(yva, rv.gated(alpha, d)[0])) for d in DELTAS]
    best = max(b for _, b in sweep)
    delta = min(d for d, b in sweep if b >= best - 1e-12)
    out = {'alpha': alpha, 'delta': delta, 'delta_sweep_val': sweep}
    ph, vh = rt.hard()
    out['tree_hard'] = full_metrics(tax, yte, ph); out['tree_hard']['experts_per_input'] = float(vh.mean())
    ps = rt.leaf_probs(alpha)[0].argmax(1)
    out['tree_soft'] = full_metrics(tax, yte, ps); out['tree_soft']['experts_per_input'] = float(len(tax.internal))
    pg, vg = rt.gated(alpha, delta)
    out['tree_gated'] = full_metrics(tax, yte, pg); out['tree_gated']['experts_per_input'] = float(vg.mean())
    pf = S['test']['flat'].argmax(1)
    out['flat'] = full_metrics(tax, yte, pf); out['flat']['experts_per_input'] = 1.0
    out['routing_visits_hard'] = {u: float(v) for u, v in visit_fractions(tax, S['test']).items()}
    np.savez_compressed(f'{run}/treetm_predictions.npz', tree_hard=ph, tree_soft=ps, tree_gated=pg, flat=pf, y=yte)
    # novel inputs (NSL-KDD attack types unseen in training): fraction routed to 'attack' by the root
    if os.path.exists(f'{run}/root_novel_scores.npy') and 'attack' in tax.children['root']:
        k = tax.children['root'].index('attack')
        out['novel_detected_as_attack_root'] = float((load_scores(run, 'root', 'novel').argmax(1) == k).mean())
        fl = load_scores(run, 'flat', 'novel').argmax(1)
        out['novel_detected_as_attack_flat'] = float((np.array(tax.level_label(fl, 1)) == 'attack').mean())
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
        print(run, 'alpha', o['alpha'], 'delta', o['delta'])
        for k in ('tree_hard', 'tree_soft', 'tree_gated', 'flat'):
            r = o[k]
            print(f"  {k:10s} acc {r['accuracy']:.4f} bal {r['balanced_accuracy']:.4f} P {r['macro_precision']:.4f} "
                  f"F1 {r['macro_f1']:.4f} hF {r['hF']:.4f} experts {r['experts_per_input']:.3f}")
