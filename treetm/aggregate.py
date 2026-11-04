"""Aggregate per-seed results of one dataset: mean and sample standard
deviation over seeds for every model and metric, Welch t-tests over seeds
(TreeTM hard routing versus every other model), and
per-seed exact McNemar tests on leaf-level correctness (TreeTM hard vs flat).

Usage: python aggregate.py <dataset>
Writes results/<dataset>/summary.json
"""
import glob
import json
import sys
import numpy as np
from scipy import stats

METRICS = ['accuracy', 'balanced_accuracy', 'macro_precision', 'macro_f1', 'hP', 'hR', 'hF', 'level1']
EXTRA = ('experts_per_input', 'fit_s', 'predict_s_per_input', 'novel')


def mcnemar(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {'a_only': b, 'b_only': c, 'p_exact': float(p)}


def main(ds):
    runs = sorted(glob.glob(f'results/{ds}/main/seed*/treetm_eval.json'))
    tm = {}
    for f in runs:
        e = json.load(open(f))
        for k in ('tree_hard', 'tree_soft', 'tree_gated', 'flat'):
            tm.setdefault(k, []).append(e[k])
    for var, pre in (('leafbal', 'leaf_'), ('budget', 'budget_')):
        for f in sorted(glob.glob(f'results/{ds}/{var}/seed*/treetm_eval.json')):
            e = json.load(open(f))
            for k in ('tree_hard', 'tree_soft', 'tree_gated'):
                tm.setdefault(pre + k[5:], []).append(e[k])
    base = {}
    for f in sorted(glob.glob(f'results/{ds}/baselines/seed*/*.json')):
        name = f.split('/')[-1][:-5]
        base.setdefault(name, []).append(json.load(open(f)))
    models = {**tm, **base}
    out = {'n_seeds': {k: len(v) for k, v in models.items()}, 'mean': {}, 'std': {}}
    for k, rs in models.items():
        out['mean'][k] = {m: float(np.mean([r[m] for r in rs])) for m in METRICS if m in rs[0]}
        out['std'][k] = {m: float(np.std([r[m] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
                         for m in METRICS if m in rs[0]}
        for extra in ('experts_per_input', 'fit_s', 'predict_s_per_input'):
            if extra in rs[0]:
                out['mean'][k][extra] = float(np.mean([r[extra] for r in rs]))
    # Welch t-tests over seeds (training runs are independent; all share one split)
    out['welch_t'] = {}
    for ref, other in [('tree_hard', k) for k in models if k != 'tree_hard'] + \
                      [('leaf_hard', k) for k in models if k not in ('leaf_hard',) and 'leaf_hard' in models] + \
                      [('budget_hard', k) for k in ('flat', 'tree_hard') if 'budget_hard' in models] + \
                      [('budget_gated', k) for k in ('flat',) if 'budget_gated' in models]:
        if len(models[ref]) < 2 or len(models[other]) < 2:
            continue
        key = other if ref == 'tree_hard' else f'{ref}_vs_{other}'
        out['welch_t'][key] = {}
        for m in METRICS:
            if m not in models[other][0]:
                continue
            a = [r[m] for r in models[ref]]
            b = [r[m] for r in models[other]]
            t = stats.ttest_ind(a, b, equal_var=False)
            out['welch_t'][key][m] = {'mean_diff': float(np.mean(a) - np.mean(b)), 'p': float(t.pvalue),
                                      'n': [len(a), len(b)]}
    # per-seed McNemar, TreeTM hard vs flat TM
    for var in ('main', 'leafbal', 'budget'):
        out[f'mcnemar_hard_vs_flat_{var}'] = []
        for f in sorted(glob.glob(f'results/{ds}/{var}/seed*/treetm_predictions.npz')):
            z = np.load(f)
            out[f'mcnemar_hard_vs_flat_{var}'].append(dict(mcnemar(z['tree_hard'] == z['y'], z['flat'] == z['y']), run=f))
    json.dump(out, open(f'results/{ds}/summary.json', 'w'), indent=1)
    for k in models:
        mm, ss = out['mean'][k], out['std'][k]
        print(f"{k:14s} n={len(models[k])} " + ' '.join(f"{m[:6]} {mm[m]:.4f}±{ss[m]:.4f}" for m in
                                                        ('accuracy', 'balanced_accuracy', 'macro_f1', 'hF')))


if __name__ == '__main__':
    main(sys.argv[1])
