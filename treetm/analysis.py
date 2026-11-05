"""Additional analyses requested by the first internal review.
  * validation-set accuracy of TreeTM-C (hard) and the flat TM (final epoch)
  * NSL-KDD: test inputs misrouted at the root, by leaf (mean over seeds)
  * leaf error rate conditioned on a correct first level for both models
  * NSL-KDD: false-alarm rate on normal records, detection rate of known attacks
  * root validation accuracy over the last scored epochs
  * ALeRCE BHRF hierarchical F-measure
Writes results/analysis.json"""
import glob
import json
import numpy as np
from taxonomy import Taxonomy
from evaluate import Router, hier_prf, load_scores

out = {}
for ds in ('nslkdd', 'alerce'):
    meta = json.load(open(f'work/{ds}/meta.json')); tax = Taxonomy(meta['taxonomy'])
    P = np.load(f'work/{ds}/prepared.npz'); yva, yte = P['yva'], P['yte']
    lv1 = lambda y: np.array(tax.level_label(y, 1))
    o = {'val': [], 'cond_leaf_err': [], 'root_val_curve': []}
    for var in ('main', 'leafbal', 'budget'):
        for run in sorted(glob.glob(f'results/{ds}/{var}/seed*')):
            if not glob.glob(f'{run}/treetm_eval.json'):
                continue
            nodes = tax.internal + ['flat']
            T = {u: json.load(open(f'{run}/{u}_hist.json'))['cfg']['T'] for u in nodes}
            Sv = {u: load_scores(run, u, 'val') for u in nodes}
            ph = Router(tax, Sv, T).hard()[0]; pf = Sv['flat'].argmax(1)
            o['val'].append({'variant': var, 'run': run,
                             'tree_acc': float((ph == yva).mean()), 'flat_acc': float((pf == yva).mean()),
                             'tree_level1': float((lv1(ph) == lv1(yva)).mean()),
                             'flat_level1': float((lv1(pf) == lv1(yva)).mean()),
                             'tree_hF': hier_prf(tax, yva, ph)['hF'], 'flat_hF': hier_prf(tax, yva, pf)['hF']})
            z = np.load(f'{run}/treetm_predictions.npz')
            t, f, y = z['tree_hard'], z['flat'], z['y']
            both = (lv1(t) == lv1(y)) & (lv1(f) == lv1(y))
            o['cond_leaf_err'].append({'variant': var, 'run': run, 'n_both_level1_correct': int(both.sum()),
                                       'tree': float((t[both] != y[both]).mean()), 'flat': float((f[both] != y[both]).mean())})
            h = json.load(open(f'{run}/root_hist.json'))
            o['root_val_curve'].append({'variant': var, 'run': run,
                                        'epochs': [e['epoch'] for e in h['epochs'] if 'node_val_acc' in e][-3:],
                                        'acc': [e['node_val_acc'] for e in h['epochs'] if 'node_val_acc' in e][-3:]})
            fh = json.load(open(f'{run}/flat_hist.json'))
            o['root_val_curve'][-1]['flat_acc'] = [e['node_val_acc'] for e in fh['epochs'] if 'node_val_acc' in e][-3:]
            if ds == 'nslkdd':
                normal = y == tax.leaf_index['normal']
                o.setdefault('alarms', []).append({
                    'variant': var, 'run': run,
                    'tree_false_alarm': float((lv1(t[normal]) == 'attack').mean()),
                    'flat_false_alarm': float((lv1(f[normal]) == 'attack').mean()),
                    'tree_detect_known': float((lv1(t[~normal]) == 'attack').mean()),
                    'flat_detect_known': float((lv1(f[~normal]) == 'attack').mean())})
                mis = {}
                for li, l in enumerate(tax.leaves):
                    m = y == li
                    if m.sum():
                        mis[l] = {'n': int(m.sum()), 'tree_wrong_level1': int((lv1(t[m]) != lv1(y[m])).sum()),
                                  'flat_wrong_level1': int((lv1(f[m]) != lv1(y[m])).sum())}
                o.setdefault('misrouted_by_leaf', []).append({'variant': var, 'run': run, 'by_leaf': mis})
    out[ds] = o
pred = np.load('../alerce_prep/results/brf/brf_pred_leaf.npy')
tax = Taxonomy(json.load(open('work/alerce/meta.json'))['taxonomy'])
out['alerce']['bhrf_hF'] = hier_prf(tax, np.load('work/alerce/prepared.npz')['yte'], pred)['hF']
json.dump(out, open('results/analysis.json', 'w'), indent=1)


def mean(var, ds, key, field):
    v = [r[field] for r in out[ds][key] if r['variant'] == var]
    return float(np.mean(v)) if v else None


for ds in out:
    print(ds, 'val tree/flat acc', mean('main', ds, 'val', 'tree_acc'), mean('main', ds, 'val', 'flat_acc'),
          'level1', mean('main', ds, 'val', 'tree_level1'), mean('main', ds, 'val', 'flat_level1'),
          'cond err', mean('main', ds, 'cond_leaf_err', 'tree'), mean('main', ds, 'cond_leaf_err', 'flat'))
print('alarms C', mean('main', 'nslkdd', 'alarms', 'tree_false_alarm'), 'L', mean('leafbal', 'nslkdd', 'alarms', 'tree_false_alarm'),
      'flat', mean('main', 'nslkdd', 'alarms', 'flat_false_alarm'))
print('bhrf hF', out['alerce']['bhrf_hF'])
