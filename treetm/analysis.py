"""NSL-KDD: test records of each leaf that the first decision sends to the wrong
top-level class (normal or attack), for TreeTM (results/nslkdd/leafbal) and the
flat TM with the same C, per seed.
Usage: python analysis.py     Writes results/analysis.json"""
import glob
import json
import numpy as np
from taxonomy import Taxonomy

ds = 'nslkdd'
tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
lv1 = lambda y: np.array(tax.level_label(y, 1))
rows = []
for run in sorted(glob.glob(f'results/{ds}/leafbal/seed*/')):
    z = np.load(f'{run}/treetm_predictions.npz')
    t, f, y = z['tree_hard'], z['flat'], z['y']
    mis = {}
    for li, l in enumerate(tax.leaves):
        m = y == li
        if m.sum():
            mis[l] = {'n': int(m.sum()), 'tree_wrong_level1': int((lv1(t[m]) != lv1(y[m])).sum()),
                      'flat_wrong_level1': int((lv1(f[m]) != lv1(y[m])).sum())}
    rows.append({'variant': 'leafbal', 'run': run.rstrip('/'), 'by_leaf': mis})
json.dump({ds: {'misrouted_by_leaf': rows}}, open('results/analysis.json', 'w'), indent=1)
for leaf in ('guess_passwd', 'warezmaster'):
    print(leaf, 'TreeTM', np.mean([r['by_leaf'][leaf]['tree_wrong_level1'] for r in rows]),
          'flat', np.mean([r['by_leaf'][leaf]['flat_wrong_level1'] for r in rows]))
