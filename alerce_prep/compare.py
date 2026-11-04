"""Side-by-side accuracy and inference-cost table: ALeRCE BHRF (reproduced)
versus the tree of TM experts and the flat TM.

Cost conventions (stated so the two model families are not conflated):
* BHRF: one comparison = a 32-bit feature fetch, compare against a threshold,
  and a branch to a child node. Counts are measured with decision_path.
  ALeRCE combines probabilities as P_top x P_child, so all four forests run.
  A hard-routed variant (top forest + one child forest) is also reported.
* Compact BHRF storage model: internal node = 1 B feature index + 4 B float32
  threshold + 4 B child index; leaf = k x 4 B float32 class probabilities.
  The measured scikit-learn pickle size is reported for reference.
* TM: dense bit operations = clauses x literals (one AND/compare per literal
  bit, as in a bit-parallel or associative-processor evaluation); included
  literals = literals actually included (a sparse evaluation needs only these);
  clause additions = non-empty clauses (signed weighted adds into class sums).
  Storage = 1 include bit per literal per clause + one weight per clause.
"""
import json
import numpy as np

brf = json.load(open('results/brf/brf_results.json'))
ev = json.load(open('results/main/evaluation.json'))
fe = json.load(open('results/final_epoch_metrics.json'))  # epoch 50, same model as the cost
c = brf['cost']


def brf_storage(f):
    return f['internal_nodes'] * 9 + f['leaves'] * f['n_classes'] * 4


forests = ['top', 'Transient', 'Stochastic', 'Periodic']
P = np.load('work/prepared.npz')
from data import LEAVES, TOP, PARENT
top_of = np.array([TOP.index(PARENT[l]) for l in LEAVES])
pred = np.load('results/brf/brf_pred_leaf.npy')
frac = {p: float((top_of[pred] == i).mean()) for i, p in enumerate(TOP)}
rows = {}
rows['BHRF (ALeRCE, soft)'] = {
    'leaf': brf['leaf'], 'top': brf['top'],
    'trees_evaluated': sum(c[f]['trees'] for f in forests),
    'comparisons': sum(c[f]['comparisons_per_sample_mean'] for f in forests),
    'storage_bytes_compact': sum(brf_storage(c[f]) for f in forests),
    'pickle_bytes': sum(c[f]['pickle_bytes'] for f in forests),
    'nodes': sum(c[f]['nodes'] for f in forests),
}
rows['BHRF hard-routed (cost only)'] = {
    'comparisons': c['top']['comparisons_per_sample_mean'] +
    sum(frac[p] * c[p]['comparisons_per_sample_mean'] for p in TOP),
}
t = ev['cost']['tree']
for name, key in [('Tree of TMs, hard', 'tree_hard'), ('Tree of TMs, soft', 'tree_soft')]:
    hard = key == 'tree_hard'
    rows[name] = {
        'leaf': fe[key]['leaf'], 'top': fe[key]['top'], 'epoch': 50,
        'dense_bitops': t['hard_dense_bitops_mean'] if hard else t['soft_dense_bitops'],
        'included_literals': t['hard_included_literals_mean'] if hard else t['soft_included_literals'],
        'clause_adds': t['hard_clause_adds_mean'] if hard else t['soft_clause_adds'],
        'storage_bytes': t['storage_bits'] / 8,
    }
if 'flat' in ev['cost']:
    f = ev['cost']['flat']
    rows['Flat TM'] = {
        'leaf': fe['flat']['leaf'], 'top': fe['flat']['top'], 'epoch': 50,
        'dense_bitops': f['dense_bitops'], 'included_literals': f['included_literals'],
        'clause_adds': f['clause_adds'], 'storage_bytes': f['storage_bits'] / 8,
    }
json.dump(rows, open('results/comparison.json', 'w'), indent=1)
for k, v in rows.items():
    out = {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
           if kk not in ('leaf', 'top')}
    if 'leaf' in v:
        out.update({'top_acc': round(v['top']['accuracy'], 4),
                    'leaf_acc': round(v['leaf']['accuracy'], 4),
                    'leaf_bal': round(v['leaf']['balanced_accuracy'], 4),
                    'leaf_P': round(v['leaf']['macro_precision'], 4),
                    'leaf_R': round(v['leaf']['macro_recall'], 4),
                    'leaf_F1': round(v['leaf']['macro_f1'], 4)})
    print(k, json.dumps(out))
print('routing fractions (BHRF top-level predictions):', frac)
