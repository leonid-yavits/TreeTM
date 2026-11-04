"""Generate paper/numbers.tex: every quantitative result quoted in the paper
is a macro written here from the result files, so text, tables and results
cannot drift apart."""
import json
import numpy as np
from data import LEAVES, TOP

R = 'results'
meta = json.load(open('work/meta.json'))
brf = json.load(open(f'{R}/brf/brf_results.json'))
fe = json.load(open(f'{R}/final_epoch_metrics.json'))           # unweighted, epoch 50
few = json.load(open(f'{R}/final_epoch_metrics_weighted.json'))  # weighted ablation, epoch 50
ev = json.load(open(f'{R}/main/evaluation.json'))
ap = json.load(open(f'{R}/ap_cost.json'))
cfg = json.load(open(f'{R}/main/root_hist.json'))['cfg']
cfgw = json.load(open(f'{R}/weighted/root_hist.json'))['cfg']
P = np.load('work/prepared.npz')
yte = P['yte']

m = {}


def put(name, val, nd=3):
    if isinstance(val, float):
        val = f'{val:.{nd}f}'
    elif isinstance(val, int):
        val = f'{val:,}'.replace(',', '{,}')
    m[name] = val


# Dataset and setup
put('NSamples', meta['n_samples']); put('NFeatures', meta['n_features'])
put('NBits', meta['n_bits']); put('NLiterals', 2 * meta['n_bits'])
put('NTrain', meta['n_train']); put('NVal', meta['n_val']); put('NTest', meta['n_test'])
put('NBins', meta['bins'])
for c, n in meta['class_counts'].items():
    put('Count' + c.replace('/', '').replace('-', ''), n)
put('Clauses', cfg['clauses']); put('Thr', cfg['T']); put('Spec', cfg['s'], 0)
put('PerClass', cfg['per_class']); put('Epochs', cfg['epochs'])
put('ThrW', cfgw['T'])

# Accuracy (leaf: 15 classes; top: 3 classes). BHRF top from its top-level forest.
rows = {'Brf': ('brf', None), 'Hard': ('fe', 'tree_hard'), 'Soft': ('fe', 'tree_soft'),
        'Flat': ('fe', 'flat'), 'HardW': ('few', 'tree_hard'), 'SoftW': ('few', 'tree_soft'),
        'FlatW': ('few', 'flat')}
for tag, (src, key) in rows.items():
    r = brf if src == 'brf' else (fe if src == 'fe' else few)[key]
    put(f'TopAcc{tag}', r['top']['accuracy'])
    put(f'LeafAcc{tag}', r['leaf']['accuracy'])
    put(f'LeafBal{tag}', r['leaf']['balanced_accuracy'])
    put(f'LeafP{tag}', r['leaf']['macro_precision'])
    put(f'LeafFone{tag}', r['leaf']['macro_f1'])

# Per-class recall and precision (BHRF, hard tree, flat), unweighted epoch 50
for j, c in enumerate(LEAVES):
    cc = c.replace('/', '').replace('-', '')
    put('Sup' + cc, int((yte == j).sum()))
    for tag, r in [('Brf', fe['BHRF']), ('Hard', fe['tree_hard']), ('Flat', fe['flat'])]:
        put(f'R{tag}{cc}', r['leaf']['per_class_recall'][j], 2)
        put(f'P{tag}{cc}', r['leaf']['per_class_precision'][j], 2)

# Per-expert validation accuracy at the last epoch
for n in ['root', 'Transient', 'Stochastic', 'Periodic', 'flat']:
    h = json.load(open(f'{R}/main/{n}_hist.json'))
    put('ValAcc' + n.capitalize(), h['epochs'][-1]['node_val_acc'])

# Routing and gated sweep
frac = ev['cost']['tree']['routing_fraction_test']
for p in TOP:
    put('Route' + p, 100 * frac[p], 1)
gate = [g for g in ev['gate_sweep'] if g['delta'] != 'all']
put('AlphaSoft', float(ev['best_alpha']), 0)
gv0 = json.load(open(f'{R}/gate_validation.json'))
g1 = [g for g in gate if g['delta'] == gv0['chosen_delta']][0]   # delta chosen on validation
put('GateDelta', float(g1['delta']), 2)
put('GateChildren', g1['mean_children'], 3); put('GateBal', g1['leaf_bal'])
put('GateAcc', g1['leaf_acc'])

# Software-level inference cost (counts per inference)
t, f = ev['cost']['tree'], ev['cost']['flat']
put('BitopsHard', t['hard_dense_bitops_mean'] / 1e6, 1); put('BitopsSoft', t['soft_dense_bitops'] / 1e6, 1)
put('BitopsFlat', f['dense_bitops'] / 1e6, 1)
put('InclHard', t['hard_included_literals_mean'] / 1e3, 1); put('InclSoft', t['soft_included_literals'] / 1e3, 1)
put('InclFlat', f['included_literals'] / 1e3, 1)
put('AddsHard', int(round(t['hard_clause_adds_mean']))); put('AddsSoft', int(t['soft_clause_adds']))
put('AddsFlat', int(f['clause_adds']))
# unweighted clauses: storage = include masks only (weights are all 1)
mb = lambda nodes: sum(ev['cost']['tree']['per_node'][n]['mask_bits'] for n in nodes)
put('StoreTree', mb(['root', 'Transient', 'Stochastic', 'Periodic']) / 8 / 1e6, 2)
from decimal import Decimal, ROUND_HALF_UP
put('StoreFlat', str((Decimal(f['per_node']['mask_bits']) / Decimal(8000000)).quantize(Decimal('0.01'), ROUND_HALF_UP)))
put('IncludedFracTree', 100 * t['soft_included_literals'] / t['soft_dense_bitops'], 2)
put('RedBitops', 100 * (1 - t['hard_dense_bitops_mean'] / f['dense_bitops']), 0)
put('RedIncl', 100 * (1 - t['hard_included_literals_mean'] / f['included_literals']), 0)
put('RedAdds', 100 * (1 - t['hard_clause_adds_mean'] / f['clause_adds']), 0)
ww = [ev['cost']['tree']['per_node'][n]['weight_width'] for n in ['root', 'Transient', 'Stochastic', 'Periodic']]
c = brf['cost']
forests = ['top', 'Transient', 'Stochastic', 'Periodic']
cmp_soft = sum(c[k]['comparisons_per_sample_mean'] for k in forests)
# a hard-routed BHRF routes by the argmax of its top-level forest (brf_prior.py)
bp = json.load(open(f'{R}/brf/brf_prior.json'))
fr_brf = bp['top_forest_routing_fraction_test']
cmp_hard = c['top']['comparisons_per_sample_mean'] + sum(fr_brf[p] * c[p]['comparisons_per_sample_mean'] for p in TOP)
put('BrfTrees', sum(c[k]['trees'] for k in forests)); put('BrfCmpSoft', int(round(cmp_soft)))
put('BrfCmpHard', int(round(cmp_hard))); put('BrfNodes', sum(c[k]['nodes'] for k in forests))
put('BrfCompactMB', sum(c[k]['internal_nodes'] * 9 + c[k]['leaves'] * c[k]['n_classes'] * 4 for k in forests) / 1e6, 2)
put('BrfPickleMB', sum(c[k]['pickle_bytes'] for k in forests) / 1e6, 1)
put('BrfMaxDepth', max(c[k]['max_depth_max'] for k in forests))

# AP(TM) model estimates (predicted, typeset in blue)
put('ApRefCycles', ap['reference_mnist']['cycles']); put('ApRefE', ap['reference_mnist']['energy_nJ'], 1)
put('ApFlatCycles', ap['flat']['cycles']); put('ApFlatE', ap['flat']['energy_nJ'], 1)
put('ApFlatRed', ap['flat']['reduction_max'])
put('ApTreeCycles', ap['tree_hard']['cycles_mean'], 1); put('ApTreeCyclesMax', ap['tree_hard']['cycles_max'])
put('ApTreeE', ap['tree_hard']['energy_nJ_mean'], 1); put('ApTreeRed', ap['tree_hard']['reduction_max_total'])
put('ApERed', 100 * (1 - ap['tree_hard']['energy_nJ_mean'] / ap['flat']['energy_nJ']), 0)
put('ApCycRed', 100 * (1 - ap['tree_hard']['cycles_mean'] / ap['flat']['cycles']), 0)
for n in ['root', 'Transient', 'Stochastic', 'Periodic']:
    put('ApE' + n.capitalize(), ap['per_expert'][n]['energy_nJ'], 1)
    put('ApCyc' + n.capitalize(), ap['per_expert'][n]['cycles'])
    put('ApCycIncl' + n.capitalize(), ap['per_expert'][n]['cycles'] + ap['per_expert'][n]['reduction_max'])
put('ApBitsRow', ap['tree_hard']['bits_per_row_max'])
put('ApTreeCyclesIncl', ap['tree_hard']['cycles_incl_mean'], 1); put('ApTreeCyclesInclMax', ap['tree_hard']['cycles_incl_max'])
put('ApFlatCyclesIncl', ap['flat']['cycles_incl'])
put('ApCycRedIncl', 100 * (1 - ap['tree_hard']['cycles_incl_mean'] / ap['flat']['cycles_incl']), 0)
put('ApRowsTree', ap['tree_hard']['rows_total']); put('ApRowsFlat', ap['flat']['rows'])

cl = json.load(open(f'{R}/clauses_root_Transient.json'))[1]
put('ClTpr', cl['tpr'], 3); put('ClFpr', cl['fpr'], 4); put('ClLits', cl['n_literals'])
put('ClFeats', len(cl['simplified']))
cls3 = json.load(open(f'{R}/clauses_root_Transient.json'))
put('ClMinLits', min(c['n_literals'] for c in cls3)); put('ClMaxLits', max(c['n_literals'] for c in cls3))
# balanced-accuracy gap and the share of the two rarest transient classes
gap = fe['BHRF']['leaf']['balanced_accuracy'] - fe['tree_hard']['leaf']['balanced_accuracy']
iS, iB = LEAVES.index('SLSN'), LEAVES.index('SNIbc')
two = sum(fe['BHRF']['leaf']['per_class_recall'][i] - fe['tree_hard']['leaf']['per_class_recall'][i] for i in (iS, iB)) / 15
put('GapBal', gap); put('GapTwo', two)
put('SupTwo', int((yte == iS).sum() + (yte == iB).sum()))
ntr_slsn = int((P['ytr'] == iS).sum()); put('TrainSLSN', ntr_slsn)
soft_all = [g for g in ev['gate_sweep'] if g['delta'] == 'all'][0]
put('SoftAllBal', soft_all['leaf_bal'])
uc = json.load(open(f'{R}/unclipped_metrics.json'))
for tag, key in [('Hard', 'tree_hard'), ('Soft', 'tree_soft'), ('Flat', 'flat')]:
    put(f'UcLeafAcc{tag}', uc[key]['leaf']['accuracy']); put(f'UcLeafBal{tag}', uc[key]['leaf']['balanced_accuracy'])
    put(f'UcLeafP{tag}', uc[key]['leaf']['macro_precision']); put(f'UcLeafFone{tag}', uc[key]['leaf']['macro_f1'])
    put(f'UcTopAcc{tag}', uc[key]['top']['accuracy'])
cs = json.load(open(f'{R}/clip_stats.json'))
put('SatFlat', 100 * cs['flat']['sat'], 0); put('SatRoot', 100 * cs['root']['sat'], 0); put('TieFlat', 100 * cs['flat']['tie'], 1)
gv = json.load(open(f'{R}/gate_validation.json')); put('GateDeltaVal', float(gv['chosen_delta']), 2)
# prior-corrected operating points (g selected on validation by macro F1)
tp = json.load(open(f'{R}/tm_prior.json'))
for tag, r, g in [('PcBrf', bp['test'], bp['chosen_g']), ('PcHard', tp['hard']['test'], tp['hard']['chosen_g']),
                  ('PcSoft', tp['soft']['test'], tp['soft']['chosen_g'])]:
    put(f'TopAcc{tag}', r['top']['accuracy']); put(f'LeafAcc{tag}', r['leaf']['accuracy'])
    put(f'LeafBal{tag}', r['leaf']['balanced_accuracy']); put(f'LeafP{tag}', r['leaf']['macro_precision'])
    put(f'LeafFone{tag}', r['leaf']['macro_f1']); put(f'G{tag}', float(g), 2)
put('ApArrayTreeMB', sum(ap['per_expert'][n]['rows'] * ap['per_expert'][n]['bits_per_row'] for n in ['root', 'Transient', 'Stochastic', 'Periodic']) / 8 / 1e6, 1)
put('ApArrayFlatMB', ap['flat']['rows'] * ap['flat']['bits_per_row'] / 8 / 1e6, 1)
with open('numbers.tex', 'w') as fh:
    fh.write('% Auto-generated by gen_numbers.py from the result files. Do not edit.\n')
    for k, v in m.items():
        fh.write(f'\\newcommand{{\\n{k}}}{{{v}}}\n')
    fh.write(f'% weight widths (bits) per expert: {ww}\n')
print(len(m), 'macros written')
