"""Combine per-epoch node scores into end-to-end results, select the epoch on
the validation set, report test metrics and inference cost, draw figures."""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data import LEAVES, TOP, TREE, PARENT
from metrics import report
from sklearn.metrics import f1_score

RUN = sys.argv[1] if len(sys.argv) > 1 else 'results/main'
P = np.load('work/prepared.npz')
yva, yte = P['yva'], P['yte']
top_of = np.array([TOP.index(PARENT[c]) for c in LEAVES])
child_leaf = {p: np.array([LEAVES.index(c) for c in TREE[p]]) for p in TOP}


def load(name):
    z = np.load(os.path.join(RUN, f'{name}_scores.npz'))
    h = json.load(open(os.path.join(RUN, f'{name}_hist.json')))
    return z['val'].astype(np.float64), z['test'].astype(np.float64), h


S, H = {}, {}
for n in ['root', 'Transient', 'Stochastic', 'Periodic', 'flat']:
    if os.path.exists(os.path.join(RUN, f'{n}_scores.npz')):
        v, t, h = load(n)
        S[n] = {'val': v, 'test': t}
        H[n] = h
tree_nodes = ['root'] + TOP
E_tree = min(S[n]['val'].shape[0] for n in tree_nodes) if all(n in S for n in tree_nodes) else 0
E_flat = S['flat']['val'].shape[0] if 'flat' in S else 0
T = {n: H[n]['cfg']['T'] for n in H}


def softmax(z, tau):
    z = z / tau
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def route_hard(split, ep):
    r = S['root'][split][ep]
    top = r.argmax(1)
    leaf = np.empty(len(top), int)
    for pi, p in enumerate(TOP):
        m = top == pi
        leaf[m] = child_leaf[p][S[p][split][ep][m].argmax(1)]
    return top, leaf


def route_soft(split, ep, alpha, delta=None):
    """ALeRCE-style product of top and child probabilities.
    If delta is given, a sample whose root margin (top1-top2)/T is >= delta is
    routed hard (one child); otherwise its two best subtrees are evaluated."""
    r = S['root'][split][ep]
    pr = softmax(r, alpha * T['root'])
    probs = np.zeros((r.shape[0], len(LEAVES)))
    for pi, p in enumerate(TOP):
        pc = softmax(S[p][split][ep], alpha * T[p])
        probs[:, child_leaf[p]] = pc * pr[:, [pi]]
    n_children = np.full(r.shape[0], 3.0)
    if delta is not None:
        srt = np.sort(r, axis=1)
        margin = (srt[:, -1] - srt[:, -2]) / T['root']
        confident = margin >= delta
        second = np.argsort(r, axis=1)[:, -2]
        top = r.argmax(1)
        keep = np.zeros_like(probs, dtype=bool)
        for pi, p in enumerate(TOP):
            keep[:, child_leaf[p]] = ((top == pi) | (~confident & (second == pi)))[:, None]
        probs = np.where(keep, probs, -1)
        n_children = np.where(confident, 1.0, 2.0)
    return top_of[probs.argmax(1)], probs.argmax(1), n_children


def bal(y, p, k):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, p)


curves = {}
if E_tree:
    # Choose the softmax temperature on validation at the final epoch.
    alphas = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    best_alpha = max(alphas, key=lambda a: bal(yva, route_soft('val', E_tree - 1, a)[1], 15))
    for mode in ['hard', 'soft']:
        c = {k: [] for k in ['top_acc', 'top_bal', 'leaf_acc', 'leaf_bal', 'leaf_f1', 'val_leaf_bal']}
        for ep in range(E_tree):
            f = (lambda s: route_hard(s, ep)) if mode == 'hard' else \
                (lambda s: route_soft(s, ep, best_alpha)[:2])
            tt, lt = f('test')
            _, lv = f('val')
            c['top_acc'].append(float((tt == top_of[yte]).mean()))
            c['top_bal'].append(float(bal(top_of[yte], tt, 3)))
            c['leaf_acc'].append(float((lt == yte).mean()))
            c['leaf_bal'].append(float(bal(yte, lt, 15)))
            c['leaf_f1'].append(float(f1_score(yte, lt, average='macro')))
            c['val_leaf_bal'].append(float(bal(yva, lv, 15)))
        curves[f'tree_{mode}'] = c
if E_flat:
    c = {k: [] for k in ['top_acc', 'top_bal', 'leaf_acc', 'leaf_bal', 'leaf_f1', 'val_leaf_bal']}
    for ep in range(E_flat):
        lt = S['flat']['test'][ep].argmax(1)
        lv = S['flat']['val'][ep].argmax(1)
        c['top_acc'].append(float((top_of[lt] == top_of[yte]).mean()))
        c['top_bal'].append(float(bal(top_of[yte], top_of[lt], 3)))
        c['leaf_acc'].append(float((lt == yte).mean()))
        c['leaf_bal'].append(float(bal(yte, lt, 15)))
        c['leaf_f1'].append(float(f1_score(yte, lt, average='macro')))
        c['val_leaf_bal'].append(float(bal(yva, lv, 15)))
    curves['flat'] = c

# Epoch selection on validation leaf balanced accuracy, final test metrics.
final = {}
for name, c in curves.items():
    ep = int(np.argmax(c['val_leaf_bal']))
    if name == 'tree_hard':
        tt, lt = route_hard('test', ep)
    elif name == 'tree_soft':
        tt, lt, _ = route_soft('test', ep, best_alpha)
    else:
        lt = S['flat']['test'][ep].argmax(1); tt = top_of[lt]
    final[name] = {'selected_epoch': ep + 1, 'top': report(top_of[yte], tt, 3),
                   'leaf': report(yte, lt, 15)}
    np.save(os.path.join(RUN, f'pred_{name}.npy'), lt)

# Confidence-gated routing sweep at the hard-routing selected epoch.
gate = []
if E_tree:
    ep = E_tree - 1  # final epoch, same model as the measured cost
    for d in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.01, 2.01]:
        _, lt, nc = route_soft('test', ep, best_alpha, delta=d)
        gate.append({'delta': d, 'mean_children': float(nc.mean()),
                     'leaf_bal': float(bal(yte, lt, 15)), 'leaf_acc': float((lt == yte).mean())})
    _, lt, _ = route_soft('test', ep, best_alpha)
    gate.append({'delta': 'all', 'mean_children': 3.0,
                 'leaf_bal': float(bal(yte, lt, 15)), 'leaf_acc': float((lt == yte).mean())})


# Inference cost of the TMs.
def node_cost(n):
    st = H[n]['final_stats']
    wbits = int(np.ceil(np.log2(st['max_clause_weight'] + 1)))
    return {'clauses': st['total_clauses'], 'nonempty': st['nonempty_clauses'],
            'literals': st['n_literals'], 'included': st['included_literals_total'],
            'mask_bits': st['total_clauses'] * st['n_literals'],
            'weight_bits': st['total_clauses'] * wbits, 'weight_width': wbits,
            'dense_bitops': st['total_clauses'] * st['n_literals']}


cost = {}
if E_tree:
    nc = {n: node_cost(n) for n in tree_nodes}
    tt, _ = route_hard('test', E_tree - 1)  # same epoch as the measured masks
    frac = {p: float((tt == i).mean()) for i, p in enumerate(TOP)}
    exp = lambda key: nc['root'][key] + sum(frac[p] * nc[p][key] for p in TOP)
    allk = lambda key: sum(nc[n][key] for n in tree_nodes)
    cost['tree'] = {
        'per_node': nc, 'routing_fraction_test': frac,
        'storage_bits': allk('mask_bits') + allk('weight_bits'),
        'hard_dense_bitops_mean': exp('dense_bitops'),
        'hard_included_literals_mean': exp('included'),
        'hard_clause_adds_mean': exp('nonempty'),
        'soft_dense_bitops': allk('dense_bitops'),
        'soft_included_literals': allk('included'),
        'soft_clause_adds': allk('nonempty'),
    }
if E_flat:
    f = node_cost('flat')
    cost['flat'] = {'per_node': f, 'storage_bits': f['mask_bits'] + f['weight_bits'],
                    'dense_bitops': f['dense_bitops'], 'included_literals': f['included'],
                    'clause_adds': f['nonempty']}

out = {'best_alpha': best_alpha if E_tree else None, 'final': final,
       'gate_sweep': gate, 'cost': cost, 'curves': curves}
with open(os.path.join(RUN, 'evaluation.json'), 'w') as fh:
    json.dump(out, fh, indent=1)

# ---------------- Figures ----------------
brf_path = 'results/brf/brf_results.json'
brf = json.load(open(brf_path)) if os.path.exists(brf_path) else None
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3})
style = {'tree_hard': ('#1f5fa8', '-', 'Tree of TMs, hard routing'),
         'tree_soft': ('#d1702a', '--', 'Tree of TMs, soft routing'),
         'flat': ('#3a8f3a', ':', 'Flat 15-class TM')}

fig = plt.figure(figsize=(11, 8.4))
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.12], hspace=0.55, wspace=0.22)
panels = [('top_acc', 'Top level (3 classes): accuracy'),
          ('leaf_acc', 'Leaf level (15 classes): accuracy'),
          ('leaf_f1', 'Leaf level: macro F1'),
          ('leaf_bal', 'Leaf level: balanced accuracy (macro recall)')]
BRF_KEY = {'top_acc': ('top', 'accuracy'), 'leaf_acc': ('leaf', 'accuracy'),
           'leaf_f1': ('leaf', 'macro_f1'), 'leaf_bal': ('leaf', 'balanced_accuracy')}
handles = []
for i, (key, title) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    lo = []
    for name, c in curves.items():
        col, ls, lab = style[name]
        h, = ax.plot(np.arange(1, len(c[key]) + 1), c[key], color=col, ls=ls, lw=1.8, label=lab)
        lo.extend(c[key])
        if i == 0:
            handles.append(h)
    if brf:
        lvl, met = BRF_KEY[key]
        v = brf[lvl][met]
        h = ax.axhline(v, color='#555555', lw=1.3, ls='-.', label='BHRF baseline (reproduced)')
        lo.append(v)
        if i == 0:
            handles.append(h)
    lo = np.array(lo)
    pad = 0.06 * (lo.max() - lo.min() + 1e-9)
    ax.set_ylim(lo.min() - pad, lo.max() + pad)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
lax = fig.add_subplot(gs[2, :]); lax.axis('off')
lax.legend(handles=handles, loc='center', ncol=len(handles), frameon=False)
fig.savefig(os.path.join(RUN, 'fig_accuracy_vs_epochs.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Per-expert validation accuracy.
fig = plt.figure(figsize=(7.5, 4.8))
gs = GridSpec(2, 1, height_ratios=[1, 0.12], hspace=0.5)
ax = fig.add_subplot(gs[0])
cols = {'root': '#1f5fa8', 'Transient': '#c0392b', 'Stochastic': '#8e44ad',
        'Periodic': '#16a085', 'flat': '#3a8f3a'}
hs = []
for n in tree_nodes + (['flat'] if 'flat' in H else []):
    if n not in H:
        continue
    acc = [e['node_val_acc'] for e in H[n]['epochs']]
    lab = {'root': 'Root expert (3-way)', 'flat': 'Flat TM (15-way)'}.get(n, f'{n} expert')
    h, = ax.plot(np.arange(1, len(acc) + 1), acc, color=cols[n], lw=1.8,
                 ls=':' if n == 'flat' else '-', label=lab)
    hs.append(h)
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation accuracy (within subtree)')
ax.set_title('Per-expert accuracy on its own subtree')
lax = fig.add_subplot(gs[1]); lax.axis('off')
lax.legend(handles=hs, loc='center', ncol=3, frameon=False, fontsize=9)
fig.savefig(os.path.join(RUN, 'fig_expert_accuracy.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# Accuracy versus compute for gated routing.
if gate:
    fig = plt.figure(figsize=(6.4, 4.6))
    gs = GridSpec(2, 1, height_ratios=[1, 0.12], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    x = [g['mean_children'] for g in gate]; y = [g['leaf_bal'] for g in gate]
    h1, = ax.plot(x, y, 'o-', color='#1f5fa8', label='Confidence-gated routing (sweep)')
    hs = [h1]
    if brf:
        hs.append(ax.axhline(brf['leaf']['balanced_accuracy'], color='#555555', ls='-.',
                             label='BHRF baseline'))
    ax.set_xlabel('Mean child experts evaluated per sample')
    ax.set_ylabel('Leaf balanced accuracy (test)')
    ax.set_title('Accuracy versus child-expert evaluations')
    lax = fig.add_subplot(gs[1]); lax.axis('off')
    lax.legend(handles=hs, loc='center', ncol=2, frameon=False, fontsize=9)
    fig.savefig(os.path.join(RUN, 'fig_gated_tradeoff.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

print(json.dumps({'best_alpha': out['best_alpha'], 'final': {k: {'ep': v['selected_epoch'],
      'top_acc': round(v['top']['accuracy'], 4), 'leaf_acc': round(v['leaf']['accuracy'], 4),
      'leaf_bal': round(v['leaf']['balanced_accuracy'], 4),
      'leaf_P': round(v['leaf']['macro_precision'], 4), 'leaf_R': round(v['leaf']['macro_recall'], 4),
      'leaf_F1': round(v['leaf']['macro_f1'], 4)} for k, v in final.items()}}, indent=1))
print(json.dumps(gate, indent=1))
