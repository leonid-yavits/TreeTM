"""Vector figures for the paper, drawn from results/main/evaluation.json and
the per-node training histories. Legends sit in dedicated GridSpec rows."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = 'figs'
os.makedirs(OUT, exist_ok=True)
ev = json.load(open('results/main/evaluation.json'))
brf = json.load(open('results/brf/brf_results.json'))
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'pdf.fonttype': 42, 'axes.titlesize': 9})
style = {'tree_hard': ('#1f5fa8', '-', 'TM-Tree, hard routing'),
         'tree_soft': ('#d1702a', '--', 'TM-Tree, soft routing'),
         'flat': ('#3a8f3a', ':', 'Flat 15-class TM')}
panels = [('top_acc', '(a) Top level: accuracy', ('top', 'accuracy')),
          ('leaf_acc', '(b) Leaf level: accuracy', ('leaf', 'accuracy')),
          ('leaf_f1', '(c) Leaf level: macro F1', ('leaf', 'macro_f1')),
          ('leaf_bal', '(d) Leaf level: balanced accuracy', ('leaf', 'balanced_accuracy'))]
fig = plt.figure(figsize=(7.0, 5.8))
gs = GridSpec(3, 2, height_ratios=[1, 1, 0.14], hspace=0.75, wspace=0.25)
handles = []
for i, (key, title, (lvl, met)) in enumerate(panels):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    vals = []
    for name in ['tree_hard', 'tree_soft', 'flat']:
        c = ev['curves'][name][key]
        col, ls, lab = style[name]
        h, = ax.plot(np.arange(1, len(c) + 1), c, color=col, ls=ls, lw=1.4, label=lab)
        vals += c
        if i == 0:
            handles.append(h)
    v = brf[lvl][met]
    h = ax.axhline(v, color='#444444', lw=1.1, ls='-.', label='ALeRCE BHRF (reproduced)')
    vals.append(v)
    if i == 0:
        handles.append(h)
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
lax = fig.add_subplot(gs[2, :]); lax.axis('off')
lax.legend(handles=handles, loc='center', ncol=2, frameon=False, fontsize=8)
fig.savefig(f'{OUT}/fig_curves.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_curves.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(3.4, 3.4))
gs = GridSpec(2, 1, height_ratios=[1, 0.24], hspace=0.75)
ax = fig.add_subplot(gs[0])
cols = {'root': '#1f5fa8', 'Transient': '#c0392b', 'Stochastic': '#8e44ad',
        'Periodic': '#16a085', 'flat': '#3a8f3a'}
labels = {'root': 'Root (3-way)', 'Transient': 'Transient (4-way)',
          'Stochastic': 'Stochastic (5-way)', 'Periodic': 'Periodic (6-way)',
          'flat': 'Flat TM (15-way)'}
hs = []
for n in ['root', 'Periodic', 'Stochastic', 'Transient', 'flat']:
    h = json.load(open(f'results/main/{n}_hist.json'))
    acc = [e['node_val_acc'] for e in h['epochs']]
    hh, = ax.plot(np.arange(1, len(acc) + 1), acc, color=cols[n], lw=1.3,
                  ls=':' if n == 'flat' else '-', label=labels[n])
    hs.append(hh)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation accuracy')
lax = fig.add_subplot(gs[1]); lax.axis('off')
lax.legend(handles=hs, loc='center', ncol=2, frameon=False, fontsize=8)
fig.savefig(f'{OUT}/fig_experts.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_experts.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('figures written to', OUT)
