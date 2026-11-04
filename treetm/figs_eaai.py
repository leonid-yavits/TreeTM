"""Figures of the EAAI paper. Legends sit in dedicated GridSpec rows.
Usage: python figs_eaai.py <paper_dir>"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.path.join(sys.argv[1], 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'pdf.fonttype': 42,
                     'axes.titlesize': 9})
DS = [('nslkdd', 'NSL-KDD'), ('alerce', 'ALeRCE'), ('wos', 'WOS')]
COL = {'tree': '#1f5fa8', 'flat': '#3a8f3a'}

# ---------------------------------------------------------------- Fig. A: accuracy versus inference work (clause sweep)
fig = plt.figure(figsize=(7.2, 3.1))
gs = GridSpec(2, 3, height_ratios=[1, 0.14], hspace=0.75, wspace=0.38)
handles = None
for i, (d, name) in enumerate(DS):
    ax = fig.add_subplot(gs[0, i])
    pts = []
    runs = sorted(glob.glob(f'results/{d}/sweep/C*')) + [f'results/{d}/main/seed0']
    for r in runs:
        if not (os.path.exists(f'{r}/treetm_eval.json') and os.path.exists(f'{r}/treetm_cost.json')):
            continue
        e, c = json.load(open(f'{r}/treetm_eval.json')), json.load(open(f'{r}/treetm_cost.json'))
        C = json.load(open(f'{r}/root_hist.json'))['cfg']['clauses']
        pts.append((C, c['work']['dense_literal_tests_tree'], e['tree_hard']['hF'],
                    c['work']['dense_literal_tests_flat'], e['flat']['hF']))
    pts.sort()
    if pts:
        P = np.array(pts, dtype=float)
        h1, = ax.plot(P[:, 1] / 1e6, P[:, 2], 'o-', color=COL['tree'], lw=1.3, ms=4, label='TreeTM-C, hard routing')
        h2, = ax.plot(P[:, 3] / 1e6, P[:, 4], 's--', color=COL['flat'], lw=1.3, ms=4, label='Flat TM')
        handles = [h1, h2]
        ax.set_xscale('log')
        from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        lo, hi = P[:, [2, 4]].min(), P[:, [2, 4]].max()
        pad = 0.12 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(name)
    ax.set_xlabel('Literal tests ($10^6$)')
    if i == 0:
        ax.set_ylabel('Hierarchical F-measure')
if handles:
    lax = fig.add_subplot(gs[1, :]); lax.axis('off')
    lax.legend(handles=handles, loc='center', ncol=2, frameon=False, fontsize=8)
fig.savefig(f'{OUT}/fig_tradeoff.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/fig_tradeoff.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------- Fig. B: errors by depth of the first wrong decision
fig = plt.figure(figsize=(7.2, 3.0))
gs = GridSpec(2, 3, height_ratios=[1, 0.15], hspace=0.75, wspace=0.32)
models = [('main', 'tree_hard', 'TreeTM-C, hard', '#1f5fa8'), ('leafbal', 'tree_hard', 'TreeTM-L, hard', '#d1702a'),
          ('main', 'flat', 'Flat TM', '#3a8f3a')]
hs = []
for i, (d, name) in enumerate(DS):
    ax = fig.add_subplot(gs[0, i])
    depth = None
    for j, (var, k, lab, col) in enumerate(models):
        ev = [json.load(open(f)) for f in sorted(glob.glob(f'results/{d}/{var}/seed*/treetm_eval.json'))]
        if not ev:
            continue
        ntest = len(np.load(f'work/{d}/prepared.npz')['yte'])
        depth = max(int(x[5:]) for e in ev for x in e[k]['errors_by_first_wrong_depth'])
        vals = np.array([[e[k]['errors_by_first_wrong_depth'].get(f'depth{t}', 0) / ntest for t in range(1, depth + 1)]
                         for e in ev])
        x = np.arange(1, depth + 1) + (j - 1) * 0.26
        b = ax.bar(x, 100 * vals.mean(0), 0.26, yerr=100 * vals.std(0, ddof=1) if len(ev) > 1 else None,
                   color=col, capsize=2, label=lab)
        if i == 0:
            hs.append(b)
    if depth:
        ax.set_xticks(range(1, depth + 1))
    ax.set_title(name)
    ax.set_xlabel('Depth of first wrong decision')
    ax.tick_params(axis='x', which='both', labelsize=8)
    if i == 0:
        ax.set_ylabel('Test errors (% of inputs)')
lax = fig.add_subplot(gs[1, :]); lax.axis('off')
lax.legend(handles=hs, loc='center', ncol=3, frameon=False, fontsize=8)
fig.savefig(f'{OUT}/fig_errors.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/fig_errors.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('figures written to', OUT)
