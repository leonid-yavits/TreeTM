"""Figures of the EAAI paper. Legends sit in dedicated GridSpec rows.
Usage: python figs_eaai.py <paper_dir>   (run gen_eaai.py first; it writes cost_ratios.json)"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

PAPER = sys.argv[1]
OUT = os.path.join(PAPER, 'figs')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.titlesize': 9})
REL = json.load(open(os.path.join(PAPER, 'cost_ratios.json')))
DS = [('nslkdd', 'NSL-KDD', '#1f5fa8'), ('alerce', 'ALeRCE', '#3a8f3a')]
MEAS = [('dense', 'Dense\nliteral tests', False), ('incl', 'Included\nliterals', False),
        ('cpu', 'CPU time', False), ('cyc', 'AP(TM)\ncycles', True), ('energy', 'AP(TM)\nenergy', True)]

# ---------------------------------------------------------------- inference cost of TreeTM relative to the flat TM
fig = plt.figure(figsize=(6.0, 2.9))
gs = GridSpec(2, 1, height_ratios=[1, 0.12], hspace=0.55)
ax = fig.add_subplot(gs[0])
x = np.arange(len(MEAS)); w = 0.36
for i, (d, name, col) in enumerate(DS):
    for j, (k, _, pred) in enumerate(MEAS):
        v = 100 * REL[d][k]
        ax.bar(x[j] + (i - 0.5) * w, v, w, color=col if not pred else 'white', edgecolor=col,
               hatch='////' if pred else None, linewidth=1.0)
        ax.text(x[j] + (i - 0.5) * w, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=7.5)
ax.axhline(100, color='0.35', lw=0.8, ls='--')
ax.text(len(MEAS) - 0.5, 101.5, 'flat TM', ha='right', va='bottom', fontsize=7.5, color='0.35')
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in MEAS])
ax.set_ylabel('TreeTM cost (% of flat TM)')
ax.set_ylim(0, 112); ax.set_xlim(-0.6, len(MEAS) - 0.4)
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
lg = fig.add_subplot(gs[1]); lg.axis('off')
handles = [Patch(facecolor=c, edgecolor=c, label=n) for _, n, c in DS] + \
          [Patch(facecolor='white', edgecolor='0.3', hatch='////', label='AP(TM) model prediction')]
lg.legend(handles=handles, loc='center', ncol=3, frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, 'fig_cost.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_cost.png'), bbox_inches='tight', dpi=200)
print('fig_cost written')
