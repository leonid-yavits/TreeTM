"""Generate numbers.tex and the result tables of the EAAI paper from the
result files, so text and tables cannot drift from the data.
Usage: python gen_eaai.py <paper_dir>

The paper reports one TreeTM configuration per dataset (Table 2 of the paper):
NSL-KDD uses leaf-balanced resampling (results/nslkdd/leafbal), and ALeRCE uses
child-balanced resampling with a budget-matched root (results/alerce/budget),
which the ALeRCE validation set selects among the trained configurations."""
import glob
import json
import os
import sys
import numpy as np
from scipy.stats import ttest_ind
from taxonomy import Taxonomy

PAPER = sys.argv[1]
os.makedirs(os.path.join(PAPER, 'tables'), exist_ok=True)
DS = [('nslkdd', 'N', 'NSL-KDD'), ('alerce', 'A', 'ALeRCE')]
VAR = {'nslkdd': 'leafbal', 'alerce': 'budget'}          # run directory of the reported TreeTM
KEY = {'nslkdd': 'leaf_hard', 'alerce': 'budget_hard'}   # its key in summary.json
MET = [('accuracy', 'Acc'), ('balanced_accuracy', 'Bal'), ('macro_f1', 'F'), ('hF', 'hF'), ('level1', 'Lone')]
m = {}


def put(name, val, nd=3):
    if isinstance(val, (float, np.floating)) and nd == 0 and abs(val) >= 1000:
        val = f'{val:,.0f}'.replace(',', '{,}')
    elif isinstance(val, (float, np.floating)):
        val = f'{val:.{nd}f}'
    elif isinstance(val, (int, np.integer)):
        val = f'{int(val):,}'.replace(',', '{,}')
    m[name] = val


def pct(x):
    return f'{100 * x:.0f}'


def pstr(p):
    return '$p<0.001$' if p < 0.001 else f'$p={p:.3f}$'


meta = {d: json.load(open(f'work/{d}/meta.json')) for d, _, _ in DS}
summ = {d: json.load(open(f'results/{d}/summary.json')) for d, _, _ in DS}
seeds = {d: summ[d]['n_seeds']['tree_hard'] for d, _, _ in DS}

# ---------------------------------------------------------------- datasets
for d, X, name in DS:
    t = Taxonomy(meta[d]['taxonomy'])
    put(f'{X}NLeaves', len(t.leaves)); put(f'{X}NInternal', len(t.internal)); put(f'{X}Depth', t.depth)
    put(f'{X}NBits', meta[d]['n_bits']); put(f'{X}NTrain', meta[d]['n_train'])
    put(f'{X}NVal', meta[d]['n_val']); put(f'{X}NTest', meta[d]['n_test'])
    put(f'{X}Seeds', seeds[d]); put(f'{X}NRootKids', len(t.children['root']))
    cnt = meta[d]['class_counts_train']
    put(f'{X}MinTrain', min(cnt.values())); put(f'{X}MaxTrain', max(cnt.values()))
import re
ma = meta['alerce']
put('ANSamples', ma['n_train'] + ma['n_val'] + ma['n_test'])
put('ANFeatures', len({re.split(r'(>=|<=|>|<|==| is | missing|=)', b)[0].strip() for b in ma['bit_names']}))
mn = meta['nslkdd']
put('NslNovel', mn['n_test_novel'])
put('NTestGuess', mn['class_counts_test']['guess_passwd']); put('NTrainGuess', mn['class_counts_train']['guess_passwd'])
put('NTestWarez', mn['class_counts_test']['warezmaster']); put('NTrainWarez', mn['class_counts_train']['warezmaster'])

feat = {'nslkdd': '\\makecell{38 numeric,\\\\ 3 categorical}', 'alerce': '152 numeric'}
dom = {'nslkdd': 'Network intrusion', 'alerce': 'Astronomical alerts'}
with open(os.path.join(PAPER, 'tables', 'tab_datasets.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Datasets and taxonomies. Train, validation, and test '
            'sizes are numbers of records. Leaf sizes give the smallest and largest leaf class in the training set.}\n'
            '\\label{tab:data}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcc}\n\\toprule\n'
            '& NSL-KDD & ALeRCE\\\\\n\\midrule\n')
    f.write('Domain & ' + ' & '.join(dom[d] for d, _, _ in DS) + '\\\\\n')
    f.write('Input features & ' + ' & '.join(feat[d] for d, _, _ in DS) + '\\\\\n')
    f.write('Boolean features (bits) & ' + ' & '.join(m[f'{X}NBits'] for _, X, _ in DS) + '\\\\\n')
    f.write('Taxonomy depth & ' + ' & '.join(m[f'{X}Depth'] for _, X, _ in DS) + '\\\\\n')
    f.write('Internal nodes (experts) & ' + ' & '.join(m[f'{X}NInternal'] for _, X, _ in DS) + '\\\\\n')
    f.write('Leaf classes & ' + ' & '.join(m[f'{X}NLeaves'] for _, X, _ in DS) + '\\\\\n')
    f.write('Leaf size in training & ' + ' & '.join(f"{m[f'{X}MinTrain']}--{m[f'{X}MaxTrain']}" for _, X, _ in DS) + '\\\\\n')
    f.write('Train / validation / test & ' + ' & '.join(
        f"\\makecell{{{m[f'{X}NTrain']} /\\\\ {m[f'{X}NVal']} / {m[f'{X}NTest']}}}" for _, X, _ in DS) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# ---------------------------------------------------------------- configuration
cfg = {}
for d, X, _ in DS:
    h = json.load(open(f'results/{d}/main/seed0/root_hist.json'))['cfg']; cfg[d] = h
    put(f'{X}C', h['clauses']); put(f'{X}T', h['T']); put(f'{X}S', float(h['s']), 0 if float(h['s']).is_integer() else 1)
    put(f'{X}P', h['per_class']); put(f'{X}E', h['epochs'])
    put(f'{X}PRoot', json.load(open(f'results/{d}/{VAR[d]}/seed0/root_hist.json'))['cfg']['per_class'])
    t = Taxonomy(meta[d]['taxonomy'])
    put(f'{X}PRootEpoch', json.load(open(f'results/{d}/{VAR[d]}/seed0/root_hist.json'))['cfg']['per_class'] * len(t.children['root']))
    put(f'{X}PFlatEpoch', h['per_class'] * len(t.leaves))
RS = {'nslkdd': 'leaf-balanced', 'alerce': 'child-balanced'}
with open(os.path.join(PAPER, 'tables', 'tab_config.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{\\tmt configuration per dataset. All experts and the flat '
            '\\tm of a dataset use the same $C$, $T$, and $s$.}\n\\label{tab:config}\n\\small\n\\begin{tabular}{lcc}\n'
            '\\toprule\n& NSL-KDD & ALeRCE\\\\\n\\midrule\n')
    for lab, key in [('Clauses per class $C$', 'C'), ('Threshold $T$', 'T'), ('Specificity $s$', 'S'),
                     ('Epochs $E$', 'E'), ('Samples per class and epoch $P$', 'P'),
                     ('Root samples per child and epoch $P_{\\mathrm{root}}$', 'PRoot')]:
        f.write(lab + ' & ' + ' & '.join(m[f'{X}{key}'] for _, X, _ in DS) + '\\\\\n')
    f.write('Resampling of the upper experts & ' + ' & '.join(RS[d] for d, _, _ in DS) + '\\\\\n')
    f.write('Seeds & ' + ' & '.join(m[f'{X}Seeds'] for _, X, _ in DS) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# ---------------------------------------------------------------- per-seed values and tests
def per_seed(d, var, k):
    return [json.load(open(f'results/{d}/{var}/seed{s}/treetm_eval.json'))[k] for s in range(seeds[d])]


def tuned(d):
    fs = sorted(glob.glob(f'results/{d}/baselines/seed*/flat_tm_tuned.json'))
    return [json.load(open(f)) for f in fs]


VALS = {}
for d, X, _ in DS:
    VALS[d] = {'tree': per_seed(d, VAR[d], 'tree_hard'), 'flat': per_seed(d, 'main', 'flat')}
    if tuned(d):
        VALS[d]['tuned'] = tuned(d)
    for k, tag in [('tree', 'Tree'), ('flat', 'Flat'), ('tuned', 'Tuned')]:
        if k not in VALS[d]:
            continue
        for mm, mt in MET:
            v = [r[mm] for r in VALS[d][k]]
            put(f'{X}{tag}{mt}', float(np.mean(v))); put(f'{X}{tag}{mt}Sd', float(np.std(v, ddof=1)))
    put(f'{X}TreeExp', float(np.mean([r['experts_per_input'] for r in VALS[d]['tree']])), 2)
    for ref, rt in [('flat', 'Flat'), ('tuned', 'Tuned')]:
        if ref not in VALS[d]:
            continue
        for mm, mt in MET:
            a = [r[mm] for r in VALS[d]['tree']]; b = [r[mm] for r in VALS[d][ref]]
            p = ttest_ind(a, b, equal_var=False).pvalue
            put(f'{X}P{rt}{mt}', pstr(p)); m[f'_{X}{rt}{mt}_p'] = p
            put(f'{X}D{rt}{mt}', float(np.mean(a) - np.mean(b)), 3)
    put(f'{X}CTuned', VALS[d]['tuned'][0]['clauses_per_class'] if 'tuned' in VALS[d] else cfg[d]['clauses'])

# ---------------------------------------------------------------- main table
with open(os.path.join(PAPER, 'tables', 'tab_main.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Test results, mean $\\pm$ standard deviation over seeds. '
            'Acc: leaf accuracy; Bal: balanced accuracy; F1: macro F1; hF: hierarchical F-measure; '
            'Top: accuracy at the first level of the taxonomy. '
            'On NSL-KDD, the flat \\tm with $C{=}' + m['NCTuned'] + '$ uses the clause budget selected on the validation set. '
            'On ALeRCE, validation selects the $C$ of Table~\\ref{tab:config}. Bold: best mean per column. '
            '$^{\\ast}$: \\tmt is significantly better than the flat \\tm with the same $C$ (Welch\'s $t$-test, $p<0.05$).}\n'
            '\\label{tab:main}\n\\footnotesize\n\\setlength{\\tabcolsep}{1.6pt}\n'
            '\\begin{tabular}{lccccc}\n\\toprule\n'
            'Model & Acc & Bal & F1 & hF & Top\\\\\n')
    for d, X, name in DS:
        f.write(f'\\midrule\n\\multicolumn{{6}}{{l}}{{\\textit{{{name}}}}}\\\\\n')
        rows = [('tree', '\\tmt', m[f'{X}TreeExp']), ('flat', f'Flat \\tm, $C{{=}}{m[f"{X}C"]}$', '1.00')]
        if 'tuned' in VALS[d]:
            rows.append(('tuned', f'Flat \\tm, $C{{=}}{m[f"{X}CTuned"]}$', '1.00'))
        best = {mm: max(np.mean([r[mm] for r in VALS[d][k]]) for k, _, _ in rows) for mm, _ in MET}
        for k, lab, e in rows:
            cells = []
            for mm, mt in MET:
                v = [r[mm] for r in VALS[d][k]]; mu, sd = np.mean(v), np.std(v, ddof=1)
                s = f'{mu:.3f}'
                if abs(mu - best[mm]) < 5e-4:
                    s = f'\\textbf{{{s}}}'
                star = (k == 'tree' and m[f'_{X}Flat{mt}_p'] < 0.05 and np.mean(v) > np.mean([r[mm] for r in VALS[d]['flat']]))
                cells.append(s + ('$^{\\ast}$' if star else '') + f'\\,{{\\scriptsize$\\pm$\\,{sd:.3f}}}')
            f.write(lab + ' & ' + ' & '.join(cells) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# ---------------------------------------------------------------- cost (seed 0)
cs = {d: json.load(open(f'results/{d}/{VAR[d]}/seed0/treetm_cost.json')) for d, _, _ in DS}
with open(os.path.join(PAPER, 'tables', 'tab_cost.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Inference cost per input of \\tmt and of the flat \\tm '
            'with the same $C$ (seed 0). Literal tests and included literals are counted on the trained '
            'models with the test-set routing fractions. CPU time: bit-packed NumPy implementation. '
            '\\aptm values are predictions of the analytical model (blue).}\n\\label{tab:cost}\n\\footnotesize\n'
            '\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\n'
            '& \\multicolumn{2}{c}{NSL-KDD} & \\multicolumn{2}{c}{ALeRCE}\\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n'
            '& \\tmt & Flat & \\tmt & Flat\\\\\n\\midrule\n')

    def row(lab, ft, ff, fmt, pred=False):
        cells = []
        for d, _, _ in DS:
            for fn in (ft, ff):
                s = fmt(fn(cs[d]))
                cells.append(f'\\pred{{{s}}}' if pred else s)
        f.write(lab + ' & ' + ' & '.join(cells) + '\\\\\n')
    row('Experts evaluated', lambda c: c['work']['expected_experts_per_input'], lambda c: 1, lambda v: f'{v:.2f}')
    row('Dense literal tests ($10^6$)', lambda c: c['work']['dense_literal_tests_tree'] / 1e6,
        lambda c: c['work']['dense_literal_tests_flat'] / 1e6, lambda v: f'{v:.2f}')
    row('Included literals ($10^3$)', lambda c: c['work']['included_literals_tree'] / 1e3,
        lambda c: c['work']['included_literals_flat'] / 1e3, lambda v: f'{v:.1f}')
    row('CPU time ($\\mu$s)', lambda c: c['cpu_seconds_per_input']['tree_hard'] * 1e6,
        lambda c: c['cpu_seconds_per_input']['flat'] * 1e6, lambda v: f'{v:.0f}')
    row('\\aptm cycles', lambda c: c['ap_tm']['tree_hard_expected']['cycles'], lambda c: c['ap_tm']['flat']['cycles'],
        lambda v: f'{v:.1f}', True)
    row('\\aptm cycles incl.\\ reduction', lambda c: c['ap_tm']['tree_hard_expected']['cycles_incl_reduction'],
        lambda c: c['ap_tm']['flat']['cycles_incl_reduction'], lambda v: f'{v:.1f}', True)
    row('\\aptm energy (nJ)', lambda c: c['ap_tm']['tree_hard_expected']['energy_nJ'],
        lambda c: c['ap_tm']['flat']['energy_nJ'], lambda v: f'{v:.1f}', True)
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

REL = {}
for d, X, _ in DS:
    c = cs[d]; w = c['work']; a = c['ap_tm']
    REL[d] = {'dense': w['dense_literal_tests_tree'] / w['dense_literal_tests_flat'],
              'incl': w['included_literals_tree'] / w['included_literals_flat'],
              'cpu': c['cpu_seconds_per_input']['tree_hard'] / c['cpu_seconds_per_input']['flat'],
              'cyc': a['tree_hard_expected']['cycles'] / a['flat']['cycles'],
              'cycr': a['tree_hard_expected']['cycles_incl_reduction'] / a['flat']['cycles_incl_reduction'],
              'energy': a['tree_hard_expected']['energy_nJ'] / a['flat']['energy_nJ']}
    put(f'{X}RedDense', pct(1 - REL[d]['dense'])); put(f'{X}RedIncl', pct(1 - REL[d]['incl']))
    put(f'{X}CpuSpeedup', 1 / REL[d]['cpu'], 1)
    put(f'{X}ApCycRed', pct(1 - REL[d]['cyc'])); put(f'{X}ApCycRedIncl', pct(1 - REL[d]['cycr']))
    put(f'{X}ApERed', pct(1 - REL[d]['energy']))
    put(f'{X}ApMaxTree', a['tree_hard_expected']['cycles_max'], 0); put(f'{X}ApFlatCyc', a['flat']['cycles'], 0)
    put(f'{X}ExpTree', w['expected_experts_per_input'], 2)
json.dump(REL, open(os.path.join(PAPER, 'cost_ratios.json'), 'w'), indent=1)
put('RedDenseMin', pct(1 - max(r['dense'] for r in REL.values()))); put('RedDenseMax', pct(1 - min(r['dense'] for r in REL.values())))
put('SpeedMin', min(1 / r['cpu'] for r in REL.values()), 1); put('SpeedMax', max(1 / r['cpu'] for r in REL.values()), 1)
put('ERedMin', pct(1 - max(r['energy'] for r in REL.values()))); put('ERedMax', pct(1 - min(r['energy'] for r in REL.values())))
w = cs['nslkdd']['work']
put('NRedDenseTuned', pct(1 - w['dense_literal_tests_tree'] / (w['dense_literal_tests_flat'] * int(m['NCTuned'].replace('{,}', '')) / cfg['nslkdd']['clauses'])))

# ---------------------------------------------------------------- interpretability (NSL-KDD)
# The DoS expert of the leaf-balanced runs is shared with the child-balanced runs (all its children are leaves).
assert os.path.realpath('results/nslkdd/leafbal/seed0/dos_final_masks.npz') == os.path.realpath('results/nslkdd/main/seed0/dos_final_masks.npz')
cl = json.load(open('results/nslkdd/main/seed0/clauses_dos_neptune.json'))
put('NClTprPct', 100 * cl[0]['tpr'], 1); put('NClFpr', cl[0]['fpr'], 4); put('NClLits', cl[0]['n_literals'])
put('NClTprBPct', 100 * cl[1]['tpr'], 1); put('NClFprBPct', 100 * cl[1]['fpr'], 1); put('NClLitsB', cl[1]['n_literals'])
clj = json.load(open('results/nslkdd/clause_length_leafbal.json'))
put('NClMedTree', clj['tree_median'], 0); put('NClMedFlat', clj['flat_median'], 0)
put('NClMeanTree', clj['tree_mean'], 0); put('NClMeanFlat', clj['flat_mean'], 0)

# ---------------------------------------------------------------- path sizes and NSL-KDD misrouting
for d, X, _ in DS:
    t = Taxonomy(meta[d]['taxonomy'])
    put(f'{X}MaxPathClasses', max(sum(len(t.children[u]) for u, _ in p) for p in t.paths))
AN = json.load(open('results/analysis.json'))
for var, vt in [('leafbal', 'Tree')]:
    mis = [r for r in AN['nslkdd']['misrouted_by_leaf'] if r['variant'] == var]
    for leaf, tag in [('guess_passwd', 'Guess'), ('warezmaster', 'Warez')]:
        put(f'NMis{tag}{vt}', float(np.mean([r['by_leaf'][leaf]['tree_wrong_level1'] for r in mis])), 0)
        put(f'NMis{tag}Flat', float(np.mean([r['by_leaf'][leaf]['flat_wrong_level1'] for r in mis])), 0)

with open(os.path.join(PAPER, 'numbers.tex'), 'w') as fh:
    fh.write('% Auto-generated by gen_eaai.py from the result files. Do not edit.\n')
    for k, v in m.items():
        if not k.startswith('_'):
            fh.write(f'\\newcommand{{\\n{k}}}{{{v}}}\n')
print(len(m), 'macros')
for k in sorted(m):
    if k.startswith(('NP', 'AP', 'ND', 'AD')) and not k.startswith(('NPRoot', 'APRoot', 'NPFlat', 'APFlat')):
        print(k, m[k])
