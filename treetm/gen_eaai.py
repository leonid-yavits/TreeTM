"""Generate numbers.tex and the result tables of the EAAI paper from the
result files, so text and tables cannot drift from the data.
Usage: python gen_eaai.py <paper_dir>"""
import glob
import json
import os
import sys
import numpy as np

PAPER = sys.argv[1]
os.makedirs(os.path.join(PAPER, 'tables'), exist_ok=True)
DS = [('nslkdd', 'N', 'NSL-KDD'), ('alerce', 'A', 'ALeRCE'), ('wos', 'W', 'WOS')]
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


meta = {d: json.load(open(f'work/{d}/meta.json')) for d, _, _ in DS}
summ = {d: json.load(open(f'results/{d}/summary.json')) for d, _, _ in DS}
TAGS = {'tree_hard': 'Hard', 'tree_soft': 'Soft', 'tree_gated': 'Gated', 'flat': 'Flat',
        'leaf_hard': 'LHard', 'leaf_soft': 'LSoft', 'leaf_gated': 'LGated',
        'budget_hard': 'BHard', 'budget_soft': 'BSoft', 'budget_gated': 'BGated',
        'flat_tm_tuned': 'FlatTuned', 'flat_rf': 'Rf', 'lcpn_rf_hard': 'LcpnHard', 'lcpn_rf_soft': 'LcpnSoft', 'flat_hgb': 'Hgb', 'flat_mlp': 'Mlp'}
MET = {'accuracy': 'Acc', 'balanced_accuracy': 'Bal', 'macro_precision': 'P', 'macro_f1': 'F', 'hF': 'hF',
       'level1': 'Lone'}

# ---------------------------------------------------------------- datasets
from taxonomy import Taxonomy
rows = []
for d, X, name in DS:
    t = Taxonomy(meta[d]['taxonomy'])
    put(f'{X}NLeaves', len(t.leaves)); put(f'{X}NInternal', len(t.internal)); put(f'{X}Depth', t.depth)
    put(f'{X}NBits', meta[d]['n_bits']); put(f'{X}NTrain', meta[d]['n_train'])
    put(f'{X}NVal', meta[d]['n_val']); put(f'{X}NTest', meta[d]['n_test'])
    put(f'{X}Seeds', summ[d]['n_seeds']['tree_hard'])
    cnt = meta[d]['class_counts_train']
    put(f'{X}MinTrain', min(cnt.values())); put(f'{X}MaxTrain', max(cnt.values()))
ma = json.load(open('../alerce_prep/work/meta.json'))
put('ANSamples', ma['n_samples']); put('ANFeatures', ma['n_features'])
put('WNSamples', meta['wos']['n_train'] + meta['wos']['n_val'] + meta['wos']['n_test']); put('WNLeaves', len(Taxonomy(meta['wos']['taxonomy']).leaves))
put('NslNovel', meta['nslkdd']['n_test_novel']); put('NslValPct', 10)
put('NslSeeds', summ['nslkdd']['n_seeds']['tree_hard']); put('ASeeds', summ['alerce']['n_seeds']['tree_hard'])
put('WSeeds', summ['wos']['n_seeds']['tree_hard'])

feat = {'nslkdd': '\\makecell{38 numeric,\\\\ 3 categorical}', 'alerce': '152 numeric', 'wos': '2{,}000 words'}
dom = {'nslkdd': 'Network intrusion', 'alerce': 'Astronomical alerts', 'wos': 'Document routing'}
with open(os.path.join(PAPER, 'tables', 'tab_datasets.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Datasets and taxonomies. Train, validation, and test '
            'sizes are numbers of records. Leaf sizes give the smallest and largest leaf class in the training set.}\n'
            '\\label{tab:data}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccc}\n\\toprule\n'
            '& NSL-KDD & ALeRCE & WOS\\\\\n\\midrule\n')
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

# ---------------------------------------------------------------- config
for d, X, _ in DS:
    put(f'{X}PBudgetRoot', json.load(open(f'results/{d}/budget/seed0/root_hist.json'))['cfg']['per_class'])
cfg = {}
for d, X, _ in DS:
    h = json.load(open(glob.glob(f'results/{d}/main/seed0/root_hist.json')[0]))['cfg']
    cfg[d] = h
    put(f'{X}C', h['clauses']); put(f'{X}T', h['T']); put(f'{X}S', float(h['s']), 0 if float(h['s']).is_integer() else 1)
    put(f'{X}P', h['per_class']); put(f'{X}E', h['epochs'])
with open(os.path.join(PAPER, 'tables', 'tab_config.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{\\tm configuration per dataset. All experts and the flat '
            '\\tm of a dataset use the same values, except the root of \\tmt-B, which draws '
            + f"{m['NPBudgetRoot']}, {m['APBudgetRoot']}, and {m['WPBudgetRoot']} samples per child and epoch.}}\n" + '\\label{tab:config}\n\\small\n\\begin{tabular}{lccc}\n'
            '\\toprule\n& NSL-KDD & ALeRCE & WOS\\\\\n\\midrule\n')
    for lab, key in [('Clauses per class $C$', 'C'), ('Threshold $T$', 'T'), ('Specificity $s$', 'S'),
                     ('Samples per class and epoch $P$', 'P'), ('Epochs $E$', 'E'), ('Seeds', 'Seeds')]:
        f.write(lab + ' & ' + ' & '.join(m[f'{X}{key}'] for _, X, _ in DS) + '\\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# ---------------------------------------------------------------- main results
for d, X, _ in DS:
    for k, tag in TAGS.items():
        if k not in summ[d]['mean']:
            continue
        for mm, mt in MET.items():
            if mm in summ[d]['mean'][k]:
                put(f'{X}{tag}{mt}', summ[d]['mean'][k][mm]); put(f'{X}{tag}{mt}Sd', summ[d]['std'][k][mm])
        if 'experts_per_input' in summ[d]['mean'][k]:
            put(f'{X}{tag}Exp', summ[d]['mean'][k]['experts_per_input'], 2)

ROWS = [('tree_hard', '\\tmt-C, hard routing'), ('tree_gated', '\\tmt-C, gated routing'),
        ('tree_soft', '\\tmt-C, soft routing'),
        ('leaf_hard', '\\tmt-L, hard routing'), ('leaf_gated', '\\tmt-L, gated routing'),
        ('budget_hard', '\\tmt-B, hard routing'), ('budget_gated', '\\tmt-B, gated routing'),
        ('flat', 'Flat \\tm'), ('flat_tm_tuned', 'Flat \\tm, tuned $C$'),
        None, ('flat_rf', 'RF'), ('lcpn_rf_hard', 'LCPN-RF, hard'), ('lcpn_rf_soft', 'LCPN-RF, soft'),
        ('flat_hgb', 'HGB'), ('flat_mlp', 'MLP')]


AN = json.load(open('results/analysis.json'))
BH = json.load(open('../alerce_prep/results/brf/brf_results.json'))['leaf']
BH = {'accuracy': BH['accuracy'], 'balanced_accuracy': BH['balanced_accuracy'], 'macro_f1': BH['macro_f1'],
      'hF': AN['alerce']['bhrf_hF']}


def best_of(d, mm):
    b = max(summ[d]['mean'][q][mm] for q in summ[d]['mean'] if mm in summ[d]['mean'][q])
    return max(b, BH[mm]) if d == 'alerce' else b


def cell(d, k, mm):
    if k not in summ[d]['mean'] or mm not in summ[d]['mean'][k]:
        return '--'
    mu, sd = summ[d]['mean'][k][mm], summ[d]['std'][k][mm]
    best = best_of(d, mm)
    s = f'\\pmstd{{{mu:.3f}}}{{{sd:.3f}}}'
    return f'\\textbf{{{mu:.3f}}}\\,{{\\scriptsize$\\pm$\\,{sd:.3f}}}' if abs(mu - best) < 5e-4 else s


with open(os.path.join(PAPER, 'tables', 'tab_main.tex'), 'w') as f:
    f.write('\\begin{table*}[!p]\n\\centering\n\\caption{Test results, mean $\\pm$ standard deviation over '
            'seeds. \\tmt-C and \\tmt-L use child-balanced and leaf-balanced resampling. \\tmt-B is \\tmt-C with a budget-matched root (Section~\\ref{subsec:errors}). The single-seed \\brf is included in the bolding. Acc: leaf accuracy; '
            'Bal: balanced accuracy; F1: macro F1; hF: hierarchical F-measure; Exp.: experts evaluated per input. Flat \\tm, tuned $C$: clauses per class selected on the validation set (1{,}000 on NSL-KDD, 200 on WOS; on ALeRCE, validation selects the $C$ of Table~\\ref{tab:config}). '
            'Bold: best mean per column.}\n\\label{tab:main}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\renewcommand{\\arraystretch}{0.87}\n'
            '\\begin{tabular}{lccccc}\n\\toprule\n')
    for d, X, name in DS:
        f.write(f'\\multicolumn{{6}}{{l}}{{\\textbf{{{name}}}}}\\\\\n')
        f.write('Model & Acc & Bal & F1 & hF & Exp.\\\\\n\\midrule\n')
        for r in ROWS:
            if r is None:
                f.write('\\midrule\n'); continue
            k, lab = r
            if k not in summ[d]['mean']:
                continue
            e = summ[d]['mean'][k].get('experts_per_input')
            f.write(lab + ' & ' + ' & '.join(cell(d, k, mm) for mm in ('accuracy', 'balanced_accuracy',
                                                                       'macro_f1', 'hF'))
                    + ' & ' + (f'{e:.2f}' if e is not None else '--') + '\\\\\n')
        if d == 'alerce':
            b = json.load(open('../alerce_prep/results/brf/brf_results.json'))['leaf']
            bc = lambda mm: (f'\\textbf{{{BH[mm]:.3f}}}' if abs(BH[mm] - best_of('alerce', mm)) < 5e-4 else f'{BH[mm]:.3f}')
            f.write('ALeRCE \\brf (one seed) & ' + ' & '.join(bc(mm) for mm in ('accuracy', 'balanced_accuracy', 'macro_f1', 'hF'))
                    + ' & --\\\\\n')
        f.write('\\midrule\n' if d != 'wos' else '\\bottomrule\n')
    f.write('\\end{tabular}\n\\end{table*}\n')

# ---------------------------------------------------------------- statistics
for d, X, _ in DS:
    w = summ[d]['welch_t']
    for key, tag in [('flat', 'Flat'), ('leaf_hard_vs_flat', 'LFlat'), ('budget_hard_vs_flat', 'BFlat'),
                     ('budget_gated_vs_flat', 'BGFlat'), ('budget_hard_vs_tree_hard', 'BC')]:
        if key in w:
            for mm, mt in MET.items():
                if mm in w[key]:
                    pv = w[key][mm]['p']
                    put(f'{X}P{tag}{mt}', 'p<0.001' if pv < 0.001 else f'p={pv:.3f}')
                    put(f'{X}D{tag}{mt}', w[key][mm]['mean_diff'], 3)
    for var, tag in [('main', 'C'), ('leafbal', 'L'), ('budget', 'B')]:
        mc = summ[d].get(f'mcnemar_hard_vs_flat_{var}', [])
        put(f'{X}Mc{tag}Sig', sum(x['p_exact'] < 0.05 for x in mc)); put(f'{X}Mc{tag}N', len(mc))
        put(f'{X}Mc{tag}TreeBetter', sum(x['p_exact'] < 0.05 and x['a_only'] > x['b_only'] for x in mc))
        put(f'{X}Mc{tag}FlatBetter', sum(x['p_exact'] < 0.05 and x['a_only'] < x['b_only'] for x in mc))

# ---------------------------------------------------------------- errors by depth, novel attacks, alpha/delta
for d, X, _ in DS:
    tot = {}
    for var, tag in [('main', 'C'), ('leafbal', 'L'), ('budget', 'B')]:
        ev = [json.load(open(f)) for f in sorted(glob.glob(f'results/{d}/{var}/seed*/treetm_eval.json'))]
        if not ev:
            continue
        for k, kt in [('tree_hard', 'Hard'), ('flat', 'Flat')]:
            e = [x[k]['errors_by_first_wrong_depth'] for x in ev]
            n = [sum(x.values()) for x in e]
            put(f'{X}{tag}{kt}ErrDepthOne', float(np.mean([x.get('depth1', 0) / max(t, 1) for x, t in zip(e, n)])) * 100, 0)
        if d == 'nslkdd':
            put(f'NNovRoot{tag}', float(np.mean([x['novel_detected_as_attack_root'] for x in ev])) * 100, 1)
            put(f'NNovFlat', float(np.mean([x['novel_detected_as_attack_flat'] for x in ev])) * 100, 1)
        al = sorted({x['alpha'] for x in ev}); de = sorted({x['delta'] for x in ev})
        put(f'{X}Alphas{tag}', ', '.join(str(a) for a in al)); put(f'{X}Deltas{tag}', ', '.join(str(a) for a in de))

# ---------------------------------------------------------------- cost (seed 0, child-balanced)
with open(os.path.join(PAPER, 'tables', 'tab_cost.tex'), 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Inference cost per input of \\tmt-C with hard routing '
            'and of the flat \\tm (seed 0). Literal tests and included literals are counted on the trained '
            'models with the test-set routing fractions. CPU time: bit-packed NumPy implementation. '
            '\\aptm values are predictions of the analytical model (blue).}\n\\label{tab:cost}\n\\scriptsize\n'
            '\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lcccccc}\n\\toprule\n'
            '& \\multicolumn{2}{c}{NSL-KDD} & \\multicolumn{2}{c}{ALeRCE} & \\multicolumn{2}{c}{WOS}\\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
            '& Tree & Flat & Tree & Flat & Tree & Flat\\\\\n\\midrule\n')
    cs = {d: json.load(open(f'results/{d}/main/seed0/treetm_cost.json')) for d, _, _ in DS}
    def row(lab, ft, ff, fmt, pred=False):
        cells = []
        for d, _, _ in DS:
            for fn in (ft, ff):
                v = fn(cs[d]); s = fmt(v)
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
    for d, X, _ in DS:
        c = cs[d]; w = c['work']; a = c['ap_tm']
        put(f'{X}RedDense', pct(1 - w['dense_literal_tests_tree'] / w['dense_literal_tests_flat']))
        put(f'{X}RedIncl', pct(1 - w['included_literals_tree'] / w['included_literals_flat']))
        put(f'{X}CpuSpeedup', c['cpu_seconds_per_input']['flat'] / c['cpu_seconds_per_input']['tree_hard'], 1)
        put(f'{X}ApCycRed', pct(1 - a['tree_hard_expected']['cycles'] / a['flat']['cycles']))
        put(f'{X}ApCycRedIncl', pct(1 - a['tree_hard_expected']['cycles_incl_reduction'] / a['flat']['cycles_incl_reduction']))
        put(f'{X}ApERed', pct(1 - a['tree_hard_expected']['energy_nJ'] / a['flat']['energy_nJ']))
        put(f'{X}ClausesTree', w['clauses_stored_tree']); put(f'{X}ClausesFlat', w['clauses_stored_flat'])
        put(f'{X}ExpTree', w['expected_experts_per_input'], 2)

# ---------------------------------------------------------------- ranges and extras
red = [1 - cs[d]['work']['dense_literal_tests_tree'] / cs[d]['work']['dense_literal_tests_flat'] for d, _, _ in DS]
spd = [cs[d]['cpu_seconds_per_input']['flat'] / cs[d]['cpu_seconds_per_input']['tree_hard'] for d, _, _ in DS]
ered = [1 - cs[d]['ap_tm']['tree_hard_expected']['energy_nJ'] / cs[d]['ap_tm']['flat']['energy_nJ'] for d, _, _ in DS]
put('RedDenseMin', pct(min(red))); put('RedDenseMax', pct(max(red)))
for d, X, Ct in [('nslkdd', 'N', 1000), ('wos', 'W', 200)]:
    w = cs[d]['work']; Cm = json.load(open(f'results/{d}/main/seed0/flat_hist.json'))['cfg']['clauses']
    put(f'{X}RedDenseTuned', pct(1 - w['dense_literal_tests_tree'] / (w['dense_literal_tests_flat'] * Ct / Cm)))
    put(f'{X}CTuned', Ct)
put('SpeedMin', min(spd), 1); put('SpeedMax', max(spd), 1)
put('ERedMin', pct(min(ered))); put('ERedMax', pct(max(ered)))
mn = meta['nslkdd']
put('NTestGuess', mn['class_counts_test']['guess_passwd']); put('NTrainGuess', mn['class_counts_train']['guess_passwd'])
rb = json.load(open('results/nslkdd/abl_root2000_child/seed0/treetm_eval.json'))
put('NRootBigLone', rb['tree_hard']['level1'])
cl = json.load(open('results/nslkdd/main/seed0/clauses_dos_neptune.json'))[0]
put('NClTpr', cl['tpr']); put('NClFpr', cl['fpr'], 4); put('NClLits', cl['n_literals'])
cl2 = json.load(open('results/nslkdd/main/seed0/clauses_dos_neptune.json'))[1]
put('NClTprB', cl2['tpr']); put('NClFprB', cl2['fpr']); put('NClLitsB', cl2['n_literals'])
for d, X in [('nslkdd', 'N'), ('wos', 'W')]:
    clj = json.load(open(f'results/{d}/clause_length.json'))
    put(f'{X}ClMedTree', clj['tree_median'], 0); put(f'{X}ClMedFlat', clj['flat_median'], 0)
put('NClTprPct', 100 * cl['tpr'], 1); put('NClTprBPct', 100 * cl2['tpr'], 1); put('NClFprBPct', 100 * cl2['fpr'], 1)
ca = json.load(open('../alerce_prep/results/clauses_root_Transient.json'))
put('AClMinLits', min(c['n_literals'] for c in ca)); put('AClMaxLits', max(c['n_literals'] for c in ca))
# per-depth error counts (mean over seeds) for the claim about deeper experts
for d, X, _ in DS:
    ev = [json.load(open(f)) for f in sorted(glob.glob(f'results/{d}/main/seed*/treetm_eval.json'))]
    dep = max(int(k[5:]) for e in ev for k in e['tree_hard']['errors_by_first_wrong_depth'])
    for t in range(2, dep + 1):
        put(f'{X}ErrDeepTree{"I" * (t - 1)}', float(np.mean([e['tree_hard']['errors_by_first_wrong_depth'].get(f'depth{t}', 0) for e in ev])), 0)
        put(f'{X}ErrDeepFlat{"I" * (t - 1)}', float(np.mean([e['flat']['errors_by_first_wrong_depth'].get(f'depth{t}', 0) for e in ev])), 0)

# ---------------------------------------------------------------- review round 1 analyses
def amean(ds, key, field, var='main'):
    v = [r[field] for r in AN[ds][key] if r['variant'] == var]
    return float(np.mean(v))
for d, X, _ in DS:
    put(f'{X}ValTreeAcc', amean(d, 'val', 'tree_acc')); put(f'{X}ValFlatAcc', amean(d, 'val', 'flat_acc'))
    put(f'{X}ValTreeLone', amean(d, 'val', 'tree_level1')); put(f'{X}ValFlatLone', amean(d, 'val', 'flat_level1'))
    put(f'{X}CondTree', amean(d, 'cond_leaf_err', 'tree')); put(f'{X}CondFlat', amean(d, 'cond_leaf_err', 'flat'))
    c = cs[d]['ap_tm']
    put(f'{X}ApMaxTree', c['tree_hard_expected']['cycles_max']); put(f'{X}ApFlatCyc', c['flat']['cycles'])
mis = [r for r in AN['nslkdd']['misrouted_by_leaf'] if r['variant'] == 'main']
for leaf, tag in [('guess_passwd', 'Guess'), ('warezmaster', 'Warez'), ('back', 'Back')]:
    put(f'NMis{tag}Tree', float(np.mean([r['by_leaf'][leaf]['tree_wrong_level1'] for r in mis])), 0)
    put(f'NMis{tag}Flat', float(np.mean([r['by_leaf'][leaf]['flat_wrong_level1'] for r in mis])), 0)
    misb = [r for r in AN['nslkdd']['misrouted_by_leaf'] if r['variant'] == 'budget']
    put(f'NMis{tag}Budget', float(np.mean([r['by_leaf'][leaf]['tree_wrong_level1'] for r in misb])), 1)
put('NTestWarez', mn['class_counts_test']['warezmaster']); put('NTestBack', mn['class_counts_test']['back'])
put('NFalseAlarmC', 100 * amean('nslkdd', 'alarms', 'tree_false_alarm'), 1)
put('NFalseAlarmL', 100 * amean('nslkdd', 'alarms', 'tree_false_alarm', 'leafbal'), 1)
put('NFalseAlarmFlat', 100 * amean('nslkdd', 'alarms', 'flat_false_alarm'), 1)
put('NFalseAlarmB', 100 * amean('nslkdd', 'alarms', 'tree_false_alarm', 'budget'), 1)
for var, tag in [('main', 'C'), ('leafbal', 'L'), ('budget', 'B')]:
    put(f'NDetKnown{tag}', 100 * amean('nslkdd', 'alarms', 'tree_detect_known', var), 1)
put('NDetKnownFlat', 100 * amean('nslkdd', 'alarms', 'flat_detect_known'), 1)
nb = [json.load(open(f)) for f in sorted(glob.glob('results/nslkdd/baselines/seed*/novel.json'))]
for k, tag in [('flat_rf', 'Rf'), ('flat_hgb', 'Hgb'), ('flat_mlp', 'Mlp'), ('lcpn_rf_hard', 'Lcpn')]:
    put(f'NNov{tag}', 100 * float(np.mean([x[k]['novel_detected_as_attack'] for x in nb])), 1)
    put(f'NDetKnown{tag}', 100 * float(np.mean([x[k]['known_detected_as_attack'] for x in nb])), 1)
put('NNovBaseN', len(nb))
for leaf, tag in [('guess_passwd', 'Guess'), ('warezmaster', 'Warez'), ('back', 'Back')]:
    misl = [r for r in AN['nslkdd']['misrouted_by_leaf'] if r['variant'] == 'leafbal']
    put(f'NMis{tag}Leaf', float(np.mean([r['by_leaf'][leaf]['tree_wrong_level1'] for r in misl])), 0)
rc = [r for r in AN['wos']['root_val_curve'] if r['variant'] == 'main']
put('WRootValA', float(np.mean([r['acc'][0] for r in rc]))); put('WRootValC', float(np.mean([r['acc'][-1] for r in rc])))
rb2 = [r for r in AN['wos']['root_val_curve'] if r['variant'] == 'budget']
put('WBRootValA', float(np.mean([r['acc'][0] for r in rb2]))); put('WBRootValC', float(np.mean([r['acc'][-1] for r in rb2])))
put('WBValLone', amean('wos', 'val', 'tree_level1', 'budget'))
gc = json.load(open('results/wos/budget/gated_cost.json'))
put('WBGatedRed', pct(float(np.mean([g['reduction'] for g in gc]))))
bl_g = [json.load(open(f))['tree_gated']['level1'] for f in sorted(glob.glob('results/wos/budget/seed*/treetm_eval.json'))]
put('WBGatedLone', float(np.mean(bl_g)))
put('BhrfBal', BH['balanced_accuracy']); put('BhrfhF', BH['hF'])
e0 = json.load(open('results/nslkdd/main/seed0/treetm_eval.json'))
put('NRootBigSeedZero', e0['tree_hard']['level1'])
put('NRootBigNovA', 100 * e0['novel_detected_as_attack_root'], 1)
put('NRootBigNovB', 100 * rb['novel_detected_as_attack_root'], 1)
tw = Taxonomy(meta['wos']['taxonomy'])
hw = json.load(open('results/wos/main/seed0/root_hist.json'))['cfg']['per_class']
hb = json.load(open('results/wos/budget/seed0/root_hist.json'))['cfg']['per_class']
put('WPRootA', hw * len(tw.children['root'])); put('WPFlat', hw * len(tw.leaves)); put('WPRootB', hb * len(tw.children['root']))

for d, X, _ in DS:
    if 'budget_hard' in summ[d]['mean']:
        bl = [json.load(open(f))['tree_hard']['level1'] for f in sorted(glob.glob(f'results/{d}/budget/seed*/treetm_eval.json'))]
        put(f'{X}BudgetLone', float(np.mean(bl)))

with open(os.path.join(PAPER, 'numbers.tex'), 'w') as fh:
    fh.write('% Auto-generated by gen_eaai.py from the result files. Do not edit.\n')
    for k, v in m.items():
        fh.write(f'\\newcommand{{\\n{k}}}{{{v}}}\n')
print(len(m), 'macros')
