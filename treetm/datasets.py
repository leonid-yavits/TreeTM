"""Dataset preparation for TreeTM. Each dataset is written to
work/<name>/prepared.npz (Btr/Bva/Bte booleanized bits, Ftr/Fva/Fte float
features for the non-TM baselines, ytr/yva/yte leaf ids) and
work/<name>/meta.json (taxonomy, leaves, sizes, provenance).

Usage:
  python datasets.py alerce   <hier_tm dir>
  python datasets.py nslkdd   <dir with KDDTrain+.txt, KDDTest+.txt, training_attack_types.txt>
  python datasets.py wos      <dir with the extracted WebOfScience/WOS46985 folder>
"""
import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from taxonomy import Taxonomy

MISSING = -999.0
SEED = 0


class Thermometer:
    """Quantile thermometer code (levels 1/8..7/8 of the non-missing training
    values) plus a missing-value bit for features with missing training values.
    A threshold at the training minimum becomes the strict test x > min."""

    def __init__(self, n_bins=8):
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.thresholds, self.strict = [], []
        for j in range(X.shape[1]):
            v = X[:, j][X[:, j] != MISSING]
            th = np.unique(np.quantile(v, qs)) if v.size else np.array([])
            # A threshold at the training minimum would give a constant bit
            # (x >= min).  For sparse features (e.g. counts that are mostly 0)
            # this discards the feature, so it becomes the strict test x > min.
            # A feature that is constant in training gets no threshold bit.
            if v.size and v.max() == v.min():
                th = np.array([])
            strict = th <= (v.min() if v.size else 0)
            self.thresholds.append(th)
            self.strict.append(strict)
        self.has_missing = (X == MISSING).any(axis=0)
        return self

    def names(self, cols):
        return [f'{c} {">" if st else ">="} {t:.6g}'
                for c, ts, sts in zip(cols, self.thresholds, self.strict) for t, st in zip(ts, sts)]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = []
        for j, th in enumerate(self.thresholds):
            v = X[:, j]
            miss = v == MISSING
            if th.size:
                st = self.strict[j][None, :]
                ge = v[:, None] >= th[None, :]
                gt = v[:, None] > th[None, :]
                cols.append(np.where(st, gt, ge) & ~miss[:, None])
            if self.has_missing[j]:
                cols.append(miss[:, None])
        return np.hstack(cols).astype(np.uint8)


def save(name, tax, B, F, y, extra_meta, extra_arrays=None):
    out = os.path.join('work', name)
    os.makedirs(out, exist_ok=True)
    arrays = {f'B{s}': B[s].astype(np.uint8) for s in B}
    arrays.update({f'F{s}': F[s].astype(np.float32) for s in F})
    arrays.update({f'y{s}': y[s].astype(np.int64) for s in y})
    arrays.update(extra_arrays or {})
    np.savez_compressed(os.path.join(out, 'prepared.npz'), **arrays)
    meta = {'name': name, 'taxonomy': tax.tree, 'leaves': tax.leaves, 'n_bits': int(B['tr'].shape[1]),
            'n_train': int(len(y['tr'])), 'n_val': int(len(y['va'])), 'n_test': int(len(y['te'])),
            'class_counts_train': {l: int((y['tr'] == i).sum()) for i, l in enumerate(tax.leaves)},
            'class_counts_test': {l: int((y['te'] == i).sum()) for i, l in enumerate(tax.leaves)}}
    meta.update(extra_meta)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(name, {k: meta[k] for k in ('n_bits', 'n_train', 'n_val', 'n_test')}, 'leaves', len(tax.leaves),
          'internal', len(tax.internal), 'depth', tax.depth)


def alerce(hier_dir):
    sys.path.insert(0, hier_dir)
    from data import LEAVES, TREE
    tax = Taxonomy({p: {l: None for l in ls} for p, ls in TREE.items()})
    assert tax.leaves == LEAVES
    P = np.load(os.path.join(hier_dir, 'work', 'prepared.npz'))
    save('alerce', tax, {s: P[f'B{s}'] for s in ('tr', 'va', 'te')}, {s: P[f'F{s}'] for s in ('tr', 'va', 'te')},
         {s: P[f'y{s}'] for s in ('tr', 'va', 'te')},
         {'bit_names': [(f"{b['feature']} >= {b['threshold']:.6g}" if b['test'] == '>=' else f"{b['feature']} missing")
                        for b in json.load(open(os.path.join(hier_dir, 'booleanizer.json')))['literals'][:P['Btr'].shape[1]]],
          'source': 'ALeRCE light-curve classifier labeled set, Zenodo 10.5281/zenodo.4279623',
          'split': 'stratified 80/20 train/test, 10% of train as validation, seed 0 (first study)'})


NSL_COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'difficulty']
NSL_CAT = ['protocol_type', 'service', 'flag']


def nslkdd(d):
    import pandas as pd
    tr = pd.read_csv(os.path.join(d, 'KDDTrain+.txt'), header=None, names=NSL_COLS)
    te = pd.read_csv(os.path.join(d, 'KDDTest+.txt'), header=None, names=NSL_COLS)
    cat_of = dict(l.split() for l in open(os.path.join(d, 'training_attack_types.txt')) if l.strip())
    groups = {}
    for a, c in sorted(cat_of.items()):
        groups.setdefault(c, {})[a] = None
    tax = Taxonomy({'normal': None, 'attack': {c: groups[c] for c in ('dos', 'probe', 'r2l', 'u2r')}})
    lid = tax.leaf_index
    # validation: stratified 10% of KDDTrain+ (stratified on the category, since
    # some attack types have only 2 to 4 records); test: official KDDTest+.
    cat = tr['label'].map(lambda a: 'normal' if a == 'normal' else cat_of[a])
    tr_i, va_i = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=cat, random_state=SEED)
    known = te['label'].isin(lid)
    num = [c for c in NSL_COLS[:-2] if c not in NSL_CAT]
    cats = {c: sorted(tr[c].iloc[tr_i].unique()) for c in NSL_CAT}

    def feats(df):
        Xn = df[num].to_numpy(np.float64)
        oh = [np.stack([(df[c] == v).to_numpy() for v in cats[c]], 1) for c in NSL_CAT]
        return Xn, np.hstack(oh).astype(np.uint8)

    Xn_tr, O_tr = feats(tr.iloc[tr_i]); Xn_va, O_va = feats(tr.iloc[va_i])
    Xn_te, O_te = feats(te[known]); Xn_nov, O_nov = feats(te[~known])
    th = Thermometer(8).fit(Xn_tr)
    bit_names = th.names(num)
    bit_names += [f'{c} == {v}' for c in NSL_CAT for v in cats[c]]
    B = lambda Xn, O: np.hstack([th.transform(Xn), O])
    Bs = {'tr': B(Xn_tr, O_tr), 'va': B(Xn_va, O_va), 'te': B(Xn_te, O_te)}
    Fs = {'tr': np.hstack([Xn_tr, O_tr]), 'va': np.hstack([Xn_va, O_va]), 'te': np.hstack([Xn_te, O_te])}
    ys = {'tr': tr['label'].iloc[tr_i].map(lid).to_numpy(), 'va': tr['label'].iloc[va_i].map(lid).to_numpy(),
          'te': te['label'][known].map(lid).to_numpy()}
    nov_labels = te['label'][~known].to_numpy()
    save('nslkdd', tax, Bs, Fs, ys,
         {'source': 'NSL-KDD (Tavallaee et al., CISDA 2009), KDDTrain+ / KDDTest+ from the GitHub mirror '
                    'defcom17/NSL_KDD; attack categories from the KDD Cup 1999 training_attack_types file (UCI)',
          'split': 'train/validation: stratified 90/10 split of KDDTrain+ on the category, seed 0; '
                   'test: KDDTest+ records of attack types seen in training',
          'n_test_novel': int((~known).sum()), 'novel_types': sorted(set(nov_labels)),
          'categorical_values_from_train': {c: [str(v) for v in cats[c]] for c in NSL_CAT},
          'n_thermometer_bits': int(th.transform(Xn_tr[:1]).shape[1]), 'n_onehot_bits': int(O_tr.shape[1]),
          'bit_names': bit_names},
         {'Bnov': np.hstack([th.transform(Xn_nov), O_nov]).astype(np.uint8),
          'Fnov': np.hstack([Xn_nov, O_nov]).astype(np.float32)})


def wos(d, vocab=2000):
    from sklearn.feature_extraction.text import CountVectorizer
    X = [l.rstrip('\n') for l in open(os.path.join(d, 'X.txt'), encoding='utf-8', errors='replace')]
    YL1 = np.loadtxt(os.path.join(d, 'YL1.txt'), dtype=int)
    YL2 = np.loadtxt(os.path.join(d, 'YL2.txt'), dtype=int)
    assert len(X) == len(YL1) == len(YL2)
    pairs = sorted(set(zip(YL1.tolist(), YL2.tolist())))
    tree = {}
    for a, b in pairs:
        tree.setdefault(f'D{a}', {})[f'D{a}A{b}'] = None
    tax = Taxonomy(tree)
    y = np.array([tax.leaf_index[f'D{a}A{b}'] for a, b in zip(YL1, YL2)])
    idx = np.arange(len(X))
    tr_i, te_i = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    tr_i, va_i = train_test_split(tr_i, test_size=0.1, stratify=y[tr_i], random_state=SEED)
    cv = CountVectorizer(binary=True, max_features=vocab, stop_words='english', lowercase=True,
                         token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b')
    cv.fit([X[i] for i in tr_i])
    Bs = {s: (cv.transform([X[i] for i in ii]) > 0).toarray().astype(np.uint8)
          for s, ii in (('tr', tr_i), ('va', va_i), ('te', te_i))}
    save('wos', tax, Bs, {s: Bs[s].astype(np.float32) for s in Bs},
         {'tr': y[tr_i], 'va': y[va_i], 'te': y[te_i]},
         {'source': 'Web of Science WOS-46985 (Kowsari et al., ICMLA 2017), Mendeley Data 10.17632/9rw3vkcfy4.6, CC BY 4.0',
          'split': 'stratified 80/20 train/test on the leaf, 10% of train as validation, seed 0',
          'features': f'binary bag of words, {vocab} most frequent training-set words, English stop words removed',
          'bit_names': [f'contains "{w}"' for w in cv.get_feature_names_out().tolist()]})


if __name__ == '__main__':
    {'alerce': alerce, 'nslkdd': nslkdd, 'wos': wos}[sys.argv[1]](sys.argv[2])
