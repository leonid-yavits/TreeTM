"""Export the artifacts needed to reuse the trained models without the dataset:
- the booleanizer (thermometer thresholds per feature, missing-bit flags)
  and a literal map (literal index -> feature, comparison, threshold),
- the split (object ids of the training, validation and test sources).
The thresholds are refit on the training split exactly as in prepare.py and
checked bit for bit against work/prepared.npz.
Usage: python export_artifacts.py <data_path> <out_dir>"""
import json
import os
import sys
import numpy as np
from data import load_raw, preprocess, split, Thermometer, LEAVES

data_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
meta = json.load(open('work/meta.json'))
P = np.load('work/prepared.npz')
X, y = preprocess(load_raw(data_path))
(Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, seed=meta['seed'])
assert list(X.columns) == meta['feature_names']
enc = Thermometer(meta['bins']).fit(Xtr.values)
for Z, key in ((Xtr, 'Btr'), (Xva, 'Bva'), (Xte, 'Bte')):
    assert np.array_equal(enc.transform(Z.values).astype(np.uint8), P[key]), key
lab = {c: i for i, c in enumerate(LEAVES)}
for yy, key in ((ytr, 'ytr'), (yva, 'yva'), (yte, 'yte')):
    assert np.array_equal(np.array([lab[c] for c in yy]), P[key]), key

bits = []
for j, name in enumerate(meta['feature_names']):
    for t in enc.thresholds[j]:
        bits.append({'feature': name, 'test': '>=', 'threshold': float(t)})
    if enc.has_missing[j]:
        bits.append({'feature': name, 'test': 'missing', 'threshold': None})
assert len(bits) == meta['n_bits']
literals = [dict(b, negated=False) for b in bits] + [dict(b, negated=True) for b in bits]
json.dump({'missing_value': -999, 'quantile_levels': [k / meta['bins'] for k in range(1, meta['bins'])],
           'feature_names': meta['feature_names'],
           'thresholds': {n: [float(t) for t in enc.thresholds[j]] for j, n in enumerate(meta['feature_names'])},
           'has_missing_bit': {n: bool(enc.has_missing[j]) for j, n in enumerate(meta['feature_names'])},
           'n_bits': len(bits), 'n_literals': len(literals),
           'literal_order': 'literals 0..n_bits-1 are the bits, n_bits..2*n_bits-1 their negations',
           'literals': literals}, open(os.path.join(out, 'booleanizer.json'), 'w'), indent=1)
json.dump({'seed': meta['seed'], 'class_order': LEAVES,
           'train_oids': list(map(str, Xtr.index)), 'val_oids': list(map(str, Xva.index)),
           'test_oids': list(map(str, Xte.index))},
          open(os.path.join(out, 'split_oids.json'), 'w'))
print('exported; bits', len(bits), 'train/val/test', len(Xtr), len(Xva), len(Xte))
