"""Load, preprocess, split and booleanize the ALeRCE labeled set.

Usage: python prepare.py <data_path> [--bins 8] [--seed 0]
Writes work/prepared.npz (float features for the BRF, bits for the TMs, labels).
"""
import argparse
import json
import os
import numpy as np
from data import load_raw, preprocess, split, Thermometer, LEAVES

ap = argparse.ArgumentParser()
ap.add_argument('data_path')
ap.add_argument('--bins', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='work')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = load_raw(a.data_path)
X, y = preprocess(df)
print('samples', X.shape[0], 'features', X.shape[1])
(Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y, seed=a.seed)
enc = Thermometer(a.bins).fit(Xtr.values)
Btr, Bva, Bte = (enc.transform(Z.values) for Z in (Xtr, Xva, Xte))
print('bits per sample', Btr.shape[1])
lab = {c: i for i, c in enumerate(LEAVES)}
np.savez_compressed(os.path.join(a.out, 'prepared.npz'),
                    Ftr=Xtr.values.astype(np.float32), Fva=Xva.values.astype(np.float32),
                    Fte=Xte.values.astype(np.float32),
                    Btr=Btr.astype(np.uint8), Bva=Bva.astype(np.uint8), Bte=Bte.astype(np.uint8),
                    ytr=np.array([lab[c] for c in ytr]), yva=np.array([lab[c] for c in yva]),
                    yte=np.array([lab[c] for c in yte]))
meta = {'n_samples': int(X.shape[0]), 'n_features': int(X.shape[1]),
        'feature_names': list(X.columns), 'n_bits': int(Btr.shape[1]),
        'bins': a.bins, 'seed': a.seed,
        'n_train': int(len(ytr)), 'n_val': int(len(yva)), 'n_test': int(len(yte)),
        'class_counts': {c: int((y == c).sum()) for c in LEAVES}}
with open(os.path.join(a.out, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != 'feature_names'}, indent=1))
