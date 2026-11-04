"""Short validation probes of (T, s) on one expert; selection metric:
validation balanced accuracy of that expert at the last probe epoch.
Usage: python probe.py <dataset> <node> <clauses> <epochs> <per_class> "T,s;T,s;..."
Writes results/<dataset>/probe/<node>_probe.json"""
import json, os, subprocess, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from taxonomy import Taxonomy
ds, node, C, E, P, grid = sys.argv[1:7]
meta = json.load(open(f'work/{ds}/meta.json')); tax = Taxonomy(meta['taxonomy'])
yva = np.load(f'work/{ds}/prepared.npz')['yva']
lab = yva if node == 'flat' else tax.node_labels(node, yva)[1]
path = f'results/{ds}/probe/{node}_probe.json'
out = json.load(open(path)) if os.path.exists(path) else []
for g in grid.split(';'):
    T, s = g.split(',')
    run = f'results/{ds}/probe/{node}_C{C}_T{T}_s{s}'
    subprocess.run([sys.executable, 'train.py', ds, '--run', run, '--clauses', C, '--T', T, '--s', s,
                    '--epochs', E, '--per_class', P, '--nodes', node, '--procs', '1'], check=True)
    sc = np.load(f'{run}/{node}_scores.npz')['val']
    m = lab >= 0
    bal = [float(balanced_accuracy_score(lab[m], sc[e][m].argmax(1))) for e in range(len(sc))]
    out.append({'T': int(T), 's': float(s), 'val_bal_per_epoch': bal})
    print(node, T, s, [round(b, 4) for b in bal], flush=True)
    json.dump(out, open(path, 'w'), indent=1)
