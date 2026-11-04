"""Short hyperparameter probe on one node using the validation split only."""
import sys
import time
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tm_tree import NodeTM
from data import LEAVES, TOP, TREE, PARENT

node_name = sys.argv[1]
grid = json.loads(sys.argv[2])
epochs = int(sys.argv[3])
P = np.load('work/prepared.npz')
ytr, yva = P['ytr'], P['yva']
if node_name == 'root':
    classes = TOP
    m = lambda y: np.array([TOP.index(PARENT[LEAVES[i]]) for i in y])
    ktr, kva = np.ones(len(ytr), bool), np.ones(len(yva), bool)
    Ytr, Yva = m(ytr), m(yva)
else:
    classes = TREE[node_name]
    loc = {LEAVES.index(c): i for i, c in enumerate(classes)}
    ktr = np.array([i in loc for i in ytr]); kva = np.array([i in loc for i in yva])
    Ytr = np.array([loc[i] for i in ytr[ktr]]); Yva = np.array([loc[i] for i in yva[kva]])
Xtr = np.ascontiguousarray(P['Btr'][ktr]).astype(np.uint32)
Xva = np.ascontiguousarray(P['Bva'][kva]).astype(np.uint32)
for g in grid:
    node = NodeTM(classes, g['clauses'], g['T'], g['s'], weighted=g.get('weighted', True))
    t0 = time.time()
    res = []
    for ep in range(epochs):
        node.train_epoch(Xtr, Ytr.astype(np.uint32), g.get('per_class', 2000))
        p = node.scores(Xva).argmax(1)
        res.append(round(balanced_accuracy_score(Yva, p), 4))
    print(node_name, g, 'bal_acc per epoch', res, f'{(time.time()-t0)/epochs:.1f}s/epoch', flush=True)
