"""Train the tree of TM experts (root + 3 children) and a flat 15-class TM.

Usage: python run_tms.py --nodes root,Transient,Stochastic,Periodic,flat --epochs 40
Results: results/<run>/<node>_scores.npz and <node>_hist.json (written every epoch).
"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
from data import LEAVES, TOP, TREE, PARENT
from tm_tree import train_node

ap = argparse.ArgumentParser()
ap.add_argument('--prep', default='work/prepared.npz')
ap.add_argument('--run', default='results/main')
ap.add_argument('--nodes', default='root,Transient,Stochastic,Periodic,flat')
ap.add_argument('--epochs', type=int, default=40)
ap.add_argument('--per_class', type=int, default=2500)
ap.add_argument('--cfg', default='{}', help='JSON dict of per-node overrides')
ap.add_argument('--procs', type=int, default=2)
ap.add_argument('--seed', type=int, default=0, help='seeds the per-epoch resampling (pyTsetlinMachine C RNG starts from its fixed internal state)')
a = ap.parse_args()

DEFAULT = {
    'root':       {'clauses': 600, 'T': 600, 's': 10.0},
    'Transient':  {'clauses': 400, 'T': 400, 's': 10.0},
    'Stochastic': {'clauses': 400, 'T': 400, 's': 10.0},
    'Periodic':   {'clauses': 400, 'T': 400, 's': 10.0},
    'flat':       {'clauses': 400, 'T': 400, 's': 10.0},
}
over = json.loads(a.cfg)


def node_data(name, P):
    leaf_tr, leaf_va = P['ytr'], P['yva']
    if name == 'root':
        classes = TOP
        f = lambda y: np.array([TOP.index(PARENT[LEAVES[i]]) for i in y])
        y_tr, y_va, keep = f(leaf_tr), f(leaf_va), np.ones(len(leaf_tr), bool)
    elif name == 'flat':
        classes = LEAVES
        y_tr, y_va, keep = leaf_tr, leaf_va, np.ones(len(leaf_tr), bool)
    else:
        classes = TREE[name]
        loc = {LEAVES.index(c): i for i, c in enumerate(classes)}
        keep = np.array([i in loc for i in leaf_tr])
        y_tr = np.array([loc[i] for i in leaf_tr[keep]])
        y_va = np.array([loc.get(i, -1) for i in leaf_va])
    return classes, {
        'Xb_tr': np.ascontiguousarray(P['Btr'][keep]).astype(np.uint32),
        'y_tr': y_tr.astype(np.uint32),
        'Xb_va': np.ascontiguousarray(P['Bva']).astype(np.uint32),
        'y_va': y_va.astype(np.int64),
        'Xb_te': np.ascontiguousarray(P['Bte']).astype(np.uint32),
    }


def job(name):
    P = np.load(a.prep)
    classes, d = node_data(name, P)
    cfg = dict(DEFAULT[name], **over.get(name, {}))
    cfg.update({'epochs': a.epochs, 'per_class': cfg.get('per_class', a.per_class), 'seed': a.seed})
    logf = open(os.path.join(a.run, f'{name}.log'), 'a')

    def log(m):
        logf.write(m + '\n'); logf.flush()
    train_node(name, classes, d, cfg, a.run, log=log)
    return name


if __name__ == '__main__':
    os.makedirs(a.run, exist_ok=True)
    nodes = a.nodes.split(',')
    # Longest jobs first for better packing on few cores.
    order = sorted(nodes, key=lambda n: {'flat': 0, 'Periodic': 1, 'Stochastic': 2,
                                         'Transient': 3, 'root': 4}[n])
    with Pool(a.procs) as p:
        for n in p.imap_unordered(job, order):
            print('done', n, flush=True)
