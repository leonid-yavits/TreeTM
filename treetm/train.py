"""Train the TreeTM experts (one multiclass TM per internal taxonomy node) and
a flat TM on any prepared dataset.

Usage: python train.py <dataset> --run <dir> --clauses C --T T --s s --epochs E
                       --per_class P --seed S [--nodes root,...,flat] [--procs 2]
Every epoch writes <dir>/<node>_scores.npz (val/test class sums) and
<dir>/<node>_hist.json; the final include masks go to <dir>/<node>_final_masks.npz.
"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
from taxonomy import Taxonomy
from tm_node import train_node

ap = argparse.ArgumentParser()
ap.add_argument('dataset')
ap.add_argument('--run', required=True)
ap.add_argument('--clauses', type=int, default=500)
ap.add_argument('--T', type=int, default=50)
ap.add_argument('--s', type=float, default=4.0)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--per_class', type=int, default=2000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--nodes', default='')
ap.add_argument('--procs', type=int, default=2)
ap.add_argument('--weighted', action='store_true')
ap.add_argument('--score_every', type=int, default=5)
ap.add_argument('--resample', default='child', choices=['child', 'leaf'],
                help='child: P draws per child, uniform over its samples; leaf: P draws per child, spread evenly over the leaves below it')
a = ap.parse_args()

PREP = os.path.join('work', a.dataset, 'prepared.npz')
META = json.load(open(os.path.join('work', a.dataset, 'meta.json')))
TAX = Taxonomy(META['taxonomy'])


def job(name):
    P = np.load(PREP)
    ytr, yva = P['ytr'], P['yva']
    if name == 'flat':
        classes, keep, ltr, lva = TAX.leaves, np.ones(len(ytr), bool), ytr, yva
    else:
        classes = TAX.children[name]
        keep, ltr = TAX.node_labels(name, ytr)
        ltr = ltr[keep]
        lva = TAX.node_labels(name, yva)[1]
    d = {'Xb_tr': np.ascontiguousarray(P['Btr'][keep]).astype(np.uint32), 'y_tr': ltr.astype(np.uint32),
         'Xb_va': np.ascontiguousarray(P['Bva']).astype(np.uint32), 'y_va': lva.astype(np.int64),
         'Xb_te': np.ascontiguousarray(P['Bte']).astype(np.uint32),
         'Xb_nov': np.ascontiguousarray(P['Bnov']).astype(np.uint32) if 'Bnov' in P.files else None,
         'g_tr': ytr[keep] if (a.resample == 'leaf' and name != 'flat') else None}
    cfg = {'clauses': a.clauses, 'T': a.T, 's': a.s, 'weighted': a.weighted, 'epochs': a.epochs,
           'per_class': a.per_class, 'seed': a.seed, 'score_every': a.score_every, 'resample': a.resample}
    logf = open(os.path.join(a.run, f'{name}.log'), 'a')
    train_node(name, classes, d, cfg, a.run, log=lambda m: (logf.write(m + '\n'), logf.flush()))
    return name


if __name__ == '__main__':
    os.makedirs(a.run, exist_ok=True)
    nodes = a.nodes.split(',') if a.nodes else TAX.internal + ['flat']
    # longest jobs first (flat, then nodes with most training samples x classes)
    P = np.load(PREP)
    size = {n: (len(P['ytr']) * len(TAX.leaves) if n == 'flat'
                else len(TAX.subtree_leaves[n]) * len(TAX.children[n])) for n in nodes}
    order = sorted(nodes, key=lambda n: -size[n])
    json.dump(vars(a), open(os.path.join(a.run, 'args.json'), 'w'), indent=1)
    with Pool(a.procs) as p:
        for n in p.imap_unordered(job, order):
            print('done', n, flush=True)
