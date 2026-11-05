"""Unclipped class sums of every trained TM in a run.

pyTsetlinMachine 0.6.6 clips the class sums to [-T, T] at inference, which
creates ties between saturated classes. This script recomputes the class sums
from the saved include masks without clipping, checks that clipping them
reproduces the stored scores exactly, and writes <run>/<node>_uscores.npz
(keys 'val', 'test') and <node>_novel_uscores.npy where novel inputs exist.
Nodes whose masks are symlinks to another run reuse that run's files.

Usage: python unclip.py <dataset> <run_dir> [<run_dir> ...]
"""
import json
import os
import sys
import numpy as np
from taxonomy import Taxonomy


class DenseTM:
    def __init__(self, path):
        z = np.load(path)
        n_lit = int(z['n_literals'])
        m = np.unpackbits(z['masks'], axis=2)[:, :, :n_lit].astype(bool)
        self.k, self.C, _ = m.shape
        self.inc = m.reshape(self.k * self.C, n_lit).T.astype(np.float32)        # (2F, kC)
        sign = np.where(np.arange(self.C) % 2 == 0, 1, -1)
        self.coef = (z['weights'] * sign[None, :] * m.any(2)).reshape(-1).astype(np.int32)
        self.n_lit = n_lit

    def votes(self, X, batch=2048):
        out = np.empty((len(X), self.k), np.int32)
        for a in range(0, len(X), batch):
            x = X[a:a + batch].astype(np.float32)
            false_lit = np.hstack([1 - x, x])                                    # literal is false
            viol = false_lit @ self.inc                                          # (b, kC)
            fire = (viol < 0.5).astype(np.int32) * self.coef[None, :]
            out[a:a + batch] = fire.reshape(len(x), self.k, self.C).sum(2)
        return out


def main(ds, runs):
    tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
    P = np.load(f'work/{ds}/prepared.npz')
    X = {'val': P['Bva'], 'test': P['Bte']}
    nov = P['Bnov'] if 'Bnov' in P else None
    for run in runs:
        for u in tax.internal + ['flat']:
            mpath = f'{run}/{u}_final_masks.npz'
            if not os.path.exists(mpath):
                continue
            if os.path.islink(mpath):
                src = os.path.dirname(os.readlink(mpath))
                for f in (f'{u}_uscores.npz', f'{u}_novel_uscores.npy'):
                    if os.path.exists(os.path.join(run, src, f)) and not os.path.lexists(f'{run}/{f}'):
                        os.symlink(os.path.join(src, f), f'{run}/{f}')
                if os.path.exists(f'{run}/{u}_uscores.npz'):
                    continue
            if os.path.exists(f'{run}/{u}_uscores.npz') and not os.path.islink(f'{run}/{u}_uscores.npz'):
                continue
            tm = DenseTM(mpath)
            T = json.load(open(f'{run}/{u}_hist.json'))['cfg']['T']
            st = np.load(f'{run}/{u}_scores.npz')
            out = {}
            for sp in ('val', 'test'):
                v = tm.votes(X[sp])
                assert np.array_equal(np.clip(v, -T, T), st[sp][-1]), (run, u, sp)
                out[sp] = v
            np.savez_compressed(f'{run}/{u}_uscores.npz', **out)
            if nov is not None and os.path.exists(f'{run}/{u}_novel_scores.npy'):
                v = tm.votes(nov)
                assert np.array_equal(np.clip(v, -T, T), np.load(f'{run}/{u}_novel_scores.npy')), (run, u, 'novel')
                np.save(f'{run}/{u}_novel_uscores.npy', v)
        print('done', run, flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
