"""Inference cost of TreeTM (hard routing) versus the flat TM for one trained
run: counted work, AP(TM) analytical estimates, and measured CPU time of a
bit-packed numpy implementation (single process).

Usage: python cost.py <dataset> <run_dir>
Writes <run_dir>/treetm_cost.json.
"""
import json
import math
import sys
import time
import numpy as np
from taxonomy import Taxonomy
from evaluate import visit_fractions

E_CMP, E_WR, N_STATE = 1.0, 0.5, 8   # AP(TM) model constants (fJ per active bit and row; TA bits)


def ap_infer(K, m, M):
    """AP(TM) analytical inference model (Srot et al., ESWA): 5K+1 cycles plus a
    reduction tree of at most ceil(log2 m) cycles; energy m * sum b_op E_op."""
    segs = math.ceil(M * (N_STATE + 1) / 512)
    e_cmptm = m * (M * E_CMP + (512 * segs - M) * E_CMP / 2)
    e_write = m * E_WR
    cmp_bits = math.ceil(math.log2(K)) + 3
    e_pc = 2 * K * m * (cmp_bits * E_CMP + (64 - cmp_bits) * E_CMP / 2)
    return {'cycles': 5 * K + 1, 'reduction': math.ceil(math.log2(m)),
            'energy_nJ': (e_cmptm + e_write + e_pc) * 1e-6}


class PackedTM:
    """Bit-packed clause evaluation from saved include masks: a clause fires
    when none of its included literals is false; empty clauses output 0."""

    def __init__(self, path):
        z = np.load(path)
        n_lit = int(z['n_literals'])
        m = np.unpackbits(z['masks'], axis=2)[:, :, :n_lit].astype(bool)
        self.k, self.C, _ = m.shape
        self.n_lit = n_lit
        self.W = (n_lit + 63) // 64
        flat = m.reshape(self.k * self.C, n_lit)
        pad = np.zeros((flat.shape[0], self.W * 64), bool); pad[:, :n_lit] = flat
        self.mask = np.packbits(pad, axis=1, bitorder='little').view(np.uint64)   # (k*C, W)
        sign = np.where(np.arange(self.C) % 2 == 0, 1, -1)
        self.coef = (z['weights'] * sign[None, :] * m.any(2)).reshape(-1).astype(np.int64)
        self.included = int(m.sum())

    def pack(self, Xbits):
        L = np.hstack([Xbits, 1 - Xbits]).astype(bool)
        pad = np.zeros((L.shape[0], self.W * 64), bool); pad[:, :self.n_lit] = L
        return np.packbits(pad, axis=1, bitorder='little').view(np.uint64)

    def votes(self, lit):   # lit: (n, W) packed literals
        viol = np.zeros((lit.shape[0], self.mask.shape[0]), bool)
        for w in range(self.W):
            viol |= (self.mask[None, :, w] & ~lit[:, None, w]) != 0
        fire = ~viol
        return (fire.astype(np.int64) * self.coef[None, :]).reshape(-1, self.k, self.C).sum(2)


def main(ds, run):
    meta = json.load(open(f'work/{ds}/meta.json'))
    tax = Taxonomy(meta['taxonomy'])
    P = np.load(f'work/{ds}/prepared.npz')
    Xte = P['Bte']
    nodes = tax.internal + ['flat']
    tm = {u: PackedTM(f'{run}/{u}_final_masks.npz') for u in nodes}
    T = {u: json.load(open(f'{run}/{u}_hist.json'))['cfg']['T'] for u in nodes}
    M = tm['flat'].n_lit
    # 1) the packed implementation reproduces the stored (clipped) class sums
    sub = Xte[:2000]
    lit = tm['flat'].pack(sub)
    for u in nodes:
        stored = np.load(f'{run}/{u}_scores.npz')['test'][-1][:2000]
        assert np.array_equal(np.clip(tm[u].votes(lit), -T[u], T[u]), stored), u
    # 2) counted work under hard routing (visit fractions on the test set)
    S = {u: np.load(f'{run}/{u}_scores.npz')['test'][-1].astype(float) for u in tax.internal}
    frac = visit_fractions(tax, S)
    work = {'dense_literal_tests_tree': sum(frac[u] * tm[u].k * tm[u].C * M for u in tax.internal),
            'dense_literal_tests_flat': tm['flat'].k * tm['flat'].C * M,
            'included_literals_tree': sum(frac[u] * tm[u].included for u in tax.internal),
            'included_literals_flat': tm['flat'].included,
            'clauses_stored_tree': sum(tm[u].k * tm[u].C for u in tax.internal),
            'clauses_stored_flat': tm['flat'].k * tm['flat'].C,
            'expected_experts_per_input': float(sum(frac.values())), 'visit_fraction': frac}
    # 3) AP(TM) analytical estimates (one array per expert, +1 dispatch cycle per routing step)
    ap = {u: ap_infer(tm[u].k, tm[u].k * tm[u].C, M) for u in nodes}
    exp_steps = sum(frac.values()) - 1
    ap_tree = {'cycles': sum(frac[u] * ap[u]['cycles'] for u in tax.internal) + exp_steps,
               'cycles_incl_reduction': sum(frac[u] * (ap[u]['cycles'] + ap[u]['reduction']) for u in tax.internal) + exp_steps,
               'energy_nJ': sum(frac[u] * ap[u]['energy_nJ'] for u in tax.internal)}
    worst = max(len(p) for p in tax.paths)
    ap_tree['cycles_max'] = max(sum(ap[u]['cycles'] for u, _ in p) + len(p) - 1 for p in tax.paths)
    ap_out = {'per_node': ap, 'tree_hard_expected': ap_tree,
              'flat': {'cycles': ap['flat']['cycles'], 'cycles_incl_reduction': ap['flat']['cycles'] + ap['flat']['reduction'],
                       'energy_nJ': ap['flat']['energy_nJ']}, 'max_depth': worst}
    # 4) measured CPU time per input (packed numpy, batches of 256, best of 3)
    def timed(fn, reps=3):
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
        return best
    Xm = Xte[:4096]
    litm = tm['flat'].pack(Xm)

    def run_flat():
        for a in range(0, len(litm), 256):
            tm['flat'].votes(litm[a:a + 256]).argmax(1)

    def run_tree():
        for a in range(0, len(litm), 256):
            L = litm[a:a + 256]
            node = np.array(['root'] * len(L), dtype=object)
            active = np.ones(len(L), bool)
            while active.any():
                for u in set(node[active]):
                    mm = active & (node == u)
                    node[mm] = np.array(tax.children[u], dtype=object)[tm[u].votes(L[mm]).argmax(1)]
                active = np.array([x in tax.children for x in node])
    t_flat, t_tree = timed(run_flat) / len(Xm), timed(run_tree) / len(Xm)
    out = {'work': work, 'ap_tm': ap_out,
           'cpu_seconds_per_input': {'flat': t_flat, 'tree_hard': t_tree, 'n_inputs': len(Xm),
                                     'implementation': 'bit-packed numpy, batch 256, best of 3, single process'},
           'packed_implementation_matches_stored_scores': True}
    json.dump(out, open(f'{run}/treetm_cost.json', 'w'), indent=1, default=float)
    print(ds, 'dense', work['dense_literal_tests_tree'] / work['dense_literal_tests_flat'],
          'incl', work['included_literals_tree'] / work['included_literals_flat'],
          'AP cycles', ap_tree['cycles'], ap_out['flat']['cycles'], 'energy', ap_tree['energy_nJ'], ap_out['flat']['energy_nJ'],
          'cpu us', t_tree * 1e6, t_flat * 1e6)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
