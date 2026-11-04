"""Inference latency and energy of the flat TM and of the tree of TM experts,
estimated with the analytical AP(TM) model (Srot et al., ESWA, to appear):

* inference = one compare-tm, one write of o_j, 2K masked compare+popcount
  pairs, and a controller argmax over K votes: 5K+1 cycles, plus a pipelined
  reduction tree of at most ceil(log2 m) cycles;
* energy = m * sum_op b_op E_op with E_cmp = 1 fJ and E_wr = 0.5 fJ per active
  bit and row; compare-tm activates all ceil(M(N+1)/512) 512-cell segments of
  the literal region, compares the M include bits and charges masked cells
  E_cmp/2; masked compares activate the 64-cell control segment and compare
  ceil(log2 K) class-identifier bits plus polarity, o_j and update_enabled.

Assumptions added for the tree (stated in the paper): each expert occupies its
own independently activated array (m = rows of that expert), and routing to a
child adds one controller dispatch cycle.

The model is first checked against the AP(TM) MNIST inference figures
(K=10, m=2000, M=1568, N=8: 51 cycles and 17.3 nJ).
"""
import json
import math

E_CMP, E_WR = 1.0, 0.5  # fJ per active bit and row
N = 8


def infer(K, m, M):
    segs = math.ceil(M * (N + 1) / 512)
    cells = 512 * segs
    e_cmptm = m * (M * E_CMP + (cells - M) * E_CMP / 2)
    e_write = m * 1 * E_WR
    cmp_bits = math.ceil(math.log2(K)) + 3
    e_pc = 2 * K * m * (cmp_bits * E_CMP + (64 - cmp_bits) * E_CMP / 2)
    return {'cycles': 5 * K + 1, 'reduction_max': math.ceil(math.log2(m)),
            'energy_nJ': (e_cmptm + e_write + e_pc) * 1e-6,
            'rows': m, 'bits_per_row': M * (N + 1) + 51 + math.ceil(math.log2(K))}


if __name__ == '__main__':
    ref = infer(10, 2000, 1568)
    print('AP(TM) MNIST check:', ref)  # expected 51 cycles, ~17.3 nJ
    ev = json.load(open('results/main/evaluation.json'))
    M = ev['cost']['flat']['per_node']['literals']
    C = json.load(open('results/main/flat_hist.json'))['cfg']['clauses']
    K = {'root': 3, 'Transient': 4, 'Stochastic': 5, 'Periodic': 6, 'flat': 15}
    res = {n: infer(k, C * k, M) for n, k in K.items()}
    frac = ev['cost']['tree']['routing_fraction_test']
    exp = lambda key: res['root'][key] + sum(frac[p] * res[p][key] for p in frac)
    tree = {'cycles_mean': exp('cycles') + 1, 'energy_nJ_mean': exp('energy_nJ'),
            'cycles_max': res['root']['cycles'] + 1 + max(res[p]['cycles'] for p in frac),
            'reduction_max_total': res['root']['reduction_max'] + max(res[p]['reduction_max'] for p in frac),
            'rows_total': sum(res[n]['rows'] for n in ['root', 'Transient', 'Stochastic', 'Periodic']),
            'bits_per_row_max': max(res[n]['bits_per_row'] for n in ['root', 'Transient', 'Stochastic', 'Periodic'])}
    # Including the pipelined reduction trees (at most ceil(log2 m) cycles each, in series on the path)
    tree['cycles_incl_mean'] = tree['cycles_mean'] + exp('reduction_max')
    tree['cycles_incl_max'] = max(res['root']['cycles'] + 1 + res['root']['reduction_max'] + res[p]['cycles'] + res[p]['reduction_max'] for p in frac)
    res['flat']['cycles_incl'] = res['flat']['cycles'] + res['flat']['reduction_max']
    out = {'reference_mnist': ref, 'per_expert': res, 'routing_fraction': frac, 'tree_hard': tree,
           'flat': res['flat'], 'literals_M': M, 'clauses_per_class': C}
    json.dump(out, open('results/ap_cost.json', 'w'), indent=1)
    print(json.dumps({'tree_hard': tree, 'flat': res['flat']}, indent=1))
    print('energy ratio tree/flat:', tree['energy_nJ_mean'] / res['flat']['energy_nJ'])
    print('cycle ratio tree/flat:', tree['cycles_mean'] / res['flat']['cycles'])
