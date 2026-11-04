"""Tree of Tsetlin Machine experts mirroring the ALeRCE label taxonomy.

Each node is a multiclass Tsetlin Machine (pyTsetlinMachine) that separates
the children of one taxonomy node. Training of each node uses only the
samples of its own subtree, with class-balanced resampling every epoch
(the TM analogue of the balanced bootstrap of the ALeRCE balanced random
forest). Per-epoch class sums on the validation and test sets are logged,
so end-to-end routing can be evaluated for every epoch offline.
"""
import json
import os
import time
import numpy as np
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


class NodeTM:
    def __init__(self, classes, clauses, T, s, state_bits=8, weighted=True, seed=0):
        self.classes = list(classes)
        self.k = len(classes)
        self.clauses = clauses
        self.T = T
        self.s = s
        self.tm = MultiClassTsetlinMachine(clauses, T, s, number_of_state_bits=state_bits,
                                           weighted_clauses=weighted)
        self.rng = np.random.default_rng(seed)
        # pyTsetlinMachine draws the negative class with libc rand(); seed it so
        # every node is reproducible regardless of process scheduling. The clause
        # feedback RNG (pcg32 in fast_rand.h) starts from a fixed internal state
        # that the library does not expose.
        import ctypes
        ctypes.CDLL(None).srand(ctypes.c_uint(1000 + seed))
        self.initialized = False

    def balanced_indices(self, y, per_class, groups=None):
        """P draws per class. With groups (the leaf class of each sample), the
        draws of a class are spread evenly over the leaves below it
        (leaf-balanced resampling); otherwise they are uniform over the
        samples of the class (child-balanced resampling)."""
        idx = []
        for c in range(self.k):
            members = np.flatnonzero(y == c)
            if groups is None:
                idx.append(self.rng.choice(members, per_class, replace=members.size < per_class))
                continue
            leaves = np.unique(groups[members])
            quota = np.full(leaves.size, per_class // leaves.size)
            quota[self.rng.permutation(leaves.size)[:per_class % leaves.size]] += 1
            for g, q in zip(leaves, quota):
                mg = members[groups[members] == g]
                idx.append(self.rng.choice(mg, q, replace=mg.size < q))
        idx = np.concatenate(idx)
        self.rng.shuffle(idx)
        return idx

    def train_epoch(self, Xb, y, per_class, groups=None):
        idx = self.balanced_indices(y, per_class, groups)
        if not self.initialized:
            # pyTsetlinMachine infers the class count from max(Y) on first fit.
            # Guarantee all classes appear in the first call.
            self.tm.fit(Xb[idx], y[idx], epochs=1, incremental=False)
            self.initialized = True
        else:
            self.tm.fit(Xb[idx], y[idx], epochs=1, incremental=True)

    def include_masks(self):
        """Boolean include matrix, shape (k, clauses, n_literals), and clause weights."""
        tm = self.tm
        chunks, bits = tm.number_of_ta_chunks, tm.number_of_state_bits
        n_lit = tm.number_of_features
        masks, weights = [], []
        for w, ta in tm.get_state():
            ta = ta.reshape(tm.number_of_clauses, chunks, bits)
            act = ta[:, :, bits - 1]  # MSB word holds the include action
            m = ((act[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool)
            masks.append(m.reshape(tm.number_of_clauses, chunks * 32)[:, :n_lit])
            weights.append(w.astype(np.int64))
        return np.stack(masks), np.stack(weights)

    def scores(self, Xb, masks=None, weights=None):
        """Clipped class sums, shape (n, k). Empty clauses output 0 at inference,
        matching pyTsetlinMachine predict."""
        if masks is None:
            masks, weights = self.include_masks()
        nonempty = masks.any(axis=2)  # (k, clauses)
        sign = np.where(np.arange(self.clauses) % 2 == 0, 1, -1)
        eff = (weights * sign[None, :] * nonempty).astype(np.float64)  # (k, clauses)
        s = np.empty((Xb.shape[0], self.k), dtype=np.int64)
        chunk = 1024  # bounded memory: chunk x k x clauses clause outputs at a time
        for a in range(0, Xb.shape[0], chunk):
            out = self.tm.transform(Xb[a:a + chunk], inverted=False)
            out = out.reshape(-1, self.k, self.clauses).astype(np.float32)
            s[a:a + chunk] = np.rint(np.einsum('nkc,kc->nk', out, eff)).astype(np.int64)
        return np.clip(s, -self.T, self.T)

    def stats(self, masks=None, weights=None):
        if masks is None:
            masks, weights = self.include_masks()
        inc = masks.sum(axis=2)
        return {
            'classes': self.k,
            'clauses_per_class': self.clauses,
            'total_clauses': int(self.k * self.clauses),
            'nonempty_clauses': int((inc > 0).sum()),
            'n_literals': int(masks.shape[2]),
            'included_literals_total': int(inc.sum()),
            'max_clause_weight': int(weights.max()),
        }


def train_node(name, classes, data, cfg, out_dir, log=print):
    """data: dict with Xb_tr, y_tr, Xb_va, y_va, Xb_te (labels as ints 0..k-1,
    val/test labels may be -1 for samples outside this subtree)."""
    os.makedirs(out_dir, exist_ok=True)
    node = NodeTM(classes, cfg['clauses'], cfg['T'], cfg['s'], seed=cfg.get('seed', 0),
                  weighted=cfg.get('weighted', True))
    hist = {'name': name, 'classes': classes, 'cfg': cfg, 'epochs': []}
    va_scores, te_scores, scored = [], [], []
    start = 1
    ck_path = os.path.join(out_dir, f'{name}_ckpt.pkl')
    if os.path.exists(os.path.join(out_dir, f'{name}_final_masks.npz')):
        log(f'[{name}] already complete, skipped')
        return None, None
    if os.path.exists(ck_path):
        # Resume after an interruption: TM state, resampling RNG and logs are
        # restored; libc rand() (negative-class draw) is reseeded deterministically.
        import pickle, ctypes
        ck = pickle.load(open(ck_path, 'rb'))
        node.tm, node.initialized = ck['tm'], True
        node.rng.bit_generator.state = ck['rng']
        hist, va_scores, te_scores, scored = ck['hist'], ck['va'], ck['te'], ck['scored']
        start = ck['epoch'] + 1
        ctypes.CDLL(None).srand(ctypes.c_uint(1000 + cfg.get('seed', 0) + 7919 * ck['epoch']))
        hist.setdefault('resumed_at_epoch', []).append(ck['epoch'])
        log(f'[{name}] resumed after epoch {ck["epoch"]}')
    for ep in range(start, cfg['epochs'] + 1):
        t0 = time.time()
        node.train_epoch(data['Xb_tr'], data['y_tr'], cfg['per_class'], data.get('g_tr'))
        t_train = time.time() - t0
        every = cfg.get('score_every', 1)
        if ep % every and ep != cfg['epochs']:
            hist['epochs'].append({'epoch': ep, 'train_s': round(t_train, 2)})
            log(f'[{name}] epoch {ep} train={t_train:.1f}s')
            _checkpoint(ck_path, node, hist, va_scores, te_scores, scored, ep)
            continue
        masks, weights = node.include_masks()
        sv = node.scores(data['Xb_va'], masks, weights)
        st = node.scores(data['Xb_te'], masks, weights)
        va_scores.append(sv.astype(np.int16))
        te_scores.append(st.astype(np.int16))
        scored.append(ep)
        in_tree = data['y_va'] >= 0
        acc_va = float((sv[in_tree].argmax(1) == data['y_va'][in_tree]).mean())
        rec = {'epoch': ep, 'train_s': round(t_train, 2), 'node_val_acc': acc_va,
               'included_literals': int(masks.sum())}
        hist['epochs'].append(rec)
        log(f'[{name}] epoch {ep} val_acc={acc_va:.4f} train={t_train:.1f}s')
        # Incremental persistence: partial results survive interruption.
        np.savez_compressed(os.path.join(out_dir, f'{name}_scores.npz'),
                            val=np.stack(va_scores), test=np.stack(te_scores), epochs=np.array(scored))
        with open(os.path.join(out_dir, f'{name}_hist.json'), 'w') as f:
            json.dump(hist, f, indent=1)
        _checkpoint(ck_path, node, hist, va_scores, te_scores, scored, ep)
    masks, weights = node.include_masks()
    if data.get('Xb_nov') is not None:   # e.g. NSL-KDD attack types unseen in training
        np.save(os.path.join(out_dir, f'{name}_novel_scores.npy'),
                node.scores(data['Xb_nov'], masks, weights).astype(np.int32))
    hist['final_stats'] = node.stats(masks, weights)
    with open(os.path.join(out_dir, f'{name}_hist.json'), 'w') as f:
        json.dump(hist, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, f'{name}_final_masks.npz'),
                        masks=np.packbits(masks, axis=2), weights=weights,
                        n_literals=masks.shape[2])
    if os.path.exists(ck_path):
        os.remove(ck_path)
    return node, hist


def _checkpoint(path, node, hist, va, te, scored, ep):
    import pickle
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'tm': node.tm, 'rng': node.rng.bit_generator.state, 'hist': hist,
                     'va': va, 'te': te, 'scored': scored, 'epoch': ep}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
