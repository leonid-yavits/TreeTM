"""Generic class taxonomy for TreeTM.

A taxonomy is a nested dict: {child_name: subtree_or_None}. None marks a leaf.
The implicit root node is called 'root'. Internal node names and leaf names
must be unique across the tree. Leaves are numbered in depth-first order.
"""
import json


class Taxonomy:
    def __init__(self, tree):
        self.tree = tree
        self.children = {}      # internal node -> list of child names
        self.parent = {}        # node -> parent node
        self.leaves = []
        self._walk('root', tree)
        self.internal = list(self.children)
        self.leaf_index = {l: i for i, l in enumerate(self.leaves)}
        self.is_leaf = {n: n in self.leaf_index for n in list(self.parent) + ['root']}
        # path of each leaf: list of (internal node, child position)
        self.paths = []
        for l in self.leaves:
            p, n = [], l
            while n != 'root':
                u = self.parent[n]
                p.append((u, self.children[u].index(n)))
                n = u
            self.paths.append(p[::-1])
        self.depth = max(len(p) for p in self.paths)
        # leaves below each node
        self.subtree_leaves = {u: [i for i, p in enumerate(self.paths) if any(v == u for v, _ in p)]
                               for u in self.internal}

    def _walk(self, name, sub):
        self.children[name] = list(sub)
        for c, s in sub.items():
            if c in self.parent or c == 'root':
                raise ValueError(f'duplicate node name {c}')
            self.parent[c] = name
            if s is None:
                self.leaves.append(c)
            else:
                self._walk(c, s)

    def node_labels(self, u, y_leaf):
        """For internal node u: mask of samples in its subtree and the child
        position on their path (-1 outside the subtree)."""
        lab = -1 * __import__('numpy').ones(len(y_leaf), dtype=int)
        for li in self.subtree_leaves[u]:
            pos = dict(self.paths[li])[u]
            lab[y_leaf == li] = pos
        return lab >= 0, lab

    def ancestors(self, leaf_id):
        """Nodes on the path from the root (excluded) to the leaf (included)."""
        return [self.children[u][k] for u, k in self.paths[leaf_id]]

    def level_label(self, leaf_ids, d):
        """Name of the depth-d ancestor (d = 1 is a child of the root); a leaf
        shallower than d keeps its own name."""
        out = []
        for i in leaf_ids:
            a = self.ancestors(int(i))
            out.append(a[min(d, len(a)) - 1])
        return out

    def to_json(self):
        return json.dumps(self.tree)

    @staticmethod
    def from_json(s):
        return Taxonomy(json.loads(s))
