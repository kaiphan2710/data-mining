"""CART (Breiman et al., 1984) binary classification tree implemented from scratch with NumPy.

Numeric features use binary threshold splits; nominal features use binary subset
splits found with Breiman's ordering theorem (sort categories by their positive
rate, then only K-1 prefix splits need to be scanned). Impurity is Gini with
optional class weights.
"""
from dataclasses import dataclass

import numpy as np

# Two numeric values closer than this are treated as equal (same constant as sklearn).
FEATURE_THRESHOLD = 1e-7


def gini_binary(pos_weight, total_weight):
    p = pos_weight / total_weight
    return 2.0 * p * (1.0 - p)


@dataclass
class Node:
    n_samples: int
    weight: float
    value: float          # weighted P(y = 1) in the node
    impurity: float       # Gini impurity of the node
    depth: int
    feature: int = -1
    threshold: float = np.nan
    is_categorical: bool = False
    left_mask: np.ndarray | None = None   # per category code: True -> left child
    seen_mask: np.ndarray | None = None   # categories present in the node during fit
    default_left: bool = True             # route for unseen categories (larger child)
    left: int = -1
    right: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left == -1


def _best_prefix_split(tot_w, pos_w, counts, min_leaf, allowed=None):
    """Best boundary for units already in split order.

    Units [:i] go left. Returns (i, W_L*G_L + W_R*G_R), or (-1, inf) if no valid split.
    """
    if len(tot_w) < 2:
        return -1, np.inf
    W, P, N = tot_w.sum(), pos_w.sum(), counts.sum()
    lw = np.cumsum(tot_w)[:-1]
    lp = np.cumsum(pos_w)[:-1]
    ln = np.cumsum(counts)[:-1]
    rw, rp, rn = W - lw, P - lp, N - ln
    valid = (ln >= min_leaf) & (rn >= min_leaf) & (lw > 0) & (rw > 0)
    if allowed is not None:
        valid &= allowed
    if not valid.any():
        return -1, np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = lw * gini_binary(lp, lw) + rw * gini_binary(rp, rw)
    cost = np.where(valid, cost, np.inf)
    i = int(np.argmin(cost))
    return i + 1, float(cost[i])


class CARTClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 class_weight=None, categorical_features=(), ccp_alpha=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.categorical_features = tuple(categorical_features)
        self.ccp_alpha = ccp_alpha

    def get_params(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "categorical_features": self.categorical_features,
            "ccp_alpha": self.ccp_alpha,
        }

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per label")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0/1 labels")

        n, d = X.shape
        self.n_features_in_ = d
        self.categorical_ = np.zeros(d, dtype=bool)
        self.categorical_[list(self.categorical_features)] = True
        self.n_categories_ = np.zeros(d, dtype=np.int64)
        for j in np.flatnonzero(self.categorical_):
            codes = X[:, j]
            if (codes < 0).any() or (codes != np.floor(codes)).any():
                raise ValueError(f"categorical feature {j} must hold non-negative integer codes")
            self.n_categories_[j] = int(codes.max()) + 1

        w = self._sample_weight(y)
        self.nodes_ = []
        self._grow(X, y, w)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
        self.feature_importances_ = self._compute_importances()
        return self

    def _sample_weight(self, y):
        if self.class_weight is None:
            return np.ones(len(y))
        if self.class_weight == "balanced":
            counts = np.bincount(y, minlength=2)
            per_class = len(y) / (2.0 * np.maximum(counts, 1))
        elif isinstance(self.class_weight, dict):
            per_class = np.array([self.class_weight.get(0, 1.0), self.class_weight.get(1, 1.0)])
        else:
            raise ValueError("class_weight must be None, 'balanced' or a dict")
        return per_class[y]

    def _new_node(self, idx, y, w, depth) -> int:
        W = w[idx].sum()
        P = (w[idx] * y[idx]).sum()
        self.nodes_.append(Node(n_samples=len(idx), weight=W, value=P / W,
                                impurity=float(gini_binary(P, W)), depth=depth))
        return len(self.nodes_) - 1

    def _is_terminal(self, node: Node) -> bool:
        return ((self.max_depth is not None and node.depth >= self.max_depth)
                or node.n_samples < self.min_samples_split
                or node.n_samples < 2 * self.min_samples_leaf
                or node.impurity <= 1e-12)

    def _grow(self, X, y, w):
        root_idx = np.arange(len(y))
        stack = [(self._new_node(root_idx, y, w, 0), root_idx)]
        while stack:
            node_id, idx = stack.pop()
            node = self.nodes_[node_id]
            if self._is_terminal(node):
                continue
            split = self._find_best_split(X, y, w, idx)
            if split is None:
                continue
            node.feature = split["feature"]
            node.is_categorical = split["is_categorical"]
            node.threshold = split.get("threshold", np.nan)
            node.left_mask = split.get("left_mask")
            node.seen_mask = split.get("seen_mask")
            node.default_left = split.get("default_left", True)

            go_left = self._go_left(node, X[idx, node.feature])
            left_idx, right_idx = idx[go_left], idx[~go_left]
            node.left = self._new_node(left_idx, y, w, node.depth + 1)
            node.right = self._new_node(right_idx, y, w, node.depth + 1)
            stack.append((node.right, right_idx))
            stack.append((node.left, left_idx))

    def _find_best_split(self, X, y, w, idx):
        wn = w[idx]
        pw = wn * y[idx]
        best = None
        for j in range(self.n_features_in_):
            x = X[idx, j]
            if self.categorical_[j]:
                cand = self._categorical_candidate(x, wn, pw, j)
            else:
                cand = self._numeric_candidate(x, wn, pw)
            if cand is None:
                continue
            if best is None or cand["cost"] < best["cost"] - 1e-9 * max(1.0, best["cost"]):
                cand["feature"] = j
                best = cand
        return best

    def _numeric_candidate(self, x, wn, pw):
        order = np.argsort(x, kind="stable")
        xs = x[order]
        if xs[-1] <= xs[0] + FEATURE_THRESHOLD:
            return None
        allowed = xs[1:] > xs[:-1] + FEATURE_THRESHOLD
        i, cost = _best_prefix_split(wn[order], pw[order], np.ones(len(xs)),
                                     self.min_samples_leaf, allowed)
        if i < 0:
            return None
        threshold = xs[i - 1] / 2.0 + xs[i] / 2.0
        if threshold == xs[i]:          # float rounding guard (same rule as sklearn)
            threshold = xs[i - 1]
        return {"cost": cost, "is_categorical": False, "threshold": float(threshold)}

    def _categorical_candidate(self, x, wn, pw, j):
        codes = x.astype(np.int64)
        K = int(self.n_categories_[j])
        tot = np.bincount(codes, weights=wn, minlength=K)
        pos = np.bincount(codes, weights=pw, minlength=K)
        cnt = np.bincount(codes, minlength=K).astype(np.float64)
        present = np.flatnonzero(cnt > 0)
        if len(present) < 2:
            return None
        order = present[np.argsort(pos[present] / tot[present], kind="stable")]
        i, cost = _best_prefix_split(tot[order], pos[order], cnt[order], self.min_samples_leaf)
        if i < 0:
            return None
        left_mask = np.zeros(K, dtype=bool)
        left_mask[order[:i]] = True
        seen_mask = np.zeros(K, dtype=bool)
        seen_mask[present] = True
        default_left = bool(tot[order[:i]].sum() >= tot[order[i:]].sum())
        return {"cost": cost, "is_categorical": True, "left_mask": left_mask,
                "seen_mask": seen_mask, "default_left": default_left}

    @staticmethod
    def _go_left(node: Node, x):
        if not node.is_categorical:
            return x <= node.threshold
        codes = x.astype(np.int64)
        known = (codes >= 0) & (codes < len(node.left_mask))
        safe = np.where(known, codes, 0)
        seen = known & node.seen_mask[safe]
        return np.where(seen, node.left_mask[safe], node.default_left)

    # -------------------------------------------------------------- predict
    def _check_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features")
        return X

    def apply(self, X):
        X = self._check_X(X)
        leaf = np.empty(len(X), dtype=np.int64)
        stack = [(0, np.arange(len(X)))]
        while stack:
            node_id, rows = stack.pop()
            if len(rows) == 0:
                continue
            node = self.nodes_[node_id]
            if node.is_leaf:
                leaf[rows] = node_id
                continue
            go_left = self._go_left(node, X[rows, node.feature])
            stack.append((node.left, rows[go_left]))
            stack.append((node.right, rows[~go_left]))
        return leaf

    def predict_proba(self, X):
        values = np.array([node.value for node in self.nodes_])
        p1 = values[self.apply(X)]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)

    # ----------------------------------------------------------- summaries
    @property
    def n_nodes_(self) -> int:
        return len(self.nodes_)

    @property
    def n_leaves_(self) -> int:
        return sum(node.is_leaf for node in self.nodes_)

    @property
    def depth_(self) -> int:
        return max(node.depth for node in self.nodes_)

    def _compute_importances(self):
        importance = np.zeros(self.n_features_in_)
        for node in self.nodes_:
            if node.is_leaf:
                continue
            left, right = self.nodes_[node.left], self.nodes_[node.right]
            importance[node.feature] += (node.weight * node.impurity
                                         - left.weight * left.impurity
                                         - right.weight * right.impurity)
        total = importance.sum()
        return importance / total if total > 0 else importance

    # ------------------------------------------------------------- pruning
    def _risk(self, node: Node) -> float:
        return node.weight / self.nodes_[0].weight * node.impurity

    def _reachable(self):
        order, stack = [], [0]
        while stack:
            node_id = stack.pop()
            order.append(node_id)
            node = self.nodes_[node_id]
            if not node.is_leaf:
                stack.extend((node.right, node.left))
        return order

    def _subtree_stats(self):
        """R(T_t) and |leaves(T_t)| for every node; children always have larger ids."""
        n = len(self.nodes_)
        risk = np.zeros(n)
        leaves = np.zeros(n, dtype=np.int64)
        for i in reversed(range(n)):
            node = self.nodes_[i]
            if node.is_leaf:
                risk[i], leaves[i] = self._risk(node), 1
            else:
                risk[i] = risk[node.left] + risk[node.right]
                leaves[i] = leaves[node.left] + leaves[node.right]
        return risk, leaves

    def _weakest_link(self):
        risk, leaves = self._subtree_stats()
        best_alpha, best_id = np.inf, -1
        for i in self._reachable():
            node = self.nodes_[i]
            if node.is_leaf:
                continue
            g = (self._risk(node) - risk[i]) / (leaves[i] - 1)
            if g < best_alpha:
                best_alpha, best_id = g, i
        return best_alpha, best_id

    def _collapse(self, node_id):
        node = self.nodes_[node_id]
        node.left = node.right = -1
        node.feature = -1

    def _compact(self):
        """Drop unreachable nodes and renumber so children keep larger ids than parents."""
        old_ids = sorted(self._reachable())
        new_id = {old: new for new, old in enumerate(old_ids)}
        nodes = [self.nodes_[old] for old in old_ids]
        for node in nodes:
            if not node.is_leaf:
                node.left, node.right = new_id[node.left], new_id[node.right]
        self.nodes_ = nodes

    def _prune(self, alpha):
        while True:
            g, node_id = self._weakest_link()
            if node_id < 0 or g > alpha:
                break
            self._collapse(node_id)
        self._compact()

    def cost_complexity_pruning_path(self, X, y):
        params = {**self.get_params(), "ccp_alpha": 0.0}
        tree = CARTClassifier(**params).fit(X, y)
        risk, _ = tree._subtree_stats()
        alphas, impurities = [0.0], [risk[0]]
        while not tree.nodes_[0].is_leaf:
            g, node_id = tree._weakest_link()
            tree._collapse(node_id)
            tree._compact()
            risk, _ = tree._subtree_stats()
            alphas.append(max(g, 0.0))
            impurities.append(risk[0])
        return np.array(alphas), np.array(impurities)

    # ------------------------------------------------------------ rules text
    def export_text(self, feature_names, category_names=None, max_depth=3) -> str:
        lines = []

        def describe(node, negate):
            name = feature_names[node.feature]
            if not node.is_categorical:
                op = ">" if negate else "<="
                return f"{name} {op} {node.threshold:.3f}"
            cats = np.flatnonzero(node.left_mask)
            if category_names is not None and name in category_names:
                labels = [str(category_names[name][c]) for c in cats]
            else:
                labels = [str(c) for c in cats]
            op = "not in" if negate else "in"
            return f"{name} {op} {{{', '.join(labels)}}}"

        def walk(node_id, level):
            node = self.nodes_[node_id]
            pad = "|   " * level
            if node.is_leaf or level >= max_depth:
                lines.append(f"{pad}P(>50K)={node.value:.3f} n={node.n_samples}")
                return
            lines.append(pad + describe(node, negate=False))
            walk(node.left, level + 1)
            lines.append(pad + describe(node, negate=True))
            walk(node.right, level + 1)

        walk(0, 0)
        return "\n".join(lines)
