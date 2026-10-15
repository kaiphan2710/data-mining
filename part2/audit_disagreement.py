"""Explain every test row on which the own CART and rpart disagree (Experiment A).

The complete rpart tree (results/trees/rpart_expA_full.txt, written by
part2/r/rpart_diagnostics.R) is aligned node by node with the own tree, using the
training rows that reach each node. A differing test row is then attributed to
(1) unseen-category routing: its category was absent at a node of the own tree during
    training (the own CART sends it to the heavier child, rpart stops it at that node), or
(2) a threshold tie: a test value equal to a numeric threshold (the own CART tests
    x <= c, rpart tests x < c), or
(3) a divergent node: a node where the two programs chose different splits.

Run from the project root: py -3.12 -m part2.audit_disagreement
"""
import json
import re

import numpy as np
import pandas as pd

from part2 import config
from part2.cart import CARTClassifier, gini_binary
from part2.prep import OrdinalCodes, load_model_ready

LINE = re.compile(r"^\s*(\d+)\) (.+) (\d+)\s+(\d+) ([01]) \(([0-9.e+-]+) ([0-9.e+-]+)\)\s*(\*)?\s*$")


def read_rpart_tree(path):
    """node id -> condition text that sends rows into that node ('root' for node 1)."""
    tree = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line)
        if m:
            tree[int(m.group(1))] = m.group(2)
    return tree


def rpart_condition(text, X, codes):
    """Boolean mask of the rows of X that satisfy one rpart split condition."""
    if "=" in text and not re.search(r"(<|>=)", text.split("=")[0] + "="):
        var, levels = text.split("=", 1)
        names = [str(c) for c in codes.categories_[var]]
        wanted = [names.index(level) for level in levels.split(",")]
        return np.isin(X[:, codes.columns_.index(var)].astype(np.int64), wanted)
    var, op, value = re.match(r"(\S+?)(<|>=)\s*(\S+)$", text).groups()
    x = X[:, codes.columns_.index(var)]
    return x < float(value) if op == "<" else x >= float(value)


def node_rows(tree, X):
    """Row indices of X that reach each node of the own tree."""
    rows, stack = {0: np.arange(len(X))}, [0]
    while stack:
        i = stack.pop()
        node = tree.nodes_[i]
        if node.is_leaf:
            continue
        go_left = tree._go_left(node, X[rows[i], node.feature])
        rows[node.left], rows[node.right] = rows[i][go_left], rows[i][~go_left]
        stack += [node.left, node.right]
    return rows


def split_cost(y, idx, left):
    l, r = idx[left], idx[~left]
    return float(len(l) * gini_binary(y[l].sum(), len(l)) + len(r) * gini_binary(y[r].sum(), len(r)))


def main():
    data = load_model_ready()
    codes = OrdinalCodes().fit(data.X_train)
    X_tr, X_te = codes.transform(data.X_train), codes.transform(data.X_test)
    y = data.y_train
    own = CARTClassifier(categorical_features=codes.categorical_indices,
                         **config.EXP_A_PARAMS).fit(X_tr, y)
    rp = read_rpart_tree(config.RESULTS_DIR / "trees" / "rpart_expA_full.txt")

    train_rows = node_rows(own, X_tr)
    divergent, matched, stack = [], 0, [(0, 1)]
    while stack:
        i, k = stack.pop()
        node, idx = own.nodes_[i], train_rows[i]
        rp_leaf = 2 * k not in rp
        if node.is_leaf or rp_leaf:
            matched += node.is_leaf and rp_leaf
            if node.is_leaf != rp_leaf:
                divergent.append({"own_node": i, "rpart_node": k, "reason": "leaf in one tree only"})
            continue
        own_left = own._go_left(node, X_tr[idx, node.feature])
        rp_left = rpart_condition(rp[2 * k], X_tr[idx], codes)
        if np.array_equal(own_left, rp_left) or np.array_equal(own_left, ~rp_left):
            matched += 1
            a, b = (2 * k, 2 * k + 1) if np.array_equal(own_left, rp_left) else (2 * k + 1, 2 * k)
            stack += [(node.left, a), (node.right, b)]
            continue
        own_var, rp_var = codes.columns_[node.feature], re.split(r"=|<|>=", rp[2 * k])[0]
        record = {"own_node": i, "rpart_node": k, "n": int(len(idx)),
                  "own_split": own_var, "rpart_split": rp_var,
                  "own_cost": round(split_cost(y, idx, own_left), 6),
                  "rpart_cost": round(split_cost(y, idx, rp_left), 6)}
        for side, var in (("own", own_var), ("rpart", rp_var)):
            if var in codes.categories_:  # categories present at the node sharing one positive rate
                c = X_tr[idx, codes.columns_.index(var)].astype(np.int64)
                rates = [y[idx][c == v].mean() for v in np.unique(c)]
                record[f"{side}_tied_category_rates"] = len(rates) - len(set(rates))
        divergent.append(record)

    test_rows = node_rows(own, X_te)
    p_own = own.predict_proba(X_te)[:, 1]
    p_rp = pd.read_csv(config.RESULTS_DIR / "predictions" / "rpart_expA.csv")["y_prob"].to_numpy()
    differ = ~np.isclose(p_own, p_rp, atol=1e-9)
    unseen = np.zeros(len(X_te), dtype=bool)
    on_threshold = np.zeros(len(X_te), dtype=bool)
    via_divergent = np.zeros(len(X_te), dtype=bool)
    for d in divergent:
        via_divergent[test_rows[d["own_node"]]] = True
    for i, idx in test_rows.items():
        node = own.nodes_[i]
        if node.is_leaf:
            continue
        if node.is_categorical:
            c = X_te[idx, node.feature].astype(np.int64)
            known = (c >= 0) & (c < len(node.seen_mask))
            unseen[idx[~known | ~node.seen_mask[np.where(known, c, 0)]]] = True
        else:
            on_threshold[idx[X_te[idx, node.feature] == node.threshold]] = True

    result = {
        "matched_internal_nodes": int(matched - sum(n.is_leaf for n in own.nodes_)),
        "own_internal_nodes": int(sum(not n.is_leaf for n in own.nodes_)),
        "divergent_nodes": divergent,
        "differing_rows": int(differ.sum()),
        "differing_unseen_category": int((differ & unseen).sum()),
        "differing_on_threshold": int((differ & on_threshold & ~unseen).sum()),
        "differing_via_divergent_node_only": int((differ & via_divergent & ~unseen
                                                  & ~on_threshold).sum()),
        "differing_unexplained": int((differ & ~via_divergent & ~unseen & ~on_threshold).sum()),
    }
    path = config.RESULTS_DIR / "disagreement_audit.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
