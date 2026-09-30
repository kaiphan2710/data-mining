import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from part2.cart import CARTClassifier, gini_binary


def _numeric_data(n=3000, seed=0, weights=None):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5,
                               random_state=seed, weights=weights)
    # sklearn trees work in float32; use identical values for exact comparisons.
    return X.astype(np.float32).astype(np.float64), y


def test_gini_binary():
    assert gini_binary(5.0, 10.0) == pytest.approx(0.5)
    assert gini_binary(0.0, 10.0) == pytest.approx(0.0)


def test_simple_numeric_threshold():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = CARTClassifier(max_depth=1).fit(X, y)
    assert tree.nodes_[0].threshold == pytest.approx(2.5)
    assert tree.predict(X).tolist() == [0, 0, 1, 1]


def test_categorical_subset_split_beats_numeric_threshold():
    codes = np.tile([0, 1, 2, 3], 50).astype(float).reshape(-1, 1)
    y = np.isin(codes[:, 0], [0, 2]).astype(int)
    cat = CARTClassifier(max_depth=1, categorical_features=[0]).fit(codes, y)
    num = CARTClassifier(max_depth=1).fit(codes, y)
    assert (cat.predict(codes) == y).mean() == 1.0
    assert (num.predict(codes) == y).mean() < 1.0
    assert set(np.flatnonzero(cat.nodes_[0].left_mask)) in ({0, 2}, {1, 3})


def test_unseen_category_goes_to_larger_child():
    codes = np.array([0] * 60 + [1] * 30 + [2] * 10, dtype=float).reshape(-1, 1)
    y = np.array([0] * 60 + [1] * 40)
    tree = CARTClassifier(max_depth=1, categorical_features=[0]).fit(codes, y)
    root = tree.nodes_[0]
    larger = tree.nodes_[root.left if root.default_left else root.right]
    assert larger.n_samples == 60
    proba = tree.predict_proba(np.array([[5.0], [-1.0]]))[:, 1]
    assert np.allclose(proba, larger.value)


def test_limits_are_respected():
    X, y = _numeric_data()
    tree = CARTClassifier(max_depth=3, min_samples_leaf=40).fit(X, y)
    assert tree.depth_ <= 3
    assert all(n.n_samples >= 40 for n in tree.nodes_ if n.is_leaf)


def test_single_class_is_single_leaf():
    X = np.random.default_rng(0).normal(size=(20, 3))
    tree = CARTClassifier().fit(X, np.zeros(20, dtype=int))
    assert tree.n_nodes_ == 1
    assert np.allclose(tree.predict_proba(X[:2]), [[1, 0], [1, 0]])


def test_min_samples_leaf_too_large_gives_root_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0] * 5 + [1] * 5)
    assert CARTClassifier(min_samples_leaf=6).fit(X, y).n_nodes_ == 1


def test_unlimited_depth_no_recursion_error():
    X, y = _numeric_data(n=3000, seed=1)
    tree = CARTClassifier().fit(X, y)
    assert (tree.predict(X) == y).mean() == 1.0


def test_rejects_bad_inputs():
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError):
        CARTClassifier().fit(X, np.array([0, 2]))
    with pytest.raises(ValueError):
        CARTClassifier(categorical_features=[0]).fit(np.array([[-1.0], [1.0]]), np.array([0, 1]))


# Leaf size 20 / depth 4 keeps nodes away from near-pure regions where several
# features tie on Gini cost; sklearn breaks such ties by random feature order.
def test_matches_sklearn_on_numeric_data():
    X, y = _numeric_data()
    params = dict(max_depth=4, min_samples_leaf=20)
    own = CARTClassifier(**params).fit(X, y)
    sk = DecisionTreeClassifier(random_state=0, **params).fit(X, y)
    assert own.n_leaves_ == sk.get_n_leaves()
    assert own.nodes_[0].feature == sk.tree_.feature[0]
    assert np.allclose(own.predict_proba(X), sk.predict_proba(X))
    assert np.allclose(own.feature_importances_, sk.feature_importances_, atol=1e-8)


def test_balanced_class_weight_matches_sklearn():
    X, y = _numeric_data(weights=[0.9])
    params = dict(max_depth=4, min_samples_leaf=20, class_weight="balanced")
    own = CARTClassifier(**params).fit(X, y)
    sk = DecisionTreeClassifier(random_state=0, **params).fit(X, y)
    assert own.n_leaves_ == sk.get_n_leaves()
    assert np.allclose(own.predict_proba(X), sk.predict_proba(X))


def test_pruning_matches_sklearn_leaf_counts():
    # Pruned from the tie-free base tree used above (fully grown trees contain exact
    # Gini ties that sklearn breaks by random feature order).
    X, y = _numeric_data()
    base = dict(max_depth=4, min_samples_leaf=20)
    alphas = DecisionTreeClassifier(random_state=0, **base).cost_complexity_pruning_path(X, y).ccp_alphas
    gaps = np.flatnonzero(np.diff(alphas) > 1e-9)
    mids = (alphas[gaps] + alphas[gaps + 1]) / 2
    assert len(mids) > 3
    for alpha in mids:
        sk = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha, **base).fit(X, y)
        own = CARTClassifier(ccp_alpha=alpha, **base).fit(X, y)
        assert own.n_leaves_ == sk.get_n_leaves(), alpha
        assert np.allclose(own.predict_proba(X), sk.predict_proba(X))


def test_pruned_importances_sum_to_one():
    X, y = _numeric_data(n=800, seed=3)
    own = CARTClassifier(ccp_alpha=0.002).fit(X, y)
    assert own.feature_importances_.sum() == pytest.approx(1.0)
    reachable = own.apply(X)
    assert all(own.nodes_[i].is_leaf for i in np.unique(reachable))


def test_pruning_path_is_monotone():
    X, y = _numeric_data(n=500, seed=4)
    alphas, impurities = CARTClassifier().cost_complexity_pruning_path(X, y)
    assert alphas[0] == 0.0
    assert np.all(np.diff(alphas) >= -1e-12)
    assert np.all(np.diff(impurities) >= -1e-12)


def test_export_text_shows_category_names():
    codes = np.tile([0, 1, 2, 3], 50).astype(float).reshape(-1, 1)
    y = np.isin(codes[:, 0], [0, 2]).astype(int)
    tree = CARTClassifier(max_depth=1, categorical_features=[0]).fit(codes, y)
    text = tree.export_text(["job"], {"job": np.array(["a", "b", "c", "d"], dtype=object)})
    assert "job in {" in text and ("a, c" in text or "b, d" in text)
