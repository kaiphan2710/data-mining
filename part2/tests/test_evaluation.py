import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from part2 import config, results_io
from part2.evaluation import (
    best_f1_threshold,
    compute_metrics,
    cross_validate_grid,
    expand_grid,
)


def test_compute_metrics_toy():
    # Same toy case is used in part2/r/test_metrics.R (AP = 0.8333...).
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx(5 / 6)


def test_threshold_is_inclusive():
    m = compute_metrics([0, 1], [0.2, 0.5], threshold=0.5)
    assert m["tp"] == 1


def test_best_f1_threshold():
    assert best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.8)


def test_expand_grid():
    grid = expand_grid({"a": [1, 2], "b": [None, "x"]})
    assert len(grid) == 4
    assert {"a": 2, "b": None} in grid


def test_cross_validate_grid_with_sklearn_tree():
    X, y = make_classification(n_samples=500, n_features=6, random_state=0)
    folds = np.arange(len(y)) % 5
    grid = {"max_depth": [1, 3], "min_samples_leaf": [1], "class_weight": [None]}
    table, best, oof = cross_validate_grid(
        lambda p: DecisionTreeClassifier(random_state=0, **p), grid, X, y, folds,
        log=lambda row: None)
    assert len(table) == 2
    assert set(table["class_weight"]) == {"none"}
    assert best in expand_grid(grid)
    assert not np.isnan(oof).any()
    assert table["cv_pr_auc_mean"].between(0, 1).all()


def test_save_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    path = results_io.save_predictions("tool", "expA", [0, 1], [0.2, 0.7])
    df = pd.read_csv(path)
    assert list(df.columns) == ["y_true", "y_prob"]
    assert path == tmp_path / "predictions" / "tool_expA.csv"


def test_best_f1_threshold_tie_picks_lowest():
    # F1 ties at thresholds 0.9 and 0.6; part2/r/test_metrics.R expects the same 0.6.
    assert best_f1_threshold([1, 0, 0, 1, 0], [0.9, 0.8, 0.7, 0.6, 0.5]) == pytest.approx(0.6)
