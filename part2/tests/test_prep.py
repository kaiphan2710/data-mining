import numpy as np
import pandas as pd
import pytest

from part2 import config
from part2.prep import (
    OrdinalCodes,
    assign_folds,
    categorical_columns,
    load_model_ready,
    split_features,
)


def _toy_frame():
    return pd.DataFrame({
        "age": [20, 30, 40, 50],
        "sex": ["Male", "Female", "Female", "Male"],
        "year": [94, 95, 94, 95],
        "instance_weight": [1.0, 2.0, 3.0, 4.0],
        "income_class": [0, 1, 0, 1],
    })


def test_split_features_drops_weight_and_target():
    X, y = split_features(_toy_frame())
    assert list(X.columns) == ["age", "sex", "year"]
    assert y.tolist() == [0, 1, 0, 1]


def test_coded_columns_become_strings():
    X, _ = split_features(_toy_frame())
    assert X["year"].map(type).eq(str).all()


def test_categorical_columns_excludes_numeric():
    assert categorical_columns(["age", "sex", "year"]) == ["sex", "year"]


def test_ordinal_codes_unseen_is_minus_one():
    X, _ = split_features(_toy_frame())
    enc = OrdinalCodes().fit(X)
    new = X.copy()
    new.loc[0, "sex"] = "Unknown-sex"
    out = enc.transform(new)
    assert out[0, 1] == -1
    assert out[1, 1] == 0            # "Female" sorts first
    assert out[:, 0].tolist() == [20, 30, 40, 50]
    assert enc.categorical_indices == [1, 2]


def test_assign_folds_is_stratified():
    y = np.array([1] * 100 + [0] * 900)
    folds = assign_folds(y, n_folds=5, seed=42)
    for k in range(5):
        assert (y[folds == k] == 1).sum() == 20
        assert (folds == k).sum() == 200


@pytest.mark.skipif(not (config.MODEL_READY_DIR / "train.csv").exists(),
                    reason="run `py -3.12 -m part2.prep` first")
def test_model_ready_schema():
    data = load_model_ready()
    assert data.X_train.shape == (196294, 40)
    assert data.X_test.shape == (98879, 40)
    assert list(data.X_train.columns) == list(data.X_test.columns)
    assert "instance_weight" not in data.X_train.columns
    assert config.TARGET not in data.X_train.columns
    assert sorted(np.unique(data.folds)) == [0, 1, 2, 3, 4]
    assert len(categorical_columns(list(data.X_train.columns))) == 33
