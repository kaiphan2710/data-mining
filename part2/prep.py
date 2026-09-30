"""Shared preprocessing: cleaned CSVs -> model-ready files used by every tool."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from part2 import config


def load_cleaned(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(config.DROP_COLS) | {config.TARGET, config.FOLD}
    return [c for c in df.columns if c not in excluded]


def categorical_columns(features: list[str]) -> list[str]:
    return [c for c in features if c not in config.NUMERIC_COLS]


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_columns(df)].copy()
    for col in config.CODED_CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(str)
    y = df[config.TARGET].to_numpy(dtype=np.int64)
    return X, y


def assign_folds(y: np.ndarray, n_folds: int = config.N_FOLDS,
                 seed: int = config.SEED) -> np.ndarray:
    folds = np.empty(len(y), dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[val_idx] = k
    return folds


def export_model_ready(out_dir: Path = config.MODEL_READY_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr = split_features(load_cleaned(config.CLEAN_TRAIN))
    X_te, y_te = split_features(load_cleaned(config.CLEAN_TEST))
    assert list(X_tr.columns) == list(X_te.columns), "train/test schema mismatch"

    X_tr[config.TARGET] = y_tr
    X_tr[config.FOLD] = assign_folds(y_tr)
    X_te[config.TARGET] = y_te

    X_tr.to_csv(out_dir / "train.csv", index=False)
    X_te.to_csv(out_dir / "test.csv", index=False)
    return X_tr.shape, X_te.shape


@dataclass
class ModelData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    folds: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def load_model_ready(in_dir: Path = config.MODEL_READY_DIR) -> ModelData:
    train = pd.read_csv(in_dir / "train.csv", keep_default_na=False)
    test = pd.read_csv(in_dir / "test.csv", keep_default_na=False)
    X_tr, y_tr = split_features(train)
    X_te, y_te = split_features(test)
    return ModelData(X_tr, y_tr, train[config.FOLD].to_numpy(np.int64), X_te, y_te)


class OrdinalCodes:
    """Map categorical strings to integer codes learned on training data (unseen -> -1)."""

    def fit(self, X: pd.DataFrame) -> "OrdinalCodes":
        self.columns_ = list(X.columns)
        self.categorical_ = categorical_columns(self.columns_)
        self.categories_ = {
            c: np.unique(X[c].to_numpy(dtype=object)) for c in self.categorical_
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert list(X.columns) == self.columns_, "column order differs from fit"
        out = np.empty((len(X), len(self.columns_)), dtype=np.float64)
        for j, col in enumerate(self.columns_):
            if col in self.categories_:
                cats = self.categories_[col]
                values = X[col].to_numpy(dtype=object)
                pos = np.clip(np.searchsorted(cats, values), 0, len(cats) - 1)
                out[:, j] = np.where(cats[pos] == values, pos, -1)
            else:
                out[:, j] = X[col].to_numpy(dtype=np.float64)
        return out

    @property
    def categorical_indices(self) -> list[int]:
        return [self.columns_.index(c) for c in self.categorical_]


if __name__ == "__main__":
    train_shape, test_shape = export_model_ready()
    print("model_ready train:", train_shape, "test:", test_shape)
