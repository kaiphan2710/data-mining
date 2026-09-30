"""Metrics, threshold selection and cross-validated grid search shared by every tool."""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": skm.roc_auc_score(y_true, y_prob),
        "pr_auc": skm.average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = skm.precision_recall_curve(y_true, y_prob)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(precision), where=denom > 0)
    return float(thresholds[np.argmax(f1[:-1])])


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def out_of_fold_proba(make_model, params, X, y, folds):
    oof = np.full(len(y), np.nan)
    seconds = 0.0
    for k in np.unique(folds):
        train_mask, val_mask = folds != k, folds == k
        model = make_model(params)
        start = time.perf_counter()
        model.fit(X[train_mask], y[train_mask])
        seconds += time.perf_counter() - start
        oof[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    return oof, seconds


def cross_validate_grid(make_model, grid, X, y, folds, log=print):
    rows, best = [], None
    for params in expand_grid(grid):
        oof, seconds = out_of_fold_proba(make_model, params, X, y, folds)
        fold_ap = [skm.average_precision_score(y[folds == k], oof[folds == k])
                   for k in np.unique(folds)]
        row = {
            **{k: ("none" if v is None else v) for k, v in params.items()},
            "cv_pr_auc_mean": float(np.mean(fold_ap)),
            "cv_pr_auc_std": float(np.std(fold_ap)),
            "fit_seconds_total": seconds,
        }
        rows.append(row)
        log(row)
        if best is None or row["cv_pr_auc_mean"] > best[0]["cv_pr_auc_mean"]:
            best = (row, params, oof)
    return pd.DataFrame(rows), best[1], best[2]
