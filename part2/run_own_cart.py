"""Own implementation: CART written from scratch in part2/cart.py."""
import argparse
import time

import pandas as pd

from part2 import config
from part2.cart import CARTClassifier
from part2.evaluation import best_f1_threshold, compute_metrics, cross_validate_grid
from part2.prep import OrdinalCodes, load_model_ready
from part2.results_io import printable_params, save_predictions, save_run, save_table, save_text

TOOL = "own_cart"
SMALL_GRID = {"max_depth": [8, 11], "min_samples_leaf": [50, 200],
              "class_weight": [None, "balanced"]}


def fit_and_report(experiment, params, threshold, codes, cat_idx, X_tr, y_tr, X_te, y_te):
    model = CARTClassifier(categorical_features=cat_idx, **params)
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prob = model.predict_proba(X_te)[:, 1]
    predict_seconds = time.perf_counter() - start

    save_predictions(TOOL, experiment, y_te, prob)
    importance = pd.Series(model.feature_importances_, index=codes.columns_)
    save_table(importance.sort_values(ascending=False).rename("importance")
               .rename_axis("feature").reset_index(), "importance", f"{TOOL}_{experiment}.csv")
    save_text(model.export_text(codes.columns_, codes.categories_, max_depth=3),
              "trees", f"{TOOL}_{experiment}.txt")
    info = {
        "tool": TOOL, "experiment": experiment, "params": printable_params(params),
        "threshold": threshold, "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds, "n_nodes": model.n_nodes_,
        "n_leaves": model.n_leaves_, "depth": model.depth_,
        "test_metrics": compute_metrics(y_te, prob, threshold),
    }
    save_run(TOOL, experiment, info)
    print(experiment, info["test_metrics"])
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "all"], default="all")
    parser.add_argument("--grid", choices=["full", "small"], default="full")
    args = parser.parse_args()

    data = load_model_ready()
    codes = OrdinalCodes().fit(data.X_train)
    X_tr, X_te = codes.transform(data.X_train), codes.transform(data.X_test)
    cat_idx = codes.categorical_indices

    if args.experiment in ("A", "all"):
        fit_and_report("expA", config.EXP_A_PARAMS, 0.5, codes, cat_idx,
                       X_tr, data.y_train, X_te, data.y_test)
    if args.experiment in ("B", "all"):
        grid = config.PARAM_GRID if args.grid == "full" else SMALL_GRID
        table, best_params, oof = cross_validate_grid(
            lambda p: CARTClassifier(categorical_features=cat_idx, **p),
            grid, X_tr, data.y_train, data.folds)
        save_table(table, "tuning", f"{TOOL}_cv.csv")
        threshold = best_f1_threshold(data.y_train, oof)
        print("best params:", best_params, "threshold:", round(threshold, 4))
        fit_and_report("expB", best_params, threshold, codes, cat_idx,
                       X_tr, data.y_train, X_te, data.y_test)


if __name__ == "__main__":
    main()
