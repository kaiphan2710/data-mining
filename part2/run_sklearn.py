"""Tool 1 (Python platform): scikit-learn DecisionTreeClassifier (optimised CART)."""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from part2 import config
from part2.evaluation import best_f1_threshold, compute_metrics, cross_validate_grid
from part2.prep import categorical_columns, load_model_ready
from part2.results_io import printable_params, save_predictions, save_run, save_table, save_text

TOOL = "sklearn"


def build_encoder(columns: list[str]) -> ColumnTransformer:
    cat = categorical_columns(columns)
    num = [c for c in columns if c not in cat]
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), cat),
         ("num", "passthrough", num)],
        sparse_threshold=1.0,
    )


def make_model(params: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(criterion="gini", random_state=config.SEED, **params)


def column_owners(encoder: ColumnTransformer) -> np.ndarray:
    onehot = encoder.named_transformers_["cat"]
    owners = [col for col, cats in zip(onehot.feature_names_in_, onehot.categories_)
              for _ in cats]
    owners += list(encoder.transformers_[1][2])
    return np.array(owners)


def fit_and_report(experiment, params, threshold, encoder, X_tr, y_tr, X_te, y_te):
    model = make_model(params)
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prob = model.predict_proba(X_te)[:, 1]
    predict_seconds = time.perf_counter() - start

    save_predictions(TOOL, experiment, y_te, prob)
    importance = (pd.Series(model.feature_importances_, index=column_owners(encoder))
                  .groupby(level=0).sum().sort_values(ascending=False))
    save_table(importance.rename("importance").rename_axis("feature").reset_index(),
               "importance", f"{TOOL}_{experiment}.csv")
    save_text(export_text(model, feature_names=list(encoder.get_feature_names_out()),
                          max_depth=3), "trees", f"{TOOL}_{experiment}.txt")
    info = {
        "tool": TOOL, "experiment": experiment, "params": printable_params(params),
        "threshold": threshold, "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds, "n_nodes": int(model.tree_.node_count),
        "n_leaves": int(model.get_n_leaves()), "depth": int(model.get_depth()),
        "test_metrics": compute_metrics(y_te, prob, threshold),
    }
    save_run(TOOL, experiment, info)
    print(experiment, info["test_metrics"])
    return info


def main():
    data = load_model_ready()
    encoder = build_encoder(list(data.X_train.columns)).fit(data.X_train)
    X_tr = encoder.transform(data.X_train).tocsr()
    X_te = encoder.transform(data.X_test).tocsr()
    print("one-hot design matrix:", X_tr.shape)

    fit_and_report("expA", config.EXP_A_PARAMS, 0.5, encoder, X_tr, data.y_train,
                   X_te, data.y_test)

    table, best_params, oof = cross_validate_grid(
        make_model, config.PARAM_GRID, X_tr, data.y_train, data.folds)
    save_table(table, "tuning", f"{TOOL}_cv.csv")
    threshold = best_f1_threshold(data.y_train, oof)
    print("best params:", best_params, "threshold:", round(threshold, 4))
    fit_and_report("expB", best_params, threshold, encoder, X_tr, data.y_train,
                   X_te, data.y_test)


if __name__ == "__main__":
    main()
