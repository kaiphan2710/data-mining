"""Check beyond the tuning grid: max_depth 17 and 20 for the two Python implementations."""
from part2.cart import CARTClassifier
from part2.evaluation import cross_validate_grid
from part2.prep import OrdinalCodes, load_model_ready
from part2.results_io import save_table
from part2.run_sklearn import build_encoder, make_model

# Same selected leaf size and weighting as Experiment B; depth 14 is the edge of the grid.
GRID = {"max_depth": [14, 17, 20], "min_samples_leaf": [50], "class_weight": [None]}


def main():
    data = load_model_ready()
    codes = OrdinalCodes().fit(data.X_train)
    X_codes = codes.transform(data.X_train)
    own, _, _ = cross_validate_grid(
        lambda p: CARTClassifier(categorical_features=codes.categorical_indices, **p),
        GRID, X_codes, data.y_train, data.folds)
    save_table(own, "tuning", "own_cart_depth_check.csv")

    encoder = build_encoder(list(data.X_train.columns)).fit(data.X_train)
    X_onehot = encoder.transform(data.X_train).tocsr()
    sk, _, _ = cross_validate_grid(make_model, GRID, X_onehot, data.y_train, data.folds)
    save_table(sk, "tuning", "sklearn_depth_check.csv")


if __name__ == "__main__":
    main()
