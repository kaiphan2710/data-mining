"""Correctness evidence for the report: with every feature treated as an ordered
numeric code, the own CART and scikit-learn must grow (almost) the same tree."""
import json

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from part2 import config
from part2.cart import CARTClassifier
from part2.prep import OrdinalCodes, load_model_ready


def main():
    data = load_model_ready()
    codes = OrdinalCodes().fit(data.X_train)
    X_tr, X_te = codes.transform(data.X_train), codes.transform(data.X_test)
    params = config.EXP_A_PARAMS

    own = CARTClassifier(**params).fit(X_tr, data.y_train)
    sk = DecisionTreeClassifier(criterion="gini", random_state=config.SEED,
                                **params).fit(X_tr, data.y_train)
    p_own = own.predict_proba(X_te)[:, 1]
    p_sk = sk.predict_proba(X_te)[:, 1]
    result = {
        "params": {k: ("none" if v is None else v) for k, v in params.items()},
        "probability_agreement": float(np.mean(np.isclose(p_own, p_sk, atol=1e-9))),
        "label_agreement": float(np.mean((p_own >= 0.5) == (p_sk >= 0.5))),
        "own_n_leaves": own.n_leaves_, "sklearn_n_leaves": int(sk.get_n_leaves()),
        "own_root": [codes.columns_[own.nodes_[0].feature], own.nodes_[0].threshold],
        "sklearn_root": [codes.columns_[sk.tree_.feature[0]], float(sk.tree_.threshold[0])],
        "max_abs_importance_diff": float(np.max(np.abs(own.feature_importances_
                                                       - sk.feature_importances_))),
    }
    path = config.RESULTS_DIR / "equivalence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
