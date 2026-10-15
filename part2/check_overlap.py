"""Audit feature-vector overlap without fitting or selecting models on test data."""
import json

import pandas as pd

from part2 import config
from part2.prep import feature_columns, load_cleaned


def membership(train, test, columns):
    # MultiIndex compares complete tuples, avoiding probabilistic hash matching.
    reference = pd.MultiIndex.from_frame(train[columns])
    queries = pd.MultiIndex.from_frame(test[columns])
    matches = queries.isin(reference)
    return {'n_columns': len(columns), 'matched_test_rows': int(matches.sum()),
            'total_test_rows': len(test), 'fraction': float(matches.mean())}


def main():
    train, test = load_cleaned(config.CLEAN_TRAIN), load_cleaned(config.CLEAN_TEST)
    result = {
        'model_features_only': membership(train, test, feature_columns(train)),
        'features_with_instance_weight': membership(
            train, test, [c for c in train if c != config.TARGET]),
        'full_cleaned_rows': membership(train, test, list(train.columns)),
        'interpretation': 'Identical feature vectors do not establish identical people or quantify '
                          'score inflation. No models are refitted by this audit.',
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / 'overlap.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
