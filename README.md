# Predicting High-Income Individuals with CART

A from-scratch CART decision tree (Python/NumPy) compared with scikit-learn and rpart on the
UCI Census-Income (KDD) dataset, predicting whether a person earns more than $50,000.
Griffith University 3031ICT / 3804ICT, Trimester 2, 2026.

The same CART algorithm is run in three implementations under one shared protocol (identical
model-ready files, one stratified 5-fold assignment and one metric function):

| Implementation | Platform | Nominal attributes | Class weighting |
|---|---|---|---|
| Own CART (`part2/cart.py`) | Python 3.12 + NumPy | Native subset split (ordering theorem) | Sample weights n / (2·n_c) |
| scikit-learn `DecisionTreeClassifier` | Python 3.12 | One-hot (503 + 7 numeric columns) | `class_weight="balanced"` |
| rpart `rpart(method = "class")` | R 4.5.0 | Native factor subset split | `parms$prior = c(0.5, 0.5)` |

## Headline results (test set, >$50K class)

| Implementation | Exp. | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Own CART | A | 0.500 | 0.710 | 0.380 | 0.495 | 0.573 | 0.931 |
| scikit-learn | A | 0.500 | 0.747 | 0.320 | 0.448 | 0.559 | 0.924 |
| rpart | A | 0.500 | 0.711 | 0.379 | 0.495 | 0.575 | 0.931 |
| Own CART | B | 0.315 | 0.558 | 0.571 | 0.565 | 0.589 | 0.924 |
| scikit-learn | B | 0.311 | 0.571 | 0.591 | 0.581 | 0.612 | 0.935 |
| rpart | B | 0.320 | 0.566 | 0.571 | 0.569 | 0.592 | 0.924 |

Experiment A uses fixed settings (max_depth 8, min_samples_leaf 50, no weighting, threshold 0.5).
Experiment B tunes a 24-configuration grid with 5-fold CV on PR-AUC, then picks the F1-optimal
threshold on out-of-fold predictions. With identical ordered-code inputs, the own CART and
scikit-learn grow the same tree (111 leaves, identical test probabilities); with native nominal
splits the own CART and rpart agree on 99.71% of test probabilities.

## Layout

```
cleaning_visualization.ipynb   Part I: data cleaning and exploratory figures
convert.py                     Part I helper: raw UCI files -> CSV with column names
data/visualizations/           Part I figures
raw/census-income.names        UCI dataset description (raw .data/.test are downloaded)
part2/
  cart.py                      Own CART implementation (no ML library for learning/prediction)
  config.py                    Paths, column groups, Experiment A settings, tuning grid
  prepare_data.py              Reproduce the Part I cleaned CSVs from the UCI archive
  prep.py                      Shared model-ready train/test files + stratified folds
  evaluation.py                Metrics, F1 threshold, cross-validated grid search
  results_io.py                Uniform output layout under part2/results/
  run_own_cart.py              Experiments A and B with the own CART
  run_sklearn.py               Experiments A and B with scikit-learn
  r/rpart_model.R              Experiments A and B with rpart
  r/metrics.R                  PR-AUC and threshold selection in R (matches scikit-learn)
  r/rpart_diagnostics.R        cp = 0 vs cp = -1 check; complete Experiment A rpart tree
  verify_equivalence.py        Own CART vs scikit-learn on identical ordered codes
  depth_check.py               Cross-validation at depths 17 and 20 (beyond the grid)
  check_overlap.py             Exact train/test feature-vector overlap audit
  audit_disagreement.py        Traces every own CART vs rpart probability difference
  report_figures.py            Comparison tables and report figures
  tests/                       pytest unit tests (28)
  results/                     Saved predictions, runs, tuning tables, trees and figures
```

## Setup

Python 3.12 and R 4.5.0 were used for the report.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Rscript -e 'install.packages(c("rpart", "jsonlite", "rpart.plot"))'   # see r-packages.txt
```

## Run order

Run everything from the project root (`python` below is `py -3.12` on Windows).

```bash
# 1. Data: download the UCI archive, rebuild the cleaned CSVs, export model-ready files
python -m part2.prepare_data --download
python -m part2.prep

# 2. Experiments (A and B) in the three implementations
python -m part2.run_own_cart
python -m part2.run_sklearn
Rscript part2/r/rpart_model.R

# 3. Checks used in the report
python -m part2.verify_equivalence
python -m part2.depth_check
python -m part2.check_overlap
Rscript part2/r/rpart_diagnostics.R
python -m part2.audit_disagreement     # needs rpart_diagnostics.R output

# 4. Tables and figures
python -m part2.report_figures

# Tests
python -m pytest part2/tests
Rscript part2/r/test_metrics.R
```

The full Experiment B grid takes several minutes per tool. `run_own_cart.py` accepts
`--experiment A|B|all` and `--grid small` for a quicker run.

## Data

Census-Income (KDD) [Dataset]. (2000). UCI Machine Learning Repository.
https://doi.org/10.24432/C5N30T. Raw and derived data files are not committed; step 1
recreates them (196,294 training and 98,879 test records after cleaning).
