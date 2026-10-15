"""Shared configuration for Assessment Part 2 (Decision Tree / CART)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_TRAIN = ROOT / "data" / "cleaned" / "census_income_train_cleaned.csv"
CLEAN_TEST = ROOT / "data" / "cleaned" / "census_income_test_cleaned.csv"
MODEL_READY_DIR = ROOT / "data" / "model_ready"
RESULTS_DIR = ROOT / "part2" / "results"

TARGET = "income_class"
FOLD = "fold"

# Survey expansion weight, excluded from predictors. Record-level metrics below
# do not use survey weights; balanced class weights are a separate model option.
DROP_COLS = ["instance_weight"]

NUMERIC_COLS = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]

# Stored as integers but semantically nominal codes.
CODED_CATEGORICAL_COLS = [
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "own_business_or_self_employed",
    "veterans_benefits",
    "year",
]

SEED = 42
N_FOLDS = 5

# Experiment A: shared settings; one-hot and native subset encodings still differ.
EXP_A_PARAMS = {"max_depth": 8, "min_samples_leaf": 50, "class_weight": None}

# Experiment B: identical grid tuned with the shared stratified folds in every tool.
PARAM_GRID = {
    "max_depth": [5, 8, 11, 14],
    "min_samples_leaf": [1, 50, 200],
    "class_weight": [None, "balanced"],
}
