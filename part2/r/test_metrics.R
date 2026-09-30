# Run from project root: Rscript part2/r/test_metrics.R
source(file.path("part2", "r", "metrics.R"))

# Same toy case as part2/tests/test_evaluation.py::test_compute_metrics_toy
stopifnot(abs(average_precision(c(0, 0, 1, 1), c(0.1, 0.6, 0.4, 0.9)) - 5 / 6) < 1e-12)
stopifnot(abs(best_f1_threshold(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9)) - 0.8) < 1e-12)
# Ties share one threshold (sklearn semantics)
stopifnot(abs(average_precision(c(1, 0, 1), c(0.5, 0.5, 0.2)) - (0.5 * 0.5 + 0.5 * (2 / 3))) < 1e-12)
# Tied best F1 (thresholds 0.9 and 0.6): pick the lowest, like Python's best_f1_threshold
stopifnot(abs(best_f1_threshold(c(1, 0, 0, 1, 0), c(0.9, 0.8, 0.7, 0.6, 0.5)) - 0.6) < 1e-12)
cat("metrics.R OK\n")
