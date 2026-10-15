# rpart diagnostics for the report (Experiment A settings).
# (1) Why cp = -1: leaf counts with cp = 0 and cp = -1.
# (2) The complete cp = -1 tree (the Experiment A model), used by part2/audit_disagreement.py.
# Run from the project root: Rscript part2/r/rpart_diagnostics.R
suppressPackageStartupMessages({
  library(rpart)
  library(jsonlite)
})

RESULTS <- file.path("part2", "results")
NUMERIC_COLS <- c("age", "wage_per_hour", "capital_gains", "capital_losses",
                  "dividends_from_stocks", "num_persons_worked_for_employer",
                  "weeks_worked_in_year")

train <- read.csv(file.path("data", "model_ready", "train.csv"), stringsAsFactors = FALSE,
                  check.names = FALSE, na.strings = character(0))
feature_cols <- setdiff(names(train), c("income_class", "fold"))
for (col in setdiff(feature_cols, NUMERIC_COLS)) {
  train[[col]] <- factor(as.character(train[[col]]),
                         levels = sort(unique(as.character(train[[col]]))))
}
train$income_class <- factor(train$income_class, levels = c(0, 1))
model_formula <- reformulate(feature_cols, response = "income_class")

fit_exp_a <- function(cp) {
  rpart(model_formula, data = train, method = "class", parms = list(split = "gini"),
        control = rpart.control(maxdepth = 8, minbucket = 50, minsplit = 100, cp = cp,
                                xval = 0, maxcompete = 0, maxsurrogate = 0,
                                usesurrogate = 0))
}

dir.create(file.path(RESULTS, "trees"), recursive = TRUE, showWarnings = FALSE)
leaves <- list()
for (cp in c(0, -1)) {
  fit <- fit_exp_a(cp)
  leaves[[sprintf("cp_%d", cp)]] <- sum(fit$frame$var == "<leaf>")
  cat(sprintf("cp = %2d: %d leaves\n", cp, leaves[[sprintf("cp_%d", cp)]]))
}
write_json(leaves, file.path(RESULTS, "rpart_cp_check.json"), auto_unbox = TRUE, pretty = TRUE)
writeLines(capture.output(print(fit, digits = 10)),  # fit is the cp = -1 model
           file.path(RESULTS, "trees", "rpart_expA_full.txt"))
