# Tool 2 (R platform): CART via rpart (Therneau & Atkinson).
# Run from the project root:  Rscript part2/r/rpart_model.R [--experiment all|A|B]
suppressPackageStartupMessages({
  library(rpart)
  library(jsonlite)
})
source(file.path("part2", "r", "metrics.R"))

args <- commandArgs(trailingOnly = TRUE)
experiment_arg <- if (length(args) >= 2 && args[1] == "--experiment") args[2] else "all"

TOOL <- "rpart"
SEED <- 42
RESULTS <- file.path("part2", "results")
NUMERIC_COLS <- c("age", "wage_per_hour", "capital_gains", "capital_losses",
                  "dividends_from_stocks", "num_persons_worked_for_employer",
                  "weeks_worked_in_year")
EXP_A <- list(max_depth = 8, min_samples_leaf = 50, class_weight = "none")
GRID <- expand.grid(max_depth = c(5, 8, 11, 14),
                    min_samples_leaf = c(1, 50, 200),
                    class_weight = c("none", "balanced"),
                    stringsAsFactors = FALSE)

for (d in c("predictions", "runs", "tuning", "importance", "trees", "figures")) {
  dir.create(file.path(RESULTS, d), recursive = TRUE, showWarnings = FALSE)
}

read_split <- function(path) {
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE, na.strings = character(0))
}
train <- read_split(file.path("data", "model_ready", "train.csv"))
test <- read_split(file.path("data", "model_ready", "test.csv"))

feature_cols <- setdiff(names(train), c("income_class", "fold"))
stopifnot(identical(feature_cols, setdiff(names(test), "income_class")))
stopifnot(length(feature_cols) == 40)
cat_cols <- setdiff(feature_cols, NUMERIC_COLS)

for (col in cat_cols) {
  train_levels <- sort(unique(as.character(train[[col]])))
  train[[col]] <- factor(as.character(train[[col]]), levels = train_levels)
  test[[col]] <- factor(as.character(test[[col]]), levels = train_levels)  # unseen -> NA
}
y_train <- train$income_class
y_test <- test$income_class
train$income_class <- factor(train$income_class, levels = c(0, 1))
folds <- train$fold
model_formula <- reformulate(feature_cols, response = "income_class")

# cp = -1: rpart measures cp on misclassification risk, so cp = 0 still snips splits that
# lower Gini but not the error count; -1 grows the same tree as sklearn / the own CART.
fit_rpart <- function(data, p) {
  parms <- list(split = "gini")
  # Equal priors == 'balanced' class weights for the Gini criterion.
  if (p$class_weight == "balanced") parms$prior <- c(0.5, 0.5)
  rpart(model_formula, data = data, method = "class", parms = parms,
        control = rpart.control(maxdepth = p$max_depth,
                                minbucket = p$min_samples_leaf,
                                minsplit = max(2, 2 * p$min_samples_leaf),
                                cp = -1, xval = 0, maxcompete = 0,
                                maxsurrogate = 0, usesurrogate = 0))
}

# Leaf fractions that differ only by floating-point noise (e.g. 1e-16) are merged so the
# values written to CSV are exactly the values the metrics were computed on.
prob_positive <- function(model, data) {
  round(predict(model, newdata = data, type = "prob")[, "1"], 12)
}

elapsed <- function() proc.time()[["elapsed"]]

tree_stats <- function(model) {
  node_ids <- as.integer(rownames(model$frame))
  list(n_nodes = nrow(model$frame),
       n_leaves = sum(model$frame$var == "<leaf>"),
       depth = max(floor(log2(node_ids))))
}

cross_validate <- function() {
  rows <- list()
  best <- NULL
  for (g in seq_len(nrow(GRID))) {
    p <- as.list(GRID[g, ])
    oof <- rep(NA_real_, nrow(train))
    seconds <- 0
    fold_ap <- c()
    for (k in sort(unique(folds))) {
      tr <- folds != k
      va <- folds == k
      start <- elapsed()
      model <- fit_rpart(train[tr, ], p)
      seconds <- seconds + elapsed() - start
      oof[va] <- prob_positive(model, train[va, ])
      fold_ap <- c(fold_ap, average_precision(y_train[va], oof[va]))
    }
    row <- data.frame(max_depth = p$max_depth, min_samples_leaf = p$min_samples_leaf,
                      class_weight = p$class_weight,
                      cv_pr_auc_mean = mean(fold_ap),
                      cv_pr_auc_std = sqrt(mean((fold_ap - mean(fold_ap))^2)),
                      fit_seconds_total = seconds)
    rows[[g]] <- row
    print(row)
    if (is.null(best) || row$cv_pr_auc_mean > best$row$cv_pr_auc_mean) {
      best <- list(row = row, params = p, oof = oof)
    }
  }
  write.csv(do.call(rbind, rows), file.path(RESULTS, "tuning", "rpart_cv.csv"),
            row.names = FALSE)
  best
}

fit_and_report <- function(experiment, p, threshold) {
  start <- elapsed()
  model <- fit_rpart(train, p)
  fit_seconds <- elapsed() - start
  start <- elapsed()
  prob <- prob_positive(model, test)
  predict_seconds <- elapsed() - start

  write.csv(data.frame(y_true = y_test, y_prob = prob),
            file.path(RESULTS, "predictions", sprintf("%s_%s.csv", TOOL, experiment)),
            row.names = FALSE)
  importance <- model$variable.importance / sum(model$variable.importance)
  write.csv(data.frame(feature = names(importance), importance = as.numeric(importance)),
            file.path(RESULTS, "importance", sprintf("%s_%s.csv", TOOL, experiment)),
            row.names = FALSE)
  writeLines(head(capture.output(print(model)), 80),
             file.path(RESULTS, "trees", sprintf("%s_%s.txt", TOOL, experiment)))

  stats <- tree_stats(model)
  info <- list(tool = TOOL, experiment = experiment, params = p, threshold = threshold,
               fit_seconds = fit_seconds, predict_seconds = predict_seconds,
               n_nodes = stats$n_nodes, n_leaves = stats$n_leaves, depth = stats$depth,
               test_metrics = list(pr_auc = average_precision(y_test, prob)))
  write_json(info, file.path(RESULTS, "runs", sprintf("%s_%s.json", TOOL, experiment)),
             auto_unbox = TRUE, digits = NA, pretty = TRUE)
  cat(experiment, "test PR-AUC:", info$test_metrics$pr_auc, "\n")
  invisible(model)
}

plot_shallow_tree <- function(p) {
  if (!requireNamespace("rpart.plot", quietly = TRUE)) return(invisible(NULL))
  shallow <- fit_rpart(train, modifyList(p, list(max_depth = 3)))
  png(file.path(RESULTS, "figures", "rpart_tree_depth3.png"),
      width = 2400, height = 1400, res = 200)
  wrap_labels <- function(x, labs, digits, varlen, faclen) {
    sapply(labs, function(l) paste(strwrap(gsub(",", ", ", l), width = 40), collapse = "
"))
  }
  rpart.plot::rpart.plot(shallow, type = 4, extra = 106, under = TRUE, faclen = 12,
                         varlen = 0, roundint = FALSE, box.palette = "Blues",
                         split.fun = wrap_labels, fallen.leaves = TRUE)
  dev.off()
}

set.seed(SEED)
if (experiment_arg %in% c("all", "A")) {
  fit_and_report("expA", EXP_A, 0.5)
}
if (experiment_arg %in% c("all", "B")) {
  best <- cross_validate()
  threshold <- best_f1_threshold(y_train, best$oof)
  cat("best params:", unlist(best$params), "threshold:", threshold, "\n")
  fit_and_report("expB", best$params, threshold)
  plot_shallow_tree(best$params)
}
if (experiment_arg == "plot") {  # redraw the figure from the saved Experiment B params
  plot_shallow_tree(fromJSON(file.path(RESULTS, "runs", "rpart_expB.json"))$params)
}
