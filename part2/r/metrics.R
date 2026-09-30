# Precision-recall helpers matching scikit-learn's average_precision_score and
# precision_recall_curve (one point per distinct score; predict 1 when p >= threshold).
pr_curve_points <- function(y, p) {
  o <- order(p, decreasing = TRUE)
  y <- y[o]
  p <- p[o]
  tp <- cumsum(y)
  fp <- cumsum(1 - y)
  last <- c(p[-1] != p[-length(p)], TRUE)   # last row of each distinct score
  list(threshold = p[last],
       precision = tp[last] / (tp[last] + fp[last]),
       recall = tp[last] / sum(y))
}

average_precision <- function(y, p) {
  pts <- pr_curve_points(y, p)
  sum(diff(c(0, pts$recall)) * pts$precision)
}

best_f1_threshold <- function(y, p) {
  pts <- pr_curve_points(y, p)
  denom <- pts$precision + pts$recall
  f1 <- ifelse(denom > 0, 2 * pts$precision * pts$recall / denom, 0)
  # Thresholds are in decreasing order: on a tie take the last (lowest) one, as Python does.
  pts$threshold[max(which(f1 == max(f1)))]
}
