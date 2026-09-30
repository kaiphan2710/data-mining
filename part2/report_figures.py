"""Build the comparison table and figures used in the Part 2 report."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from part2 import config
from part2.evaluation import compute_metrics

TOOLS = ["own_cart", "sklearn", "rpart"]
LABELS = {"own_cart": "Own CART (Python/NumPy)", "sklearn": "scikit-learn (Python)",
          "rpart": "rpart (R)"}
# Categorical slots 1-3 of the validated reference palette (CVD-checked, fixed order).
COLORS = {"own_cart": "#2a78d6", "sklearn": "#eb6834", "rpart": "#1baf7a"}
# rpart is dashed: its curves overlay the own CART almost exactly.
STYLES = {"own_cart": "-", "sklearn": "-", "rpart": (0, (4, 2))}
REFERENCE = "#8a8984"
R = config.RESULTS_DIR
FIG = R / "figures"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
BAR = {"edgecolor": "white", "linewidth": 1.5}


def load_run(tool, experiment):
    info = json.loads((R / "runs" / f"{tool}_{experiment}.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(R / "predictions" / f"{tool}_{experiment}.csv")
    return info, pred


def summary_table() -> pd.DataFrame:
    rows = []
    for experiment in ["expA", "expB"]:
        for tool in TOOLS:
            info, pred = load_run(tool, experiment)
            m = compute_metrics(pred["y_true"], pred["y_prob"], info["threshold"])
            if tool == "rpart":  # cross-check R's own AP implementation
                gap = abs(info["test_metrics"]["pr_auc"] - m["pr_auc"])
                assert gap < 1e-6, f"R/Python PR-AUC mismatch {gap}"
            rows.append({"tool": tool, "experiment": experiment,
                         "params": json.dumps(info["params"]), **m,
                         "fit_seconds": info["fit_seconds"],
                         "n_leaves": info["n_leaves"], "depth": info["depth"]})
    table = pd.DataFrame(rows)
    table.to_csv(R / "metrics_summary.csv", index=False)
    return table


def agreement_table(experiment="expA") -> pd.DataFrame:
    """Share of test rows where two tools give the same probability / label."""
    probs = {t: load_run(t, experiment)[1]["y_prob"].to_numpy() for t in TOOLS}
    rows = []
    for i, a in enumerate(TOOLS):
        for b in TOOLS[i + 1:]:
            rows.append({"pair": f"{a} vs {b}",
                         "probability_agreement": np.mean(np.isclose(probs[a], probs[b], atol=1e-9)),
                         "label_agreement": np.mean((probs[a] >= 0.5) == (probs[b] >= 0.5))})
    table = pd.DataFrame(rows)
    table.to_csv(R / f"agreement_{experiment}.csv", index=False)
    return table


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)


def curves(experiment="expB"):
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for tool in TOOLS:
        _, pred = load_run(tool, experiment)
        fpr, tpr, _ = skm.roc_curve(pred["y_true"], pred["y_prob"])
        prec, rec, _ = skm.precision_recall_curve(pred["y_true"], pred["y_prob"])
        auc = skm.roc_auc_score(pred["y_true"], pred["y_prob"])
        ap = skm.average_precision_score(pred["y_true"], pred["y_prob"])
        ax_roc.plot(fpr, tpr, color=COLORS[tool], linewidth=2, linestyle=STYLES[tool],
                    label=f"{LABELS[tool]} (AUC={auc:.3f})")
        ax_pr.plot(rec, prec, color=COLORS[tool], linewidth=2, linestyle=STYLES[tool],
                   label=f"{LABELS[tool]} (AP={ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1,
                label="Random classifier")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC curves on the test set (Experiment B)")
    base = pred["y_true"].mean()
    ax_pr.axhline(base, color=REFERENCE, linestyle="--", linewidth=1,
                  label=f"Class prior ({base:.3f})")
    ax_pr.set(xlabel="Recall (>50K)", ylabel="Precision (>50K)",
              title="Precision-recall curves on the test set (Experiment B)")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    _save(fig_roc, "roc_expB.png")
    _save(fig_pr, "pr_expB.png")


def confusion(table, experiment="expB"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, tool in zip(axes, TOOLS):
        row = table[(table.tool == tool) & (table.experiment == experiment)].iloc[0]
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "#0b0b0b")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["<=50K", ">50K"],
               yticklabels=["<=50K", ">50K"], xlabel="Predicted", ylabel="Actual",
               title=f"{LABELS[tool]}\nthreshold={row.threshold:.3f}")
    _save(fig, "confusion_expB.png")


def metric_bars(table):
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    width = 0.26
    for ax, experiment in zip(axes, ["expA", "expB"]):
        x = np.arange(len(metrics))
        for k, tool in enumerate(TOOLS):
            row = table[(table.tool == tool) & (table.experiment == experiment)].iloc[0]
            ax.bar(x + (k - 1) * width, [row[m] for m in metrics], width,
                   color=COLORS[tool], label=LABELS[tool], **BAR)
        ax.set_xticks(x, ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"])
        ax.set_title("Experiment A (fixed, thr=0.5)" if experiment == "expA"
                     else "Experiment B (tuned, F1 threshold)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Score on test set (>50K class)")
    axes[0].legend(loc="upper left")
    _save(fig, "metrics_bars.png")


def importance_plot(top=15):
    frames = {t: pd.read_csv(R / "importance" / f"{t}_expB.csv").set_index("feature")["importance"]
              for t in TOOLS}
    features = frames["own_cart"].sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(features))
    for k, tool in enumerate(TOOLS):
        ax.barh(y + (1 - k) * 0.26, frames[tool].reindex(features).fillna(0), 0.26,
                color=COLORS[tool], label=LABELS[tool], **BAR)
    ax.set_yticks(y, features)
    ax.set(xlabel="Normalised importance", title=f"Top {top} features (Experiment B)")
    ax.legend(loc="lower right")
    _save(fig, "importance_top15_expB.png")


def cv_depth_curve(table):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for tool in TOOLS:
        cv = pd.read_csv(R / "tuning" / f"{tool}_cv.csv", keep_default_na=False)
        best = json.loads(table[(table.tool == tool) & (table.experiment == "expB")]
                          .iloc[0].params)
        mask = ((cv.min_samples_leaf == best["min_samples_leaf"])
                & (cv.class_weight == best["class_weight"]))
        sub = cv[mask].sort_values("max_depth")
        ax.errorbar(sub.max_depth, sub.cv_pr_auc_mean, yerr=sub.cv_pr_auc_std,
                    marker="o", markersize=5, linewidth=2, capsize=3, linestyle=STYLES[tool],
                    color=COLORS[tool], label=LABELS[tool])
    ax.set(xlabel="max_depth", ylabel="Mean 5-fold CV PR-AUC",
           title="Validation PR-AUC vs tree depth")
    ax.legend(loc="lower right")
    _save(fig, "cv_depth_curve.png")


def fit_time(table):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(2)
    for k, tool in enumerate(TOOLS):
        vals = [table[(table.tool == tool) & (table.experiment == e)].iloc[0].fit_seconds
                for e in ["expA", "expB"]]
        bars = ax.bar(x + (k - 1) * 0.26, vals, 0.26, color=COLORS[tool],
                      label=LABELS[tool], **BAR)
        ax.bar_label(bars, fmt="%.1f s", padding=2, fontsize=8, color="#52514e")
    ax.set_xticks(x, ["Experiment A", "Experiment B"])
    ax.set(ylabel="Fit time on full training set (s)", title="Training time per tool")
    ax.legend(loc="upper left")
    _save(fig, "fit_time.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    table = summary_table()
    print(agreement_table().round(4).to_string(index=False))
    curves()
    confusion(table)
    metric_bars(table)
    importance_plot()
    cv_depth_curve(table)
    fit_time(table)
    cols = ["tool", "experiment", "threshold", "precision", "recall", "f1",
            "pr_auc", "roc_auc", "fit_seconds", "n_leaves", "depth"]
    print(table[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
