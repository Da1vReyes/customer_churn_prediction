"""Plots to evaluate and compare classification models."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from churn.modeling.evaluate import METRICS


def plot_confusion_matrices(results: dict, y_true):
    """One confusion matrix per model (counts, with the row percentage below)."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = confusion_matrix(y_true, r["y_pred"])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)", ha="center", va="center", color=color)
        ax.set_xticks([0, 1], ["No churn", "Churn"])
        ax.set_yticks([0, 1], ["No churn", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    fig.tight_layout()
    return fig


def plot_roc_curves(results: dict, y_true):
    """ROC curves of all models on the same axes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_true, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {r['metrics']['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_model_comparison(comparison):
    """Grouped bar chart of the five metrics for every model."""
    ax = comparison[METRICS].plot(kind="bar", figsize=(10, 4.5), width=0.8, colormap="viridis")
    ax.set_ylim(0, 1)
    ax.set_title("Model performance on the test set")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    return ax.figure


def plot_strategy_comparison(table, metrics=("Precision", "Recall", "F1-Score", "ROC-AUC")):
    """Grouped bars: one panel per metric, one group per model, one bar per imbalance strategy."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), sharey=True)
    for ax, metric in zip(axes, metrics):
        table[metric].unstack("Strategy")[table.index.get_level_values("Strategy").unique()].plot(
            kind="bar", ax=ax, width=0.8, legend=False, colormap="viridis"
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score (test set)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return fig


def plot_cv_scores(cv_results, metric="F1-Score"):
    """Boxplot of one metric across the cross-validation folds, per model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    models = list(dict.fromkeys(cv_results["Model"]))
    ax.boxplot([cv_results.loc[cv_results["Model"] == m, metric] for m in models], labels=models)
    for i, m in enumerate(models, start=1):
        scores = cv_results.loc[cv_results["Model"] == m, metric]
        ax.scatter([i] * len(scores), scores, color="#DD8452", zorder=3, s=18)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} in each cross-validation fold (dots) and its spread")
    ax.tick_params(axis="x", labelrotation=15)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_feature_importance(permutation, coefficients, top: int = 10):
    """Left: permutation importance of the final model. Right: Logistic Regression coefficients (direction)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    best = permutation.head(top).iloc[::-1]
    axes[0].barh(best.index, best["importance"], xerr=best["std"], color="#4C72B0")
    axes[0].set_title("Random Forest: permutation importance\n(drop in F1 when the feature is shuffled)")

    shown = coefficients.iloc[: top // 2].tolist() + coefficients.iloc[-(top // 2):].tolist()
    names = coefficients.index[: top // 2].tolist() + coefficients.index[-(top // 2):].tolist()
    colors = ["#4C72B0" if v < 0 else "#DD8452" for v in shown]
    axes[1].barh(names, shown, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Logistic Regression coefficients\n(orange = more churn, blue = less churn)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_risk_deciles(table):
    """Churn rate in each risk decile (bars) and the cumulative share of churners captured (line)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table.index.astype(str), table["churn_rate_%"], color="#DD8452")
    ax.set_xlabel("Risk decile (1 = the 10% of customers with the highest predicted risk)")
    ax.set_ylabel("Actual churn rate (%)")
    ax2 = ax.twinx()
    ax2.plot(table.index.astype(str), table["cumulative_churners_captured_%"], color="#4C72B0", marker="o")
    ax2.set_ylabel("Cumulative churners captured (%)")
    ax2.set_ylim(0, 105)
    ax.set_title("Customers ranked by predicted risk")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax.grid(False)
    ax2.grid(False)
    return fig
