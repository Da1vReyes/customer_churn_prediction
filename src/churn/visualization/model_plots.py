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
