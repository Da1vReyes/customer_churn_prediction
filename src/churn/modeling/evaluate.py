"""Model evaluation: metrics, comparison table and confusion matrices.

The positive class is churn (1), so Precision and Recall describe how well churners are found.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """The five metrics used in the project (churn is the positive class)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def evaluate_models(fitted: dict, X_train, y_train, X_test, y_test) -> dict:
    """Evaluate fitted models. Returns {name: {predictions, probabilities, metrics, train ROC-AUC}}."""
    results = {}
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics": compute_metrics(y_test, y_pred, y_proba),
            "train_roc_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        }
    return results


def compare_models(results: dict) -> pd.DataFrame:
    """Model comparison table (test metrics), plus the train ROC-AUC to detect overfitting."""
    table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T[METRICS]
    table["Train ROC-AUC"] = pd.Series({name: r["train_roc_auc"] for name, r in results.items()})
    return table


def confusion_matrix_table(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix with readable labels (rows = actual, columns = predicted)."""
    labels = ["No churn", "Churn"]
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels],
    )
