"""Cross-validation: how consistent is each model across different slices of the training data?"""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn.config import RANDOM_STATE
from churn.modeling.evaluate import METRICS

SCORING = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1-Score": "f1",
    "ROC-AUC": "roc_auc",
}


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    """Stratified K-fold: every fold keeps the churn proportion."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_models(models: dict, X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """Score every model on each fold. Returns one row per (model, fold)."""
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=make_cv(n_splits), scoring=SCORING, n_jobs=-1)
        for fold in range(n_splits):
            rows.append({"Model": name, "Fold": fold + 1, **{m: scores[f"test_{m}"][fold] for m in METRICS}})
    return pd.DataFrame(rows)


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across folds, formatted as 'mean ± std'."""
    grouped = cv_results.groupby("Model", sort=False)[METRICS]
    mean, std = grouped.mean(), grouped.std()
    return mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
