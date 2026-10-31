"""Model interpretation: feature importance, error analysis and customer risk ranking."""
import pandas as pd
from sklearn.inspection import permutation_importance

from churn.config import RANDOM_STATE


def permutation_importance_table(model, X_test, y_test, scoring: str = "f1", n_repeats: int = 10) -> pd.DataFrame:
    """How much the score drops when one feature is shuffled (larger drop = more important)."""
    result = permutation_importance(
        model, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1
    )
    return (
        pd.DataFrame({"importance": result.importances_mean, "std": result.importances_std}, index=X_test.columns)
        .sort_values("importance", ascending=False)
    )


def impurity_importance_table(forest, columns) -> pd.Series:
    """Built-in importance of a tree ensemble (share of the splits' impurity reduction)."""
    return pd.Series(forest.feature_importances_, index=columns).sort_values(ascending=False)


def logistic_coefficients(pipeline, columns, numeric_features) -> pd.Series:
    """Coefficients of a Logistic Regression pipeline in the order of the original columns.

    Numeric features are standardized, so coefficients are comparable: positive = more churn.
    """
    ordered = [*numeric_features, *[c for c in columns if c not in numeric_features]]
    return pd.Series(pipeline.named_steps["model"].coef_[0], index=ordered).sort_values()


def error_groups(y_true, y_pred) -> pd.Series:
    """Label every customer as TP, TN, FP or FN (churn is the positive class)."""
    labels = pd.Series("TN", index=y_true.index)
    labels[(y_true == 1) & (y_pred == 1)] = "TP"
    labels[(y_true == 0) & (y_pred == 1)] = "FP"
    labels[(y_true == 1) & (y_pred == 0)] = "FN"
    return labels


def risk_deciles(y_true, y_proba) -> pd.DataFrame:
    """Customers sorted by predicted risk and split in 10 groups: churn rate and share of churners captured."""
    ranked = pd.DataFrame({"churn": y_true.values, "risk": y_proba}).sort_values("risk", ascending=False)
    ranked["decile"] = pd.qcut(range(len(ranked)), 10, labels=range(1, 11))
    table = ranked.groupby("decile", observed=True)["churn"].agg(customers="size", churners="sum")
    table["churn_rate_%"] = (table["churners"] / table["customers"] * 100).round(1)
    table["cumulative_churners_captured_%"] = (table["churners"].cumsum() / table["churners"].sum() * 100).round(1)
    return table
