"""Class-imbalance handling: build every model with each strategy.

Strategies:
- Baseline: no imbalance handling.
- Class weight: errors on churners weigh more during training (no data is changed).
- SMOTE: oversampling, creates synthetic churners until both classes have the same size.
- Undersampling: removes random non-churners until both classes have the same size.

Resampling is placed inside an imbalanced-learn Pipeline, so it is applied to the training
data only (also inside each cross-validation fold) and never to validation or test data.
"""
import os
import warnings

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from churn.config import RANDOM_STATE
from churn.modeling.evaluate import compare_models, evaluate_models
from churn.modeling.train import get_models, train_models

# imbalanced-learn 0.12 still calls scikit-learn methods that are deprecated in 1.6.
# The environment variable also silences the parallel worker processes used by cross-validation.
_DEPRECATED_SKLEARN_API = r".*(BaseEstimator\._|force_all_finite).*"
warnings.filterwarnings("ignore", message=_DEPRECATED_SKLEARN_API, category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

STRATEGIES = ["Baseline", "Class weight", "SMOTE", "Undersampling"]


class BalancedGradientBoosting(GradientBoostingClassifier):
    """Gradient Boosting that weights each sample by the inverse frequency of its class.

    sklearn's GradientBoostingClassifier has no `class_weight` parameter, so the
    weights are passed as `sample_weight` while fitting.
    """

    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def _with_class_weight(estimator, y_train):
    """Return the estimator configured to give more importance to the minority class."""
    if "class_weight" in estimator.get_params():  # Logistic Regression, Decision Tree, Random Forest
        return estimator.set_params(class_weight="balanced")
    if "scale_pos_weight" in estimator.get_params():  # XGBoost
        negatives, positives = (y_train == 0).sum(), (y_train == 1).sum()
        return estimator.set_params(scale_pos_weight=negatives / positives)
    if isinstance(estimator, GradientBoostingClassifier):
        return BalancedGradientBoosting(**estimator.get_params())
    raise ValueError(f"Class weighting is not supported for {type(estimator).__name__}")


def build_model(name: str, strategy: str, y_train) -> ImbPipeline:
    """Build one of the Week 3 models with the given imbalance strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {STRATEGIES}")

    base = get_models()[name]
    steps = list(base.steps) if isinstance(base, Pipeline) else [("model", base)]
    preprocessing, estimator = steps[:-1], steps[-1][1]

    if strategy == "Class weight":
        estimator = _with_class_weight(estimator, y_train)
    elif strategy == "SMOTE":
        preprocessing.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif strategy == "Undersampling":
        preprocessing.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))

    return ImbPipeline([*preprocessing, ("model", estimator)])


def build_models(strategy: str, y_train) -> dict:
    """All models (same names as `get_models`) built with one strategy."""
    return {name: build_model(name, strategy, y_train) for name in get_models()}


def run_strategies(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate every model with every strategy. Returns {strategy: evaluation results}."""
    experiments = {}
    for strategy in STRATEGIES:
        fitted = train_models(build_models(strategy, y_train), X_train, y_train)
        experiments[strategy] = evaluate_models(fitted, X_train, y_train, X_test, y_test)
    return experiments


def strategy_table(experiments: dict):
    """One row per (model, strategy) with the test metrics."""
    import pandas as pd

    tables = {strategy: compare_models(results) for strategy, results in experiments.items()}
    table = pd.concat(tables, names=["Strategy", "Model"]).swaplevel().sort_index(
        level="Model", sort_remaining=False
    )
    return table.reindex(
        pd.MultiIndex.from_product([list(get_models()), STRATEGIES], names=["Model", "Strategy"])
    )


def improvement_table(table):
    """Change of each metric versus the baseline (same model, no imbalance handling)."""
    metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    baseline = table.xs("Baseline", level="Strategy")[metrics]
    delta = table[metrics] - baseline.reindex(table.index.get_level_values("Model")).set_axis(table.index)
    return delta.drop("Baseline", level="Strategy")
