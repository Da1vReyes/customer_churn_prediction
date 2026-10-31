"""Hyperparameter tuning with RandomizedSearchCV / GridSearchCV.

Models are tuned with class weighting (see `churn.modeling.imbalance`) and the F1-score as
the objective, which balances catching churners (recall) and not bothering loyal customers
(precision). Every candidate is scored with stratified cross-validation on the training set only.
"""
import time

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from churn.config import RANDOM_STATE
from churn.modeling.imbalance import build_model
from churn.modeling.train import XGBOOST_AVAILABLE
from churn.modeling.validation import make_cv

BOOSTING_NAME = "XGBoost" if XGBOOST_AVAILABLE else "Gradient Boosting"

# Parameter names use the "model__" prefix because the estimator is the last step of a pipeline.
SEARCHES = {
    "Logistic Regression": ("grid", {"model__C": [0.01, 0.1, 1, 10, 100]}),
    "Decision Tree": (
        "grid",
        {
            "model__max_depth": [3, 4, 5, 6, 8, 10],
            "model__min_samples_leaf": [1, 5, 10, 20, 50],
            "model__criterion": ["gini", "entropy"],
        },
    ),
    "Random Forest": (
        "random",
        {
            "model__n_estimators": randint(100, 400),
            "model__max_depth": randint(3, 15),
            "model__min_samples_leaf": randint(1, 30),
            "model__max_features": ["sqrt", "log2", 0.5],
        },
    ),
    BOOSTING_NAME: (
        "random",
        {
            "model__n_estimators": randint(50, 300),
            "model__learning_rate": uniform(0.01, 0.19),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
        },
    ),
}


def tune_model(name: str, X_train, y_train, scoring: str = "f1", n_iter: int = 25, n_splits: int = 5):
    """Search the best hyperparameters of one class-weighted model. Returns the fitted search object."""
    kind, space = SEARCHES[name]
    model = build_model(name, "Class weight", y_train)
    common = dict(scoring=scoring, cv=make_cv(n_splits), n_jobs=-1)
    if kind == "grid":
        search = GridSearchCV(model, space, **common)
    else:
        search = RandomizedSearchCV(model, space, n_iter=n_iter, random_state=RANDOM_STATE, **common)
    return search.fit(X_train, y_train)


def tune_all(X_train, y_train, **kwargs) -> dict:
    """Tune every model in SEARCHES. Returns {name: fitted search}."""
    searches = {}
    for name in SEARCHES:
        start = time.time()
        searches[name] = tune_model(name, X_train, y_train, **kwargs)
        print(f"{name}: best CV score {searches[name].best_score_:.4f} ({time.time() - start:.0f}s)")
    return searches


def tuning_summary(searches: dict) -> pd.DataFrame:
    """Best CV score and best hyperparameters of every search."""
    rows = {
        name: {
            "Best CV F1": round(s.best_score_, 4),
            "Best parameters": {k.replace("model__", ""): (round(v, 3) if isinstance(v, float) else v)
                                for k, v in s.best_params_.items()},
        }
        for name, s in searches.items()
    }
    return pd.DataFrame(rows).T
