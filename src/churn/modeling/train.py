"""Model definitions and training.

Run as a script to train every model and print the comparison table:
    PYTHONPATH=src python -m churn.modeling.train
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from churn.config import RANDOM_STATE
from churn.data.loader import load_ml_ready_data
from churn.features.preprocessing import NUMERIC_FEATURES
from churn.modeling.evaluate import compare_models, evaluate_models
from churn.modeling.split import split_data

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # xgboost missing, or its native library (libomp on macOS) is not installed
    XGBOOST_AVAILABLE = False


def get_models() -> dict:
    """Untuned models for the first comparison (Week 3). Tuning and imbalance handling come in Week 4.

    Logistic Regression is the baseline. It needs scaled numeric features, so the scaler
    lives inside its pipeline and is fitted on the training data only. Tree-based models
    do not need scaling.
    """
    baseline = Pipeline(
        [
            ("scale", ColumnTransformer(
                [("numeric", StandardScaler(), NUMERIC_FEATURES)], remainder="passthrough"
            )),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    if XGBOOST_AVAILABLE:
        boosting = ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE))
    else:
        print("XGBoost is not available (is libomp installed?). Using sklearn's Gradient Boosting instead.")
        boosting = ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE))

    return {
        "Logistic Regression": baseline,
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        boosting[0]: boosting[1],
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """Fit every model on the training set and return the fitted models."""
    return {name: model.fit(X_train, y_train) for name, model in models.items()}


def main() -> None:
    X_train, X_test, y_train, y_test = split_data(load_ml_ready_data())
    fitted = train_models(get_models(), X_train, y_train)
    results = evaluate_models(fitted, X_train, y_train, X_test, y_test)
    print(compare_models(results).round(4).to_string())


if __name__ == "__main__":
    main()
