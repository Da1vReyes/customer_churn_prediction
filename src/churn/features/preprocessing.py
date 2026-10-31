"""Feature selection, encoding and scaling to build the machine-learning-ready dataset.

Run as a script to regenerate the feature-engineered and ML-ready datasets:
    PYTHONPATH=src python -m churn.features.preprocessing
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

from churn.config import FEATURES_DATA_PATH, ID_COLUMN, ML_READY_DATA_PATH, TARGET
from churn.data.loader import load_processed_data
from churn.features.engineering import add_features

# Dropped after the chi-square test in the notebook: no statistically significant relation with churn.
LOW_SIGNAL_FEATURES = ["gender", "PhoneService"]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "num_services",
    "num_protection_services",
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier and the features with no relation to churn."""
    return df.drop(columns=[ID_COLUMN, *LOW_SIGNAL_FEATURES])


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode every categorical column as numbers.

    - Yes/No columns (and the target) become 1/0.
    - Multi-category columns are one-hot encoded (first level dropped to avoid redundancy).
    """
    df = df.copy()
    categorical = df.select_dtypes(include="object").columns

    binary = [col for col in categorical if set(df[col].unique()) == {"No", "Yes"}]
    for col in binary:
        df[col] = df[col].map({"No": 0, "Yes": 1})

    multi = [col for col in categorical if col not in binary]
    return pd.get_dummies(df, columns=multi, drop_first=True, dtype=int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target y."""
    return df.drop(columns=[TARGET]), df[TARGET]


def scale_numeric(train: pd.DataFrame, *others: pd.DataFrame):
    """Standardize the numeric features.

    The scaler is fitted on `train` only and then applied to the other sets, so no
    information from the test set leaks into training.
    """
    scaler = StandardScaler().fit(train[NUMERIC_FEATURES])
    scaled = []
    for part in (train, *others):
        part = part.copy()
        part[NUMERIC_FEATURES] = scaler.transform(part[NUMERIC_FEATURES])
        scaled.append(part)
    return (*scaled, scaler)


def main() -> None:
    featured = add_features(load_processed_data())
    featured.to_csv(FEATURES_DATA_PATH, index=False)

    ml_ready = encode_features(select_features(featured))
    ml_ready.to_csv(ML_READY_DATA_PATH, index=False)

    print(f"Feature-engineered dataset: {featured.shape} -> {FEATURES_DATA_PATH}")
    print(f"ML-ready dataset:           {ml_ready.shape} -> {ML_READY_DATA_PATH}")


if __name__ == "__main__":
    main()
