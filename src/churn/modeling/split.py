"""Stratified train/test split of the machine-learning-ready dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import RANDOM_STATE, TARGET, TEST_DATA_PATH, TEST_SIZE, TRAIN_DATA_PATH
from churn.data.loader import load_ml_ready_data
from churn.features.preprocessing import split_features_target


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Split into X_train, X_test, y_train, y_test keeping the churn proportion in both sets."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def save_split(X_train, X_test, y_train, y_test) -> None:
    """Save the training and testing datasets (features + target) to processed_data/."""
    X_train.assign(**{TARGET: y_train}).to_csv(TRAIN_DATA_PATH, index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(TEST_DATA_PATH, index=False)


def main() -> None:
    X_train, X_test, y_train, y_test = split_data(load_ml_ready_data())
    save_split(X_train, X_test, y_train, y_test)
    print(f"Train: {X_train.shape} | churn rate {y_train.mean():.2%} -> {TRAIN_DATA_PATH}")
    print(f"Test:  {X_test.shape} | churn rate {y_test.mean():.2%} -> {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
