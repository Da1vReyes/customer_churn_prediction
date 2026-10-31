"""Input layer: load datasets from disk and inspect their structure."""
import pandas as pd

from churn.config import ML_READY_DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset exactly as downloaded."""
    return pd.read_csv(path)


def load_processed_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset produced by `churn.data.cleaning`."""
    return pd.read_csv(path)


def load_ml_ready_data(path=ML_READY_DATA_PATH) -> pd.DataFrame:
    """Load the numeric dataset produced by `churn.features.preprocessing`."""
    return pd.read_csv(path)


def describe_structure(df: pd.DataFrame) -> None:
    """Print the basic structure of a dataframe."""
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn names:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")


def target_distribution(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Return counts and percentages of each class in the target variable."""
    counts = df[target].value_counts()
    percentages = df[target].value_counts(normalize=True).mul(100).round(2)
    return pd.DataFrame({"count": counts, "percentage": percentages})
