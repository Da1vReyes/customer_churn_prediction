"""Data cleaning: fix types, duplicates, blanks and inconsistent categories.

Run as a script to regenerate the processed dataset:
    PYTHONPATH=src python -m churn.data.cleaning
"""
import pandas as pd

from churn.config import ID_COLUMN, PROCESSED_DATA_PATH
from churn.data.loader import load_raw_data

# Add-on services that contain "No internet service" when the customer has no internet.
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column data quality summary: dtype, missing values, blank strings, unique values."""
    blanks = df.select_dtypes(include="object").apply(lambda s: s.str.strip().eq("").sum())
    report = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "blank_strings": blanks,
            "unique_values": df.nunique(),
        }
    )
    report["blank_strings"] = report["blank_strings"].fillna(0).astype(int)
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw churn dataframe."""
    df = df.copy()

    # 1. Remove leading/trailing spaces in every text column.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # 2. Remove duplicate rows and duplicate customers.
    df = df.drop_duplicates().drop_duplicates(subset=ID_COLUMN)

    # 3. TotalCharges is stored as text because 11 rows contain a blank string.
    #    Those customers have tenure = 0 (no bill generated yet), so 0 is the correct value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"].eq(0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    assert df["TotalCharges"].isna().sum() == 0, "Unexpected missing TotalCharges after cleaning"

    # 4. SeniorCitizen is encoded as 0/1 while every other binary column uses Yes/No.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 5. Consolidate redundant categories. The information is kept in
    #    InternetService / PhoneService, so "No internet service" / "No phone service" == "No".
    df[INTERNET_ADDON_COLUMNS] = df[INTERNET_ADDON_COLUMNS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return df.reset_index(drop=True)


def main() -> None:
    raw = load_raw_data()
    clean = clean_data(raw)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Raw shape:   {raw.shape}")
    print(f"Clean shape: {clean.shape}")
    print(f"Saved to:    {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
