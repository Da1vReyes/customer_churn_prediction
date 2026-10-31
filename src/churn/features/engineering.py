"""Feature engineering: create new, more informative features from the cleaned data.

Every feature is documented in `docs/feature_engineering.md`.
"""
import pandas as pd

from churn.data.cleaning import INTERNET_ADDON_COLUMNS

PROTECTION_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_COLUMNS = ["StreamingTV", "StreamingMovies"]

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

SPENDING_BINS = [0, 35, 70, float("inf")]
SPENDING_LABELS = ["Low", "Medium", "High"]

ENGINEERED_FEATURES = [
    "tenure_group",
    "num_services",
    "num_protection_services",
    "has_streaming",
    "spending_category",
    "auto_payment",
    "avg_monthly_spend",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the cleaned dataframe with the engineered features added."""
    df = df.copy()

    # Customer lifecycle stage: churn drops sharply as tenure grows.
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

    # Breadth of the subscription: phone, internet and every add-on counted once.
    service_flags = pd.concat(
        [
            df["PhoneService"].eq("Yes"),
            df["MultipleLines"].eq("Yes"),
            df["InternetService"].ne("No"),
            *[df[col].eq("Yes") for col in INTERNET_ADDON_COLUMNS],
        ],
        axis=1,
    )
    df["num_services"] = service_flags.sum(axis=1)

    # Add-ons that make the customer "stickier" (security and support).
    df["num_protection_services"] = df[PROTECTION_COLUMNS].eq("Yes").sum(axis=1)
    df["has_streaming"] = df[STREAMING_COLUMNS].eq("Yes").any(axis=1).map({True: "Yes", False: "No"})

    # Spending band based on the monthly bill.
    df["spending_category"] = pd.cut(
        df["MonthlyCharges"], bins=SPENDING_BINS, labels=SPENDING_LABELS
    ).astype(str)

    # Automatic payments (bank transfer / credit card) require no monthly action by the customer.
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").map({True: "Yes", False: "No"})

    # Average amount paid per month over the customer's life (new customers: current monthly charge).
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].where(df["tenure"] > 0)).fillna(
        df["MonthlyCharges"]
    ).round(2)

    return df
