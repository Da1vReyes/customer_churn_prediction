"""Project-wide configuration: paths and constants."""
from pathlib import Path

# Project root (this file lives in <root>/src/churn/config.py)
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw_data" / "telco_churn.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed_data" / "telco_churn_clean.csv"
VISUALS_DIR = BASE_DIR / "visuals"

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMN = "customerID"
