"""Cleaning and preprocessing for the UCI Heart Disease dataset.

Usage:
    python src/data_preprocessing.py
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "heart_disease_clean.csv"

NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal",
]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope"]


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw UCI CSV, treating '?' as missing values."""
    return pd.read_csv(path, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values, cast types, and binarize the target column."""
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)

    # Original target is disease severity (0-4); the assignment requires
    # a binary label.
    df["target"] = (df["target"] > 0).astype(int)

    return df


def run() -> Path:
    df = load_raw_data()
    cleaned = clean_data(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"Cleaned dataset saved to: {PROCESSED_PATH}")
    return PROCESSED_PATH


if __name__ == "__main__":
    run()
