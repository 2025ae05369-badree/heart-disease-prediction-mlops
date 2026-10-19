"""Feature engineering pipeline for the Heart Disease dataset.

Usage:
    from features import (
        build_preprocessor,
        CONTINUOUS_FEATURES,
        CATEGORICAL_FEATURES,
    )
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Truly continuous measurements -> standardized.
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Discrete/coded clinical categories -> one-hot encoded.
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def build_preprocessor() -> ColumnTransformer:
    """Return a preprocessor for continuous and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
