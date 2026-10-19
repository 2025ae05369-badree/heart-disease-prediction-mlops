import pandas as pd

from src.data_preprocessing import clean_data


def test_clean_data_converts_target_to_binary_and_fills_missing_values(
) -> None:
    df = pd.DataFrame(
        {
            "age": [52, 63, None],
            "sex": [1, 0, 1],
            "cp": [3, 2, None],
            "trestbps": [120, None, 140],
            "chol": [230, 240, 250],
            "fbs": [0, 1, None],
            "restecg": [1, 0, 1],
            "thalach": [150, 160, None],
            "exang": [0, 1, 0],
            "oldpeak": [1.2, None, 0.5],
            "slope": [1, 2, None],
            "ca": [0, 1, None],
            "thal": [3, 7, None],
            "target": [0, 2, 1],
        }
    )

    cleaned = clean_data(df)

    assert cleaned["target"].isin([0, 1]).all()
    assert cleaned.isna().sum().sum() == 0
    assert cleaned["target"].tolist() == [0, 1, 1]
