import joblib
import pandas as pd
from pathlib import Path

from src import train

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"


def test_training_pipeline_returns_expected_model_results() -> None:
    results = train.run()

    assert set(results) == {"logistic_regression", "random_forest"}
    assert "roc_auc" in results["logistic_regression"]["test_metrics"]
    assert MODEL_PATH.exists()

    packaged_model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        ]
    )
    prediction = packaged_model.predict(sample)
    probability = packaged_model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in {0, 1}
    assert probability.shape == (1, 2)
    assert 0.0 <= probability[0, 1] <= 1.0
