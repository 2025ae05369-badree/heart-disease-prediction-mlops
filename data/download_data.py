"""Download the UCI Heart Disease (Cleveland) dataset to data/raw/heart_disease.csv.

Usage:
    python data/download_data.py
"""
from pathlib import Path

import requests

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

RAW_DIR = Path(__file__).parent / "raw"
RAW_PATH = RAW_DIR / "heart_disease.csv"


def download(url: str = UCI_URL, out_path: Path = RAW_PATH) -> Path:
    """Fetch the raw UCI dataset and save it as a headered CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download dataset from {url}: {exc}\n"
            "If you are offline, manually download the file from the UCI "
            "Heart Disease repository and save it as "
            f"'{out_path}' with columns: {','.join(COLUMNS)}"
        ) from exc

    out_path.write_text(",".join(COLUMNS) + "\n" + text)
    print(f"Dataset downloaded to: {out_path}")
    return out_path


if __name__ == "__main__":
    download()
