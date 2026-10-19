"""Run the full MLOps pipeline end-to-end, step by step.

Usage:
    python main.py
"""

from data import download_data
from src import data_preprocessing
from src import eda
from src import train

# Add new (name, callable) steps here as later tasks (packaging, ...) are built.
STEPS = [
    ("Data Acquisition", download_data.download),
    ("Data Preprocessing", data_preprocessing.run),
    ("Exploratory Data Analysis", eda.run),
    ("Model Training & Evaluation", train.run),
]


def main() -> None:
    for name, step in STEPS:
        print(f"\n{'=' * 60}\n[STEP] {name}\n{'=' * 60}")
        step()
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
