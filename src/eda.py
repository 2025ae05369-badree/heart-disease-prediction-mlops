"""Exploratory Data Analysis for the UCI Heart Disease dataset.

Prints summary statistics and saves professional visualizations
(histograms, correlation heatmap, class balance) as PNGs.

Usage:
    python src/eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from src.data_preprocessing import clean_data, load_raw_data
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    from data_preprocessing import clean_data, load_raw_data

FIGURES_DIR = Path(__file__).parent.parent / "outputs" / "figures"

sns.set_theme(style="whitegrid")


def print_summary(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nDescribe:\n", df.describe().T)
    print("\nTarget value counts:\n", df["target"].value_counts())


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.drop("target")
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
    for ax in axes.flatten()[len(numeric_cols):]:
        ax.axis("off")
    fig.suptitle("Feature Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "histograms.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(
        x="target",
        data=df,
        hue="target",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Class Balance (0 = No Disease, 1 = Disease)")
    ax.set_xlabel("Target")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def run() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_raw_data())

    print_summary(df)
    plot_histograms(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)
    plot_class_balance(df, FIGURES_DIR)

    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    run()
