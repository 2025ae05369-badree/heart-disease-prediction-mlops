"""Model training: builds pipelines, tunes hyperparameters, and evaluates
Logistic Regression and Random Forest classifiers on the processed Heart
Disease dataset.

Usage:
    python src/train.py
"""
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

try:
    from src.features import build_preprocessor
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    from features import build_preprocessor

ROOT_DIR = Path(__file__).parent.parent
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "heart_disease_clean.csv"
MLRUNS_DIR = ROOT_DIR / "mlruns"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
ARTIFACTS_DIR = ROOT_DIR / "outputs" / "figures" / "mlflow_runs"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"
MLFLOW_EXPERIMENT = "heart-disease-classification"
LOCAL_MLFLOW_EXPERIMENT = f"{MLFLOW_EXPERIMENT}-local"
TARGET_COLUMN = "target"
RANDOM_STATE = 42

SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# (model name, estimator, hyperparameter grid for GridSearchCV)
MODEL_CONFIGS = [
    (
        "logistic_regression",
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__solver": ["lbfgs"],
        },
    ),
    (
        "random_forest",
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    )
]


def configure_mlflow() -> None:
    """Configure tracking and keep artifacts inside this project."""
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        _set_project_local_experiment(client, MLFLOW_EXPERIMENT)
        return

    if _is_project_local_artifact_location(experiment.artifact_location):
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        return

    _set_project_local_experiment(client, LOCAL_MLFLOW_EXPERIMENT)


def _is_project_local_artifact_location(artifact_location: str) -> bool:
    parsed_location = urlparse(artifact_location)
    if parsed_location.scheme != "file":
        return False
    artifact_path = Path(unquote(parsed_location.path)).resolve()
    return artifact_path.is_relative_to(MLRUNS_DIR.resolve())


def _set_project_local_experiment(
    client: MlflowClient,
    experiment_name: str,
) -> None:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (
            MLRUNS_DIR / experiment_name
        ).resolve().as_uri()
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        return

    if not _is_project_local_artifact_location(experiment.artifact_location):
        raise RuntimeError(
            f"MLflow experiment artifacts are outside this project: "
            f"{experiment.artifact_location}. Expected a location under "
            f"{MLRUNS_DIR.resolve()}."
        )

    mlflow.set_experiment(experiment_id=experiment.experiment_id)


def load_dataset(
    path: Path = PROCESSED_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def tune_model(
    pipeline: Pipeline,
    param_grid: dict,
    X_train,
    y_train,
    cv: StratifiedKFold,
) -> GridSearchCV:
    """Grid-search pipeline hyperparameters selecting by ROC-AUC."""
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def cross_validated_metrics(
    estimator,
    X_train,
    y_train,
    cv: StratifiedKFold,
) -> dict:
    """Mean +/- std of metrics across cross-validation folds."""
    results = cross_validate(
        estimator,
        X_train,
        y_train,
        scoring=SCORING,
        cv=cv,
    )
    return {
        metric: (
            results[f"test_{metric}"].mean(),
            results[f"test_{metric}"].std(),
        )
        for metric in SCORING
    }


def test_set_metrics(estimator, X_test, y_test) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def plot_confusion_matrix(estimator, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        estimator, X_test, y_test, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(estimator, X_test, y_test, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def log_run_to_mlflow(
    name,
    search,
    cv_metrics,
    test_metrics,
    X_test,
    y_test,
) -> None:
    """Log params, metrics, and plots for one model's MLflow run."""
    best_estimator = search.best_estimator_
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(
            {
                k.replace("classifier__", ""): v
                for k, v in search.best_params_.items()
            }
        )
        for metric, (mean, std) in cv_metrics.items():
            mlflow.log_metric(f"cv_{metric}_mean", mean)
            mlflow.log_metric(f"cv_{metric}_std", std)
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        cm_path = ARTIFACTS_DIR / f"{name}_confusion_matrix.png"
        roc_path = ARTIFACTS_DIR / f"{name}_roc_curve.png"
        plot_confusion_matrix(best_estimator, X_test, y_test, cm_path)
        plot_roc_curve(best_estimator, X_test, y_test, roc_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))

        mlflow.sklearn.log_model(
            best_estimator,
            name="model",
            serialization_format="cloudpickle",
        )


def save_best_model(best_name: str, results: dict) -> Path:
    """Persist the best-performing estimator to the model directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = results[best_name]["estimator"]
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def run() -> dict:
    """Train and evaluate all configured models, returning results."""
    configure_mlflow()

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}
    for name, estimator, param_grid in MODEL_CONFIGS:
        pipeline = build_pipeline(estimator)
        search = tune_model(pipeline, param_grid, X_train, y_train, cv)
        cv_metrics = cross_validated_metrics(
            search.best_estimator_, X_train, y_train, cv
        )
        test_metrics = test_set_metrics(search.best_estimator_, X_test, y_test)

        log_run_to_mlflow(
            name,
            search,
            cv_metrics,
            test_metrics,
            X_test,
            y_test,
        )

        results[name] = {
            "best_params": search.best_params_,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "estimator": search.best_estimator_,
        }

        print(f"\n=== {name} ===")
        print("Best params:", search.best_params_)
        print("Cross-validated metrics (mean +/- std):")
        for metric, (mean, std) in cv_metrics.items():
            print(f"  {metric:>10}: {mean:.4f} +/- {std:.4f}")
        print("Held-out test metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric:>10}: {value:.4f}")

    best_name = max(
        results,
        key=lambda n: results[n]["test_metrics"]["roc_auc"],
    )
    model_path = save_best_model(best_name, results)
    print(f"\nBest model by test ROC-AUC: {best_name}")
    print(f"Saved best model to: {model_path}")
    print(
        f"MLflow tracking database: {MLFLOW_DB_PATH} "
        f"(experiment: {MLFLOW_EXPERIMENT})"
    )

    return results


if __name__ == "__main__":
    run()
