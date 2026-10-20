# Heart Disease Risk Prediction — MLOps Report

- **Course:** AIMLCZG523 — MLOps
- **Assignment:** 01
- **Dataset:** UCI Heart Disease (Cleveland)
- **Name:** BADRINATHAN S.
- **Student ID:** 2025AE05369
- **Email:** 2025ae05369@wilp.bits-pilani.ac.in
- **Git Repository:** https://github.com/2025ae05369-badree/heart-disease-prediction-mlops.git
- **Google Drive Link:** https://wilpbitspilaniacin0-my.sharepoint.com/:f:/g/personal/2025ae05369_wilp_bits-pilani_ac_in/IgBM36qUB9bsS4vpV3o4zAECAew-QsCe5ayX4JSdlY00rq4?e=PvuaY0 (Folder containing project files and video recordings)

## 1. Objective

This project builds a binary heart-disease risk classifier from 13 clinical features and delivers it as a tested, containerized, monitored API. The workflow covers data preparation, exploratory analysis, model development, experiment tracking, packaging, continuous integration, deployment, and observability.

## 2. Architecture

```mermaid
flowchart LR
    A[UCI Heart Disease CSV] --> B[Download and Preprocess]
    B --> C[EDA and Feature Engineering]
    C --> D[Model Training]
    D --> E[MLflow Tracking]
    D --> F[Persisted Pipeline]
    F --> G[FastAPI Service]
    G --> H[Docker Image]
    H --> I[Kubernetes Deployment]
    G --> J[Prometheus Metrics]
    J --> K[Grafana Dashboard]
```

Data flows from the UCI dataset through preprocessing and feature
engineering into model training. The selected pipeline is tracked, persisted,
and served by FastAPI. Docker packages the service for local Kubernetes
deployment, while Prometheus and Grafana provide operational visibility.

## 3. Reproducibility

The implementation runs locally with Python (managed via `.venv`). Setup and execution commands are documented in the repository [README](../README.md). The complete pipeline is started with:

```bash
python main.py
```

The pipeline downloads the dataset, preprocesses the data, generates EDA
outputs, trains both classifiers, records the experiments, and saves the
selected model.

## 4. Data and exploratory analysis

- **Source:** UCI Machine Learning Repository,
  `processed.cleveland.data`.
- **Cleaning:** `?` values are converted to missing values and imputed using
  numeric medians or categorical modes.
- **Target:** the original severity score (0–4) is converted to a binary
  target: `0` for no disease and `1` for disease present.
- **Dataset:** 303 rows, 13 features, and no missing values after cleaning.
- **Class distribution:** 164 negative and 139 positive records.

EDA outputs include feature distributions, a correlation heatmap, and class
balance visualizations in `outputs/figures/`.

**EDA visualizations:**

#### Histograms for each feature

![Histograms for each feature](../outputs/figures/histograms.png)

#### Correlation heatmap

![Correlation heatmap](../outputs/figures/correlation_heatmap.png)

#### Class balance visualizations

![Class balance visualizations](../outputs/figures/class_balance.png)

## 5. Feature engineering and model development

The feature pipeline uses a `ColumnTransformer`:

- Standard scaling for `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`.
- One-hot encoding for `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`,
  and `thal`.

Logistic Regression and Random Forest models were tuned with five-fold
stratified cross-validation using ROC-AUC as the optimization metric. Final
metrics were measured on a stratified 80/20 holdout set.

| Model               |      CV ROC-AUC | Test ROC-AUC | Test accuracy | Test precision | Test recall |
| ------------------- | --------------: | -----------: | ------------: | -------------: | ----------: |
| Logistic Regression | 0.9025 ± 0.0144 |   **0.9665** |    **0.8852** |     **0.8387** |      0.9286 |
| Random Forest       | 0.9024 ± 0.0279 |       0.9470 |        0.8689 |         0.8125 |      0.9286 |

Logistic Regression was selected because it achieved better holdout ROC-AUC,
accuracy, and precision with the same recall and lower cross-validation
variance.

### Logistic Regression:

#### Logistic Regression - Confusion matrix

![Logistic Regression - Confusion Matrix](../outputs/figures/mlflow_runs/logistic_regression_confusion_matrix.png)

#### Logistic Regression - ROC curve

![Logistic Regression - ROC Curve](../outputs/figures/mlflow_runs/logistic_regression_roc_curve.png)

### Random Forest:

#### Random Forest - Confusion matrix

![Random Forest - Confusion Matrix](../outputs/figures/mlflow_runs/random_forest_confusion_matrix.png)

#### Random Forest - ROC curve

![Random Forest - ROC Curve](../outputs/figures/mlflow_runs/random_forest_roc_curve.png)

## 6. Experiment tracking and model packaging

MLflow runs locally with the `heart-disease-classification` experiment. Each
run records:

- Tuned model parameters.
- Cross-validation and holdout accuracy, precision, recall, and ROC-AUC.
- The fitted preprocessing-and-model pipeline.
- Confusion matrix and ROC curve artifacts.

The selected pipeline is saved to
`models/heart_disease_model.joblib`. Persisting preprocessing and prediction
as one artifact prevents training and serving transformations from diverging.
Dependencies are pinned in `requirements.txt`.

### MLflow Training Runs

![MLflow Training Runs](../screenshots/mlflow-run/Training-runs.png)

### Logistic Regression

#### Logistic Regression - Metrics

![Logistic Regression - Metrics](../screenshots/mlflow-run/logistic-regression-metrics.png)

#### Logistic Regression - Parameters

![Logistic Regression - Parameters](../screenshots/mlflow-run/logistic-regression-params.png)

#### Logistic Regression - Visualisation

![Logistic Regression - Visualisation](../screenshots/mlflow-run/logistic-regression-visualisation.png)

#### Logistic Regression - Artifacts

![Logistic Regression - Artifacts](../screenshots/mlflow-run/logistic-regression-artifacts.png)

#### Logistic Regression - ML Model

![Logistic Regression - ML Model](../screenshots/mlflow-run/logistic-regression-model.png)

#### Logistic Regression - Dependencies

![Logistic Regression - Dependencies](../screenshots/mlflow-run/logistic-regression-conda-yaml.png)

### Random Forest

#### Random Forest - Metrics

![Random Forest - Metrics](../screenshots/mlflow-run/random-forest-metrics.png)

#### Random Forest - Parameters

![Random Forest - Parameters](../screenshots/mlflow-run/random-forest-params.png)

#### Random Forest - Visualisation

![Random Forest - Visualisation](../screenshots/mlflow-run/random-forest-visualisation.png)

#### Random Forest - Artifacts

![Random Forest - Artifacts](../screenshots/mlflow-run/random-forest-artifacts.png)

#### Random Forest - ML Model

![Random Forest - ML Model](../screenshots/mlflow-run/random-forest-model.png)

#### Random Forest - Dependencies

![Random Forest - Dependencies](../screenshots/mlflow-run/random-forest-conda-yaml.png)

## 7. API and containerization

The FastAPI service exposes:

| Endpoint        | Purpose                                    |
| --------------- | ------------------------------------------ |
| `GET /health`   | Confirms that the model is available       |
| `POST /predict` | Returns prediction, probability, and label |
| `GET /metrics`  | Exposes Prometheus metrics                 |

The `PatientRecord` request model validates the 13 input features, and the
typed response model preserves the integer prediction value and response
schema. The Docker image uses Python 3.12, installs the pinned dependencies,
and serves the API on port 8000.

### OpenAPI

![OpenAPI documentation](../screenshots/fastapi-service/openapi.png)

### Health endpoint

![Health endpoint](../screenshots/fastapi-service/health.png)

### Predict endpoint

#### Predict no disease

![Predict no disease](../screenshots/fastapi-service/predict-no-disease.png)

#### Predict disease present

![Predict disease present](../screenshots/fastapi-service/predict-disease-present.png)

#### Unprocessable entity error

![Unprocessable entity error](../screenshots/fastapi-service/unprocessable-entity-error.png)

### Metrics endpoint

![Metrics endpoint](../screenshots/fastapi-service/metrics.png)

## 8. CI and automated testing

The test suite covers data cleaning, target encoding, model training, metric
generation, and model packaging. `flake8` checks code quality.

The GitHub Actions workflow runs the following sequence on pushes and pull
requests to the main branches:

```text
flake8 .  →  pytest -q  →  python main.py
```

### GitHub Actions workflow result

![CI workflow result](../screenshots/ci-run/github-workflows.png)

### Workflow triggers on pull requests and pushes to the main branches

![Pull request workflow triggers](../screenshots/ci-run/pull-request.png)
![Merge into main workflow triggers](../screenshots/ci-run/merge-into-main.png)
![Merge into main build info](../screenshots/ci-run/merge-into-main-build-info.png)

## 9. Deployment

`deployment.yaml` defines a Kubernetes Deployment with one API replica and a
Service that maps port 80 to the container's port 8000. The image can be
loaded into Minikube and exposed locally with `kubectl port-forward`.

Deployment verification requires:

1. A ready Kubernetes node.
2. A running API pod.
3. A reachable Service.
4. Successful health and prediction requests.

### Kubernetes deployment snapshots

Kubernetes deployment snapshots include loading the image, state of the deployment, pods, and services.
![Snapshot 1](../screenshots/kubernetes/snapshot-1.png)
![Snapshot 2](../screenshots/kubernetes/snapshot-2.png)
![Snapshot 3](../screenshots/kubernetes/snapshot-3.png)

### Port-forwarding to access the service locally

![Port-forwarding](../screenshots/kubernetes/ports.png)

## 10. Monitoring and logging

The API logs health and prediction requests through Python logging, while
Uvicorn provides access logs. `prometheus-fastapi-instrumentator` exposes
request counts and latency histograms at `/metrics`.

The Compose monitoring stack provides:

- Prometheus for metric collection and queries.
- Grafana for the provisioned API dashboard.
- Recording rules for request rate, latency, success rate, and error rate.

### Prometheus query example

![Custom prometheus query](../screenshots/prometheus/custom-metrics.png)
![HTTP requests prometheus query](../screenshots/prometheus/http-requests.png)
![Predict requests prometheus query](../screenshots/prometheus/predict-requests.png)

### Grafana dashboard

![Snapshot 1](../screenshots/grafana/snapshot-1.png)
![Snapshot 2](../screenshots/grafana/snapshot-2.png)

### API request logs

![API request logs](../screenshots/fastapi-service/metrics.png)

## 11. Evidence register

The table below links each evidence category to the folder containing the
corresponding images embedded in the report.

| Evidence                   | Image folder path(s)                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| Environment version check  | `../screenshots/version-check/`                                                            |
| End-to-end model execution | `../screenshots/e2e-model-execution/`                                                      |
| EDA outputs                | `../outputs/figures/`                                                                      |
| Model evaluation           | `../outputs/figures/mlflow_runs/`                                                          |
| MLflow comparison          | `../screenshots/mlflow-run/`                                                               |
| API health and prediction  | `../screenshots/fastapi-service/`                                                          |
| CI workflow                | `../screenshots/ci-run/`                                                                   |
| Kubernetes deployment      | `../screenshots/kubernetes/`                                                               |
| Monitoring                 | `../screenshots/prometheus/`, `../screenshots/grafana/`, `../screenshots/fastapi-service/` |
