# Heart Disease Risk Prediction

An end-to-end MLOps project that trains a heart-disease classifier from the
UCI Heart Disease dataset and serves it through a monitored FastAPI service.
The project includes data preparation, exploratory analysis, model training,
MLflow tracking, automated tests, containerization, monitoring, and local
Kubernetes deployment.

## Prerequisites

- Python (managed via `.venv`)
- [`uv`](https://docs.astral.sh/uv/)
- `kubectl` and either Docker Desktop Kubernetes or Minikube for deployment

Run the commands below from the repository root.

## Quick start

### 1. Create the environment

Check the required tools before starting:

```bash
python3 --version
uv --version
docker --version
kubectl version --client
```

Install `uv` with Astral's official installer if it is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Setup the virtual environment and install dependencies:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For subsequent sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

### 2. Run the ML pipeline

The pipeline downloads and cleans the data, generates EDA plots, trains and
tunes Logistic Regression and Random Forest models, logs the runs to MLflow,
and saves the best model.

```bash
python main.py
```

Generated artifacts include:

| Artifact         | Location                                 |
| ---------------- | ---------------------------------------- |
| Cleaned dataset  | `data/processed/heart_disease_clean.csv` |
| EDA plots        | `outputs/figures/`                       |
| MLflow database  | `mlflow.db`                              |
| MLflow artifacts | `mlruns/`                                |
| Packaged model   | `models/heart_disease_model.joblib`      |

Run individual stages when needed:

```bash
python data/download_data.py
python src/data_preprocessing.py
python src/eda.py
python src/train.py
```

### 3. Run tests and linting

```bash
python -m flake8 .
python -m pytest -q
```

The CI workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml) runs
the same lint, test, and training checks.

## MLflow

Start the local tracking UI in a separate terminal:

```bash
source .venv/bin/activate
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root "$(pwd)/mlruns"
```

Open <http://localhost:5000> and select the
`heart-disease-classification` experiment.

## Run the API

Start the service:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API provides:

| Endpoint        | Purpose                    |
| --------------- | -------------------------- |
| `GET /health`   | Service health check       |
| `POST /predict` | Predict heart-disease risk |
| `GET /metrics`  | Prometheus metrics         |

Example requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Generate varied API traffic before querying metrics. The first two requests exercise both model outcomes (`no_disease` and `disease_present`):

```bash
curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'

curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

## Docker

Build and run the API container:

```bash
docker build -t heart-disease-api:latest .
docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-api:latest
```

Use the API commands above while the container is running.

## Monitoring stack

The Compose stack runs the API, Prometheus, and Grafana:

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

Open:

- API: <http://localhost:8000>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (`admin` / `admin`)

The Grafana dashboard is provisioned automatically in the **MLOps** folder.

Prometheus recording rules are defined in
[`monitoring/rules/api_rules.yml`](monitoring/rules/api_rules.yml), and the
dashboard is defined in
[`monitoring/dashboards/heart-disease-api-overview.json`](monitoring/dashboards/heart-disease-api-overview.json).

If the standalone API container is still running, remove it before starting
the Compose stack because both use port 8000:

```bash
docker rm -f heart-disease-api
```

Stop the monitoring stack with:

```bash
docker compose -f docker-compose.monitoring.yml down
```

## Kubernetes

Enable Kubernetes in Docker Desktop or start Minikube, then build the image:

```bash
docker build -t heart-disease-api:latest .
```

For Minikube, load the local image into the cluster:

```bash
minikube image load heart-disease-api:latest
```

Apply the deployment and expose the service locally:

```bash
kubectl apply -f deployment.yaml
kubectl get pods -l app=heart-disease-api
kubectl get service heart-disease-api
kubectl port-forward service/heart-disease-api 8000:80
```

Run the health and prediction requests from the API section while port
forwarding is active.

Remove the deployment when finished:

```bash
kubectl delete -f deployment.yaml
```

### Troubleshooting Kubernetes Deployment

If the pod remains `0/1 Pending`, inspect the scheduler, node, and cluster
networking:

```bash
kubectl describe pod -l app=heart-disease-api
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl logs -l app=heart-disease-api

kubectl get nodes
kubectl describe node minikube
kubectl get pods -n kube-system -o wide
```

An untolerated `node.kubernetes.io/not-ready` taint together with
`NetworkPluginNotReady` means Minikube's CNI did not initialize. If the
`kindnet` pod is in `ImagePullBackOff` because DNS resolution for Docker Hub
timed out, preload the exact image tag reported by `kubectl describe`:

```bash
kubectl describe pod -n kube-system -l k8s-app=kindnet
docker pull docker.io/kindest/kindnetd:v20260820-69b56db7
minikube image load docker.io/kindest/kindnetd:v20260820-69b56db7
kubectl delete pod -n kube-system -l k8s-app=kindnet
kubectl wait --for=condition=Ready node/minikube --timeout=3m
kubectl rollout status deployment/heart-disease-api --timeout=5m
```

## Repository structure

```text
.
├── main.py                              # Runs the complete pipeline
├── app.py                               # FastAPI prediction, health, and metrics endpoints
├── Dockerfile                           # API container definition
├── docker-compose.monitoring.yml        # API, Prometheus, and Grafana stack
├── deployment.yaml                      # Kubernetes Deployment and Service
├── requirements.txt                     # Python dependencies
├── data/
│   └── download_data.py                 # Downloads the UCI dataset
├── src/
│   ├── data_preprocessing.py            # Cleans and prepares the data
│   ├── eda.py                           # Generates exploratory analysis and plots
│   ├── features.py                      # Defines feature transformations
│   └── train.py                         # Trains models and packages the best pipeline
├── tests/                               # Unit and API tests
├── monitoring/                          # Prometheus and Grafana configuration
├── outputs/                             # Output files such as plots and analysis results
├── screenshots/                         # Screenshots of the application
├── report/
│   └── Final_Report.md                  # Project report
└── models/                              # Packaged models
```
