# Architecture Overview

```mermaid
flowchart LR
    A[UCI Heart Disease CSV] --> B[Data Download + Preprocessing]
    B --> C[EDA + Feature Engineering]
    C --> D[Model Training]
    D --> E[MLflow Experiment Tracking]
    D --> F[Persisted Pipeline Model]
    F --> G[FastAPI Prediction Service]
    G --> H[Docker Container]
    H --> I[Kubernetes Deployment]
    G --> J[Prometheus Metrics]
    J --> K[Grafana Dashboard]
```

This layout reflects the production-ready flow used in the assignment: raw data enters the pipeline, a trained model is validated and logged, then the packaged model is served through a containerized API, monitored, and exposed behind a Kubernetes service.
