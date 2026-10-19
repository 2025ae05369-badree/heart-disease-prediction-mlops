from __future__ import annotations

import logging
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

MODEL_PATH = ROOT / "models" / "heart_disease_model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("heart_disease_api")

if not MODEL_PATH.exists():
    try:
        from src import train as train_module
        train_module.run()
    except Exception:
        try:
            import train as train_module  # type: ignore
            train_module.run()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"Unable to build model before API startup: {exc}"
            ) from exc

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

# Exposes request-count/latency metrics in Prometheus format at /metrics.
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    logger.info("Incoming %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Completed %s %s in %.2f ms with status %s",
        request.method,
        request.url.path,
        elapsed_ms,
        response.status_code,
    )
    return response


class PatientRecord(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    model_config = ConfigDict(extra="forbid")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str


@app.get("/health")
def health() -> dict[str, str | bool]:
    model_ready = MODEL_PATH.exists()
    logger.info("GET /health -> model_ready=%s", model_ready)
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord) -> PredictionResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train the pipeline first.",
        )

    model = joblib.load(MODEL_PATH)
    payload = pd.DataFrame([record.model_dump()])
    prediction = int(model.predict(payload)[0])
    probability = float(model.predict_proba(payload)[0, 1])
    logger.info(
        "POST /predict -> prediction=%d probability=%.6f",
        prediction,
        probability,
    )
    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 6),
        label="disease_present" if prediction == 1 else "no_disease",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
