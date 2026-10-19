from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True


def test_predict_endpoint() -> None:
    payload = {
        "age": 52,
        "sex": 1,
        "cp": 3,
        "trestbps": 120,
        "chol": 230,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "probability", "label"}
    assert body["prediction"] in {0, 1}
    assert 0.0 <= body["probability"] <= 1.0


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_openapi_groups_endpoints_by_domain() -> None:
    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/predict"]["post"]["tags"] == ["Prediction"]
    assert schema["paths"]["/health"]["get"]["tags"] == ["Health"]
    assert schema["paths"]["/metrics"]["get"]["tags"] == ["Monitoring"]
    assert {tag["name"] for tag in schema["tags"]} == {
        "Prediction",
        "Health",
        "Monitoring",
    }
