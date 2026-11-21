import pandas as pd
from fastapi.testclient import TestClient

from src.sentiment import app as app_module
from src.sentiment import model


def test_health():
    client = TestClient(app_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_and_validation():
    df = pd.DataFrame({
        "text": ["harika", "çok güzel", "kötü", "berbat"],
        "label": ["positive", "positive", "negative", "negative"],
    })
    app_module.loaded_model = model.build_pipeline().fit(df["text"], df["label"])
    client = TestClient(app_module.app)

    response = client.post("/predict", json={"text": " çok güzel "})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"positive", "negative"}
    assert body["text"] == "çok güzel"
    assert 0.0 <= body["confidence"] <= 1.0

    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_returns_503_without_model():
    previous = app_module.loaded_model
    app_module.loaded_model = None
    try:
        client = TestClient(app_module.app)
        response = client.post("/predict", json={"text": "harika"})
        assert response.status_code == 503
    finally:
        app_module.loaded_model = previous
