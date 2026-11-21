"""FastAPI prediction service."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline

from . import model

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/sentiment.joblib"))


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float


app = FastAPI(title="Turkish Sentiment Classifier")
loaded_model: Pipeline | None = None

try:
    loaded_model = model.load(MODEL_PATH)
except FileNotFoundError:
    loaded_model = None


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": loaded_model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    result = model.predict(loaded_model, [request.text])[0]
    return PredictionResponse(**result)
