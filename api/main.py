"""
FastAPI serving layer for the Customer Churn Intelligence System.

Run with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "saved_models" / "churn_model.pkl"

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predicts customer churn probability and recommends retention action.",
    version="1.0.0",
)

_bundle = None


def get_model_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class CustomerFeatures(BaseModel):
    gender: int
    senior_citizen: int
    partner: int
    dependents: int
    tenure: int
    contract: int
    internet_service: int
    tech_support: int
    online_security: int
    paperless_billing: int
    payment_method: int
    monthly_charges: float
    total_charges: float
    num_support_calls: int
    sentiment_polarity: float
    frustration_score: float
    topic_billing: int
    topic_service: int
    topic_leaving: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Customer Churn Intelligence API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: CustomerFeatures):
    bundle = get_model_bundle()
    model, feature_order = bundle["model"], bundle["features"]

    row = pd.DataFrame([features.model_dump()])[feature_order]
    proba = float(model.predict_proba(row)[0, 1])
    prediction = int(proba >= 0.5)

    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": prediction,
        "risk_level": "high" if proba >= 0.6 else "medium" if proba >= 0.3 else "low",
        "model_used": bundle["name"],
    }
