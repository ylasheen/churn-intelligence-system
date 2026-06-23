"""Unit tests for model training and evaluation utilities."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.train_model import evaluate, get_models


def _toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + X["b"] > 0).astype(int)
    return X, y


def test_get_models_returns_expected_keys():
    models = get_models(random_state=42)
    expected = {"logistic_regression", "random_forest", "xgboost", "voting_ensemble"}
    assert expected.issubset(models.keys())


def test_evaluate_returns_all_metrics():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    metrics = evaluate(model, X, y)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_model_predicts_binary_labels():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
