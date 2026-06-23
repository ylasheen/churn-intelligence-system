"""
train_model.py
---------------
Trains and compares multiple churn-prediction models (Logistic Regression,
Random Forest, XGBoost, Voting Ensemble), tracks every run with MLflow,
and persists the best model + a SHAP explainability summary.
"""
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("train_model")
ROOT = Path(__file__).resolve().parents[2]


def get_models(random_state: int) -> dict:
    log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=random_state, eval_metric="logloss",
    )
    voting = VotingClassifier(
        estimators=[("log_reg", log_reg), ("rf", rf), ("xgb", xgb)], voting="soft"
    )
    return {
        "logistic_regression": log_reg,
        "random_forest": rf,
        "xgboost": xgb,
        "voting_ensemble": voting,
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    config = load_config()
    model_cfg = config["model"]
    target_col = model_cfg["target_column"]

    processed_path = ROOT / config["data"]["processed_path"]
    logger.info(f"Loading processed feature set from {processed_path}")
    df = pd.read_csv(processed_path)

    feature_cols = [c for c in df.columns if c not in [target_col, "customer_id"]]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg["test_size"], random_state=model_cfg["random_state"], stratify=y
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    Path(ROOT / "mlflow_runs").mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    models = get_models(model_cfg["random_state"])
    results = {}
    best_model_name, best_model, best_f1 = None, None, -1

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            results[name] = metrics

            mlflow.log_params({"model_type": name})
            mlflow.log_metrics(metrics)

            logger.info(f"{name} -> {metrics}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = name
                best_model = model

    logger.info(f"Best model: {best_model_name} (F1={best_f1:.4f})")

    saved_model_path = ROOT / model_cfg["saved_model_path"]
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "features": feature_cols, "name": best_model_name}, saved_model_path)
    logger.info(f"Saved best model -> {saved_model_path}")

    results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    results_path = ROOT / "reports" / "model_comparison.csv"
    results_df.to_csv(results_path)
    logger.info(f"Saved model comparison table -> {results_path}")

    return best_model_name, results_df


if __name__ == "__main__":
    main()
