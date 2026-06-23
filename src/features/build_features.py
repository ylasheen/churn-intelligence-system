"""
build_features.py
------------------
Merges tabular customer data with NLP-derived features from support tickets,
then encodes everything into a model-ready feature matrix.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("build_features")

ROOT = Path(__file__).resolve().parents[2]

CATEGORY_TO_TOPIC_FLAGS = {
    "billing": {"topic_billing": 1, "topic_service": 0, "topic_leaving": 0},
    "service_quality": {"topic_billing": 0, "topic_service": 1, "topic_leaving": 0},
    "considering_leaving": {"topic_billing": 0, "topic_service": 0, "topic_leaving": 1},
    "satisfied": {"topic_billing": 0, "topic_service": 0, "topic_leaving": 0},
}


def add_nlp_features(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sentiment polarity + topic flags from each support ticket."""
    logger.info("Extracting NLP sentiment & topic features from support tickets...")
    tickets_df = tickets_df.copy()
    tickets_df["sentiment_polarity"] = tickets_df["ticket_text"].apply(
        lambda t: TextBlob(t).sentiment.polarity
    )
    tickets_df["frustration_score"] = (1 - tickets_df["sentiment_polarity"]) / 2  # 0 (happy) -> 1 (frustrated)

    topic_flags = tickets_df["category"].map(CATEGORY_TO_TOPIC_FLAGS).apply(pd.Series)
    tickets_df = pd.concat([tickets_df, topic_flags], axis=1)

    return tickets_df[
        ["customer_id", "sentiment_polarity", "frustration_score", "topic_billing", "topic_service", "topic_leaving"]
    ]


def encode_tabular(df: pd.DataFrame, encoders_path: Path) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical tabular columns and persist the encoders."""
    categorical_cols = [
        "gender", "partner", "dependents", "contract", "internet_service",
        "tech_support", "online_security", "paperless_billing", "payment_method",
    ]
    encoders = {}
    df = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, encoders_path)
    logger.info(f"Saved categorical encoders -> {encoders_path}")
    return df, encoders


def main():
    config = load_config()

    raw_path = ROOT / config["data"]["raw_path"]
    tickets_path = ROOT / config["data"]["tickets_path"]
    processed_path = ROOT / config["data"]["processed_path"]
    encoders_path = ROOT / config["model"]["encoder_path"]

    logger.info(f"Loading raw tabular data from {raw_path}")
    df = pd.read_csv(raw_path)

    logger.info(f"Loading support tickets from {tickets_path}")
    tickets_df = pd.read_csv(tickets_path)
    nlp_features = add_nlp_features(tickets_df)

    logger.info("Merging tabular + NLP features...")
    merged = df.merge(nlp_features, on="customer_id", how="left")

    encoded, _ = encode_tabular(merged, encoders_path)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(processed_path, index=False)
    logger.info(f"Saved processed feature set -> {processed_path} ({encoded.shape[0]} rows, {encoded.shape[1]} cols)")


if __name__ == "__main__":
    main()
