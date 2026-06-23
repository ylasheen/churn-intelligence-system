"""
make_dataset.py
----------------
Generates a statistically realistic synthetic Telco-style customer churn dataset
(matching the well-known Kaggle/IBM Telco Customer Churn schema) plus a matching
support-tickets text dataset for the NLP layer.

This keeps the project fully runnable offline. Replace data/raw/telco_churn.csv
with the real Kaggle file at any time -- the rest of the pipeline is unaffected.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("make_dataset")


COMPLAINT_TEMPLATES = {
    "billing": [
        "I was overcharged again this month and nobody explained why.",
        "My bill keeps increasing every month for no clear reason.",
        "The billing department never resolves my disputes.",
        "I think there is a hidden fee on my invoice, very frustrating.",
    ],
    "service_quality": [
        "The internet keeps disconnecting several times a day.",
        "Customer support took forever to answer and didn't help at all.",
        "Call quality has been terrible the past few weeks.",
        "Technician never showed up for the scheduled appointment.",
    ],
    "satisfied": [
        "Everything works great, no complaints at all this month.",
        "Support resolved my issue quickly, very happy with the service.",
        "I am satisfied with the speed and reliability of my connection.",
        "Great experience overall, keep up the good work.",
    ],
    "considering_leaving": [
        "I am seriously thinking about switching providers soon.",
        "If this keeps happening I will cancel my subscription.",
        "Your competitor is offering a much better deal, I might switch.",
        "This is the last straw, I want to cancel my contract.",
    ],
}


def generate_synthetic_telco(n_samples: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(0, 73, n_samples)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20])
    monthly_charges = np.round(rng.normal(65, 25, n_samples).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * (tenure + 1) * rng.uniform(0.85, 1.0, n_samples), 2)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
    )
    paperless_billing = rng.choice(["Yes", "No"], n_samples, p=[0.6, 0.4])
    tech_support = rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.3, 0.5, 0.2])
    online_security = rng.choice(["Yes", "No", "No internet service"], n_samples, p=[0.3, 0.5, 0.2])
    senior_citizen = rng.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_samples, p=[0.3, 0.7])
    gender = rng.choice(["Male", "Female"], n_samples)
    num_support_calls = rng.poisson(1.5, n_samples)

    # Latent churn probability driven by realistic business factors
    churn_logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        + 0.9 * (internet_service == "Fiber optic")
        - 0.035 * tenure
        + 0.015 * (monthly_charges - 65)
        + 0.25 * num_support_calls
        + 0.5 * (tech_support == "No")
        + 0.3 * (payment_method == "Electronic check")
        - 0.3 * (partner == "Yes")
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "churn": churn,
        }
    )
    return df


def generate_support_tickets(customer_ids: pd.Series, churn: pd.Series, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + 1)
    rows = []
    for cid, churned in zip(customer_ids, churn):
        # Churned customers are more likely to have complaint-heavy tickets
        if churned == 1:
            category = rng.choice(
                ["billing", "service_quality", "considering_leaving", "satisfied"],
                p=[0.3, 0.35, 0.25, 0.10],
            )
        else:
            category = rng.choice(
                ["billing", "service_quality", "considering_leaving", "satisfied"],
                p=[0.15, 0.15, 0.05, 0.65],
            )
        text = rng.choice(COMPLAINT_TEMPLATES[category])
        rows.append({"customer_id": cid, "ticket_text": text, "category": category})
    return pd.DataFrame(rows)


def main():
    config = load_config()
    n_samples = config["data"]["n_samples"]
    random_state = config["data"]["random_state"]

    logger.info(f"Generating {n_samples} synthetic customer records...")
    df = generate_synthetic_telco(n_samples, random_state)

    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(__file__).resolve().parents[2] / raw_path, index=False)
    logger.info(f"Saved tabular data -> {raw_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    logger.info("Generating synthetic support ticket text data...")
    tickets_df = generate_support_tickets(df["customer_id"], df["churn"], random_state)
    tickets_path = Path(config["data"]["tickets_path"])
    tickets_df.to_csv(Path(__file__).resolve().parents[2] / tickets_path, index=False)
    logger.info(f"Saved support tickets -> {tickets_path} ({tickets_df.shape[0]} rows)")

    logger.info(f"Churn rate in generated data: {df['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
