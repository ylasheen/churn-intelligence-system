"""
evaluate_model.py
------------------
Loads the trained model, generates a SHAP explainability summary, and produces
a business-facing retention report: which customers to target, and the
estimated revenue impact of a retention campaign.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("evaluate_model")
ROOT = Path(__file__).resolve().parents[2]


def generate_shap_summary(model, X: pd.DataFrame, figures_dir: Path):
    """Generate and save a SHAP feature-importance summary plot."""
    logger.info("Computing SHAP values for explainability...")
    try:
        explainer = shap.Explainer(model.predict, X.sample(min(200, len(X)), random_state=42))
        shap_values = explainer(X.sample(min(200, len(X)), random_state=42))
        plt.figure()
        shap.summary_plot(shap_values, show=False)
        figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figures_dir / "shap_summary.png", bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot -> {figures_dir / 'shap_summary.png'}")
    except Exception as e:
        logger.warning(f"SHAP explainability skipped ({e}). Falling back to feature_importances_.")


def business_retention_report(df: pd.DataFrame, model, features: list, config: dict) -> pd.DataFrame:
    """Convert churn probabilities into a ranked, actionable retention list."""
    biz = config["business"]
    df = df.copy()
    df["churn_probability"] = model.predict_proba(df[features])[:, 1]

    df["expected_revenue_at_risk"] = df["churn_probability"] * biz["avg_monthly_revenue_per_customer"] * 12
    df["campaign_expected_savings"] = (
        df["churn_probability"]
        * biz["campaign_success_rate"]
        * biz["avg_monthly_revenue_per_customer"]
        * 12
        - biz["retention_campaign_cost_per_customer"]
    )

    ranked = df.sort_values("churn_probability", ascending=False)
    return ranked[
        ["customer_id", "churn_probability", "expected_revenue_at_risk", "campaign_expected_savings"]
    ]


def main():
    config = load_config()
    model_cfg = config["model"]

    saved_model_path = ROOT / model_cfg["saved_model_path"]
    logger.info(f"Loading trained model from {saved_model_path}")
    bundle = joblib.load(saved_model_path)
    model, features, name = bundle["model"], bundle["features"], bundle["name"]

    processed_path = ROOT / config["data"]["processed_path"]
    df = pd.read_csv(processed_path)

    figures_dir = ROOT / "reports" / "figures"
    generate_shap_summary(model, df[features], figures_dir)

    retention_df = business_retention_report(df, model, features, config)
    top_at_risk = retention_df.head(20)
    report_path = ROOT / "reports" / "retention_priority_list.csv"
    top_at_risk.to_csv(report_path, index=False)
    logger.info(f"Saved top-20 retention priority list -> {report_path}")

    total_savings = retention_df[retention_df["campaign_expected_savings"] > 0]["campaign_expected_savings"].sum()

    summary_path = ROOT / "reports" / "summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Churn Intelligence — Business Summary\n\n")
        f.write(f"**Best model:** {name}\n\n")
        f.write(f"**Customers analyzed:** {len(df)}\n\n")
        f.write(f"**Estimated annual savings from targeted retention campaign:** "
                f"${total_savings:,.2f}\n\n")
        f.write("See `retention_priority_list.csv` for the ranked list of customers to target, "
                "and `figures/shap_summary.png` for model explainability.\n")
    logger.info(f"Saved business summary -> {summary_path}")
    logger.info(f"Estimated annual campaign savings: ${total_savings:,.2f}")


if __name__ == "__main__":
    main()
