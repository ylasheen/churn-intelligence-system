# Customer Churn Intelligence System

An end-to-end, production-style Machine Learning system that predicts customer churn
using **tabular customer data + NLP-based support ticket analysis**, and translates the
prediction into a **business retention recommendation** (who to target, expected revenue saved).

Built by **Youssef Lasheen** — AI & Machine Learning Engineer.

## Why this project is different
Most churn projects stop at "train a model, report accuracy." This project goes further:

1. **Tabular ML** — Gradient Boosting / Voting Ensemble on customer usage & billing data.
2. **NLP layer** — Sentiment & complaint-topic extraction from customer support tickets,
   fed back into the model as engineered features (frustration_score, topic_billing, etc.).
3. **Explainability** — SHAP feature importance to explain *why* a customer is at risk.
4. **Business layer** — Converts churn probability into an actionable retention list with
   estimated revenue impact.
5. **Production-ready** — FastAPI serving layer, MLflow experiment tracking, Docker
   containerization, automated tests, and CI-ready structure.

## Project Structure
```
churn-intelligence-system/
├── data/
│   ├── raw/            # Original, untouched data
│   ├── processed/      # Cleaned, feature-engineered data
│   └── external/       # Support ticket / NLP source data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/            # Data loading & validation
│   ├── features/        # Feature engineering (tabular + NLP)
│   ├── models/           # Training, prediction, evaluation
│   ├── visualization/    # Plots & business reporting
│   └── utils/            # Config & logging helpers
├── models/
│   ├── saved_models/    # Trained model artifacts (.pkl)
│   ├── artifacts/        # Encoders, vectorizers
│   └── logs/             # Training logs
├── reports/
│   ├── figures/
│   ├── reports.html
│   └── summary.md
├── config/config.yaml
├── tests/
├── api/main.py           # FastAPI serving layer
├── dashboard/app.py      # Streamlit business dashboard
├── Dockerfile
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Generate / load data
python src/data/make_dataset.py

# 2. Feature engineering (tabular + NLP)
python src/features/build_features.py

# 3. Train model (tracked with MLflow)
python src/models/train_model.py

# 4. Evaluate & generate business report
python src/models/evaluate_model.py

# 5. Serve predictions via API
uvicorn api.main:app --reload

# 6. (Optional) Launch the interactive business dashboard
streamlit run dashboard/app.py
```

The dashboard (`dashboard/app.py`) gives a non-technical view: KPI overview, a live
"predict a customer" form, the ranked retention priority list, and EDA insights —
no code or API calls required to explore the model.

## Dataset note
The tabular structure mirrors the well-known IBM/Kaggle **Telco Customer Churn** schema
(tenure, contract type, monthly charges, services subscribed, etc.). Since this environment
has no internet access to Kaggle, a **statistically realistic synthetic dataset** (5,000 rows)
is generated with the same schema and realistic churn drivers, plus synthetic support-ticket
text for the NLP layer. Swap `data/raw/telco_churn.csv` with the real Kaggle file any time —
the entire pipeline runs unchanged.

## Tech stack
Python, Pandas, scikit-learn, XGBoost, NLTK/TextBlob (NLP), SHAP, MLflow, FastAPI, Docker, Pytest.

## Author
**Youssef Lasheen** — AI & Machine Learning Engineer
