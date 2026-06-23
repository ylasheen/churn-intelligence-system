"""Unit tests for the data generation & integrity layer."""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.make_dataset import generate_synthetic_telco, generate_support_tickets


def test_generate_synthetic_telco_shape():
    df = generate_synthetic_telco(n_samples=100, random_state=42)
    assert df.shape[0] == 100
    assert "churn" in df.columns
    assert set(df["churn"].unique()).issubset({0, 1})


def test_generate_synthetic_telco_no_nulls():
    df = generate_synthetic_telco(n_samples=100, random_state=42)
    assert df.isnull().sum().sum() == 0


def test_generate_support_tickets_alignment():
    df = generate_synthetic_telco(n_samples=50, random_state=1)
    tickets = generate_support_tickets(df["customer_id"], df["churn"], random_state=1)
    assert len(tickets) == len(df)
    assert set(tickets["customer_id"]) == set(df["customer_id"])


def test_monthly_charges_within_bounds():
    df = generate_synthetic_telco(n_samples=200, random_state=7)
    assert df["monthly_charges"].between(18, 120).all()
