"""Reusable plotting helpers for EDA and reporting."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_churn_distribution(df: pd.DataFrame, target_col: str, save_path: Path):
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=target_col)
    plt.title("Churn Distribution")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_numeric_vs_churn(df: pd.DataFrame, numeric_col: str, target_col: str, save_path: Path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=target_col, y=numeric_col)
    plt.title(f"{numeric_col} vs {target_col}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save_path: Path):
    plt.figure(figsize=(7, 4))
    results_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar")
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
