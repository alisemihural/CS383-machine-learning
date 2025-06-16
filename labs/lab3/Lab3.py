#!/usr/bin/env python3
# author: Ali Ural
# date: 28-04-2025
# description: LAB 3 - LINEAR REGRESSION

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV and one‑hot encode categorical columns
def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = pd.get_dummies(frame, columns=["sex", "smoker", "region"], drop_first=True)
    return frame.astype(float)

# Shuffle *df* and return (train_df, val_df)
def train_val_split(df: pd.DataFrame, seed: int, ratio: float = 2 / 3):
    rng = np.random.default_rng(seed)
    shuffled = df.sample(frac=1, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)
    cut = int(len(df) * ratio)
    return shuffled.iloc[:cut], shuffled.iloc[cut:]

# Return (X, y) with a bias column appended to X
def to_matrix(df: pd.DataFrame):
    X = df.drop(columns=["charges"]).to_numpy()

    # bias term
    X = np.c_[X, np.ones(len(X))]  

    y = df["charges"].to_numpy()
    return X, y

# Compute weights via the normal equation using a pseudo‑inverse
def closed_form(X: np.ndarray, y: np.ndarray):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(
        100 * np.mean(np.abs(pred - actual) / ((np.abs(actual) + np.abs(pred))))
    )


def run(csv_file: Path, seed: int, show_plot: bool):
    df = load_csv(csv_file)
    train_df, val_df = train_val_split(df, seed)

    X_tr, y_tr = to_matrix(train_df)
    X_va, y_va = to_matrix(val_df)

    w = closed_form(X_tr, y_tr)

    pred_tr = X_tr @ w
    pred_va = X_va @ w

    print(f"Training   RMSE: {rmse(y_tr, pred_tr):,.2f}  SMAPE: {smape(y_tr, pred_tr):.2f}%")
    print(f"Validation RMSE: {rmse(y_va, pred_va):,.2f}  SMAPE: {smape(y_va, pred_va):.2f}%")

    if show_plot:
        plt.scatter(y_va, pred_va, alpha=0.6)
        line = [y_va.min(), y_va.max()]
        plt.plot(line, line, "--")
        plt.xlabel("Actual charges")
        plt.ylabel("Predicted charges")
        plt.title("Predicted vs. actual (validation)")
        plt.show()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default="insurance.csv", help="CSV file path")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--show", action="store_true", help="Show scatter plot")
    opts = parser.parse_args(argv)

    if not opts.data.exists():
        sys.exit(f"File not found: {opts.data}")

    run(opts.data, opts.seed, opts.show)


if __name__ == "__main__":
    main(sys.argv[1:])
