#!/usr/bin/env python3
# author: Ali Ural
# date: 05-05-2025
# description: LAB 4 - LOGISTIC REGRESSION

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -709, 709)
    return 1.0 / (1.0 + np.exp(-z_clip))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def shuffle_and_split(
    X: np.ndarray, y: np.ndarray, train_ratio: float = 2 / 3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n_train = int(np.ceil(train_ratio * len(X)))
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def standardize(
    X_train: np.ndarray, X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0 
    return (X_train - mu) / sigma, (X_val - mu) / sigma, mu, sigma


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack((np.ones((X.shape[0], 1)), X))


# Training

def train_logreg_gd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.3,
    max_epochs: int = 5000,
    tol: float = 1e-5,
    seed: int = 0,
    verbose: bool = True,
):
    
    n_samples, n_features = X_train.shape
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=0.01, size=n_features)

    train_losses, val_losses = [], []

    last_loss = np.inf
    for epoch in range(1, max_epochs + 1):
        y_prob = sigmoid(X_train @ w)
        grad = X_train.T @ (y_prob - y_train) / n_samples
        w -= lr * grad

        # Losses after update
        train_loss = log_loss(y_train, sigmoid(X_train @ w))
        val_loss = log_loss(y_val, sigmoid(X_val @ w))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Progress every 100 epochs
        if verbose and epoch % 100 == 0:
            print(f"[epoch {epoch:5d}] train={train_loss:.4f}  val={val_loss:.4f}")

        # Convergence
        if abs(last_loss - train_loss) < tol:
            if verbose:
                print(f"Converged at epoch {epoch} (|Δ loss| < {tol})")
            break
        last_loss = train_loss

    return w, train_losses, val_losses


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ w)


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (predict_proba(w, X) >= 0.5).astype(int)


# Metrics

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# Main Routine

def run(dataset_path: Path, lr: float, max_epochs: int, tol: float, silent: bool):
    # 1. Load CSV
    data = np.loadtxt(dataset_path, delimiter=",")
    X, y = data[:, :-1], data[:, -1].astype(int)

    # 2–3. Randomize & split
    X_train, X_val, y_train, y_val = shuffle_and_split(X, y)

    # 4. Standardize with training stats
    X_train, X_val, _, _ = standardize(X_train, X_val)

    # 4b. Add bias
    X_train, X_val = add_bias(X_train), add_bias(X_val)

    # Priors (training set)
    unique, counts = np.unique(y_train, return_counts=True)
    priors = {int(k): v / len(y_train) for k, v in zip(unique, counts)}

    # 5. Train
    w, train_losses, val_losses = train_logreg_gd(
        X_train, y_train, X_val, y_val,
        lr=lr,
        max_epochs=max_epochs,
        tol=tol,
        verbose=not silent,
    )

    # 6–7. Stats
    y_train_pred = predict(w, X_train)
    y_val_pred = predict(w, X_val)
    acc_train = accuracy(y_train, y_train_pred)
    acc_val = accuracy(y_val, y_val_pred)
    prec, rec, f1 = precision_recall_f1(y_val, y_val_pred)

    # Output
    print("\nClass priors (training set):")
    for cls in sorted(priors):
        print(f"  Class {cls}: {priors[cls]:.3f}")
        
    print("\nLogistic Regression statistics:")
    print(f"Training accuracy  : {acc_train * 100:.2f}%")
    print(f"Validation accuracy: {acc_val * 100:.2f}%")
    print(f"Validation precision: {prec * 100:.2f}%")
    print(f"Validation recall   : {rec * 100:.2f}%")
    print(f"Validation F1‑score : {f1 * 100:.2f}%")

    # 8. Plotting
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_losses, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Mean log‑loss")
    plt.title("Learning curve — Logistic Regression (GD)")
    plt.legend()
    plt.grid(True, ls=":", lw=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functional logistic regression (from scratch)")
    parser.add_argument("dataset", nargs="?", default="spambase.data", type=Path,
                        help="CSV path (features + binary label in last column)")
    parser.add_argument("--lr", type=float, default=0.3, help="learning rate (default 0.3)")
    parser.add_argument("--max_epochs", type=int, default=5000, help="training epochs cap")
    parser.add_argument("--tol", type=float, default=1e-5, help="convergence tolerance")
    parser.add_argument("--silent", action="store_true", help="suppress progress prints")
    args = parser.parse_args()

    try:
        run(args.dataset, lr=args.lr, max_epochs=args.max_epochs, tol=args.tol, silent=args.silent)
    except FileNotFoundError:
        sys.exit(f"Dataset not found: {args.dataset}")