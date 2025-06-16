#!/usr/bin/env python3
# author: Ali Ural
# date: 26-05-2025
# description: LAB 6 - K Nearest Neighbors

import numpy as np
import pandas as pd
import random, math, os, re
from collections import Counter
from pathlib import Path
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
K_CTG = 7
K_YALE = 3

# Utils
def euclidean(a, b):
    return math.sqrt(np.sum((a - b) ** 2))

def neighbors(X, y, q, k):
    d = np.sum((X - q) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    return y[idx]

def predict(X, y, q, k):
    labs = neighbors(X, y, q, k)
    return Counter(labs).most_common(1)[0][0]

def accuracy(y_true, y_pred):
    return 100 * np.mean(y_true == y_pred)

def priors(y):
    cnt = Counter(y)
    n = len(y)
    return {c: cnt[c] / n for c in sorted(cnt)}

def conf_matrix(y_true, y_pred, labels):
    labels = sorted(labels)
    m = np.zeros((len(labels), len(labels)), int)
    m_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        m[m_idx[t]][m_idx[p]] += 1
    return labels, m

def minmax_fit(X):
    mn, mx = X.min(0), X.max(0)
    rng = np.where(mx - mn == 0, 1, mx - mn)
    return mn, rng

def minmax_tr(X, mn, rng):
    return (X - mn) / rng


# PART 1  – CTG

def run_ctg(path: str, k: int):
    print("PART 1: CTG")
    df = pd.read_csv(path, na_values=["?", " "])
    df.dropna(inplace=True)

    y = df["NSP"].astype(int).values if "NSP" in df else df.iloc[:, -1].astype(int).values
    X = df.drop(columns=[c for c in ["CLASS", "NSP"] if c in df]).astype(float).values

    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = math.ceil(len(X) * 2 / 3)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]

    mn, rng = minmax_fit(X_tr)
    X_tr, X_va = minmax_tr(X_tr, mn, rng), minmax_tr(X_va, mn, rng)

    preds = np.array([predict(X_tr, y_tr, q, k) for q in X_va])

    print("K =", k, "\n")
    for c, p in priors(y_tr).items():
        print(f"Class {c}: {p:.4f}")
    print("\nAccuracy:", f"{accuracy(y_va, preds):.2f}% \n")

    lbls, cm = conf_matrix(y_va, preds, np.unique(y))
    header = "    " + " ".join(f"{l:4d}" for l in lbls)

    print("Confusion Matrix:")
    print(header)
    for i, l in enumerate(lbls):
        row = " ".join(f"{n:4d}" for n in cm[i])
        print(f"{l:4d} {row}")

# PART 2  – YALE FACES

def load_faces(root: Path):
    feats, labs = [], []
    for f in root.glob("subject*.*"):
        m = re.match(r"subject(\d{2})\.", f.name)

        if not m: continue
        sid = int(m.group(1))

        if sid == 1: continue
        img = Image.open(f).convert("L").resize((40, 40))
        feats.append(np.array(img, float).flatten() / 255.0)
        labs.append(sid)
    return np.vstack(feats), np.array(labs)

def run_faces(root: str, k: int):
    print("\n \nPART 2: Yale Faces")
    X, y = load_faces(Path(root))

    X_tr, y_tr, X_va, y_va = [], [], [], []
    for sid in np.unique(y):
        idx = np.where(y == sid)[0]
        np.random.shuffle(idx)
        cut = math.ceil(len(idx) * 2 / 3)
        X_tr.extend(X[idx[:cut]]);   y_tr.extend(y[idx[:cut]])
        X_va.extend(X[idx[cut:]]);   y_va.extend(y[idx[cut:]])

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_va, y_va = np.array(X_va), np.array(y_va)

    preds = np.array([predict(X_tr, y_tr, q, k) for q in X_va])

    print("K =", k, "\n")
    for c, p in priors(y_tr).items():
        print(f"Subject {c}: {p:.4f}")

    print("\nAccuracy:", f"{accuracy(y_va, preds):.2f}%")

    lbls, cm = conf_matrix(y_va, preds, np.unique(y))
    header = "    " + " ".join(f"{l:4d}" for l in lbls)
    print("Confusion Matrix:")
    print(header)

    for i, l in enumerate(lbls):
        row = " ".join(f"{n:4d}" for n in cm[i])
        print(f"{l:4d} {row}")

# Main
if __name__ == "__main__":
    run_ctg("CTG.csv", K_CTG)
    run_faces("yalefaces", K_YALE)
