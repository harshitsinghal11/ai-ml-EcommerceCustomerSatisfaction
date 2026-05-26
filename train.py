# train.py
# Trains Logistic Regression and Decision Tree models, evaluates them,
# and serialises both to disk using pickle.

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import load_and_preprocess


def evaluate(name, y_test, y_pred):
    """Print accuracy + classification report and plot confusion matrix."""
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{name.lower().replace(' ', '_')}.png", dpi=120)
    plt.show()
    return acc


def train():
    # ------------------------------------------------------------------ #
    # 1. LOAD PREPROCESSED DATA
    # ------------------------------------------------------------------ #
    X, y, selected_features, _ = load_and_preprocess(verbose=True)

    # ------------------------------------------------------------------ #
    # 2. TRAIN / TEST SPLIT
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------ #
    # 3. LOGISTIC REGRESSION  (needs feature scaling)
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    acc_lr = evaluate("Logistic Regression", y_test, y_pred_lr)

    # ------------------------------------------------------------------ #
    # 4. DECISION TREE  (no scaling required)
    # ------------------------------------------------------------------ #
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = evaluate("Decision Tree", y_test, y_pred_dt)

    # ------------------------------------------------------------------ #
    # 5. COMPARISON SUMMARY
    # ------------------------------------------------------------------ #
    print("\n" + "="*50)
    print("  MODEL COMPARISON")
    print("="*50)
    print(f"  Logistic Regression accuracy : {acc_lr:.4f}")
    print(f"  Decision Tree accuracy       : {acc_dt:.4f}")
    winner = "Logistic Regression" if acc_lr >= acc_dt else "Decision Tree"
    print(f"  Best model                   : {winner}")

    # ------------------------------------------------------------------ #
    # 6. SAVE ARTEFACTS WITH PICKLE
    # ------------------------------------------------------------------ #
    with open("lr_model.pkl",  "wb") as f: pickle.dump(lr_model, f)
    with open("dt_model.pkl",  "wb") as f: pickle.dump(dt_model, f)
    with open("scaler.pkl",    "wb") as f: pickle.dump(scaler,   f)
    with open("features.pkl",  "wb") as f: pickle.dump(selected_features, f)

    print("\n[train] Models saved: lr_model.pkl | dt_model.pkl | scaler.pkl | features.pkl")


if __name__ == "__main__":
    train()