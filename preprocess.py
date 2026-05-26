# preprocess.py
# Loads, cleans, encodes, and prepares the dataset for model training.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(filepath="dataset_raw.csv", verbose=True):
    """
    Full preprocessing pipeline.

    Steps:
        1. Load CSV
        2. Handle missing values
        3. Label-encode categorical columns
        4. Drop Customer ID and highly correlated features (|r| > 0.9)
        5. Correlation-based feature selection (top features vs target)

    Returns:
        X  : pd.DataFrame  – selected feature columns
        y  : pd.Series     – encoded target (Satisfaction Level)
        selected_features : list[str]
        label_encoders    : dict  – for inverse-transforming in the app
    """

    # ------------------------------------------------------------------
    # 1. LOAD
    # ------------------------------------------------------------------
    df = pd.read_csv(filepath)
    if verbose:
        print(f"[preprocess] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
        print(df.head(5))

    # ------------------------------------------------------------------
    # 2. MISSING VALUES
    # ------------------------------------------------------------------
    missing = df.isnull().sum()
    if verbose:
        print("\n[preprocess] Missing values per column:")
        print(missing[missing > 0] if missing.any() else "  None")

    # Drop rows with missing target; fill numeric cols with median
    df = df.dropna(subset=["Satisfaction Level"])
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ------------------------------------------------------------------
    # 3. LABEL ENCODING
    # ------------------------------------------------------------------
    label_encoders = {}
    cat_cols = ["Gender", "City", "Membership Type", "Satisfaction Level"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # last fitted; Satisfaction Level kept separately

    # Boolean fix
    if "Discount Applied" in df.columns:
        df["Discount Applied"] = df["Discount Applied"].astype(int)

    if verbose:
        print("\n[preprocess] After encoding:")
        print(df.head(5))

    # ------------------------------------------------------------------
    # 4. DROP UNNECESSARY / HIGHLY CORRELATED COLUMNS
    # ------------------------------------------------------------------
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    to_drop = list(set(to_drop + ["Customer ID"]))
    df = df.drop(columns=to_drop, errors="ignore")
    if verbose:
        print(f"\n[preprocess] Dropped columns: {to_drop}")

    # ------------------------------------------------------------------
    # 5. FEATURE SELECTION (correlation with target)
    # ------------------------------------------------------------------
    target = "Satisfaction Level"
    corr_with_target = (
        df.corr()[target].drop(target).abs().sort_values(ascending=False)
    )
    if verbose:
        print("\n[preprocess] Feature correlations with target:")
        print(corr_with_target)

    # Select top-2 features (matching both members' notebooks)
    selected_features = list(corr_with_target.index[:2])
    if verbose:
        print(f"\n[preprocess] Selected features: {selected_features}")

    X = df[selected_features]
    y = df[target]

    return X, y, selected_features, label_encoders


if __name__ == "__main__":
    X, y, feats, _ = load_and_preprocess()
    print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
    print(f"Features used: {feats}")