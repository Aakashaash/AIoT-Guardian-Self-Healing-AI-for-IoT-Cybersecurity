#!/usr/bin/env python3
"""
AIoT-Guardian :: data_pipeline.py
=================================
Stages 1-5:
  1. Load & combine BETH, BoT-IoT, Edge-IIoTset
  2. Stratified 70/15/15 split (leakage-resistant)
  3. Preprocess: KNN imputation + One-Hot + Z-score
  4. SMOTE (train only)
  5. mRMR feature selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATASETS,
    IDENTIFIER_COLUMNS,
    MRMR_TOP_K,
    RANDOM_STATE,
    SOURCE_COL,
    TARGET_COL,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    canonical_name,
    convert_target_to_binary,
    find_target_column,
    is_target_like_column,
)


# ======================================================================
# 1. LOAD + COMBINE
# ======================================================================
def load_one_dataset(dataset_name: str, path) -> pd.DataFrame:
    """Read one CSV, binarise its label, drop leakage-prone columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"{path} is empty.")

    df.columns = [str(c).strip() for c in df.columns]
    if pd.Index(df.columns).duplicated().any():
        raise ValueError(f"{path} has duplicate column names.")

    target_col = find_target_column(df, dataset_name)
    y = convert_target_to_binary(df[target_col], dataset_name)

    drop_cols = [
        c for c in df.columns
        if is_target_like_column(c) or canonical_name(c) in IDENTIFIER_COLUMNS
    ]

    X = df.drop(columns=drop_cols, errors="ignore").copy()
    X[SOURCE_COL] = dataset_name
    X[TARGET_COL] = y.to_numpy()

    print(f"Loaded {dataset_name}: {len(X):,} rows, {X.shape[1] - 2} features")
    return X


def load_and_combine_datasets() -> pd.DataFrame:
    """Load every dataset, concatenate, drop exact duplicates."""
    frames = [load_one_dataset(n, p) for n, p in DATASETS.items()]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(combined):,} exact duplicates.")

    combined[TARGET_COL] = combined[TARGET_COL].astype(int)

    if combined[TARGET_COL].nunique() != 2:
        counts = combined[TARGET_COL].value_counts().to_dict()
        raise ValueError(f"Need both classes. Counts: {counts}")

    return combined


# ======================================================================
# 2. SPLIT
# ======================================================================
def split_dataset(combined: pd.DataFrame):
    """
    Leakage-resistant stratified 70/15/15 split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    y = combined[TARGET_COL].astype(int)
    X = combined.drop(columns=[TARGET_COL, SOURCE_COL], errors="ignore")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_share_of_remainder = VAL_FRACTION / (TRAIN_FRACTION + VAL_FRACTION)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_share_of_remainder,
        random_state=RANDOM_STATE,
        stratify=y_tv,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ======================================================================
# 3. PREPROCESSING
# ======================================================================
def make_ohe() -> OneHotEncoder:
    """OneHotEncoder compatible with old and new scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                       # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def determine_feature_types(X_train, X_val, X_test):
    """
    Infer numeric vs categorical columns from the training split, then
    coerce all three splits in place.
    """
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    for frame in (X_train, X_val, X_test):
        for c in numeric_cols:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        for c in categorical_cols:
            frame[c] = frame[c].astype("object")

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """ColumnTransformer: KNN impute + z-score | mode impute + one-hot."""
    numeric_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),          # z-score
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_ohe()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ======================================================================
# 4. SMOTE
# ======================================================================
def apply_smote(X_train, y_train):
    """Oversample the minority class with SMOTE (train split only)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("pip install imbalanced-learn") from exc

    counts = pd.Series(np.asarray(y_train)).value_counts()
    if len(counts) != 2:
        raise ValueError(f"Need both classes in train. Found: {counts.to_dict()}")

    k = min(5, int(counts.min()) - 1)
    if k < 1:
        raise ValueError("SMOTE needs >= 2 minority samples.")

    print(f"Train before SMOTE: {counts.sort_index().to_dict()}")

    X_res, y_res = SMOTE(
        random_state=RANDOM_STATE, k_neighbors=k
    ).fit_resample(X_train, y_train)

    print(f"Train after  SMOTE: {pd.Series(y_res).value_counts().sort_index().to_dict()}")
    return X_res, np.asarray(y_res)


# ======================================================================
# 5. mRMR
# ======================================================================
def mrmr_select(X, y, k: int = MRMR_TOP_K) -> np.ndarray:
    """
    Greedy mRMR: relevance (MI with label) minus mean redundancy
    (|correlation| with already-selected features).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    n_features = X.shape[1]
    k = min(k, n_features)

    relevance = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)

    selected = [int(np.argmax(relevance))]
    remaining = set(range(n_features)) - set(selected)

    while len(selected) < k and remaining:
        best_gain, best_idx = -np.inf, None
        for j in remaining:
            redundancy = np.mean([abs(corr[j, s]) for s in selected])
            gain = relevance[j] - redundancy
            if gain > best_gain:
                best_gain, best_idx = gain, j
        selected.append(best_idx)
        remaining.discard(best_idx)

    print(f"mRMR selected {len(selected)} / {n_features} features.")
    return np.array(selected, dtype=int)