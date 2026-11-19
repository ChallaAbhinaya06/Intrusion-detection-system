"""
Preprocessing Module

DFD Level-2: Data Cleaning & Transformation

Pipeline (in order):
    1. Detect & separate label column (returns is_labelled=False when absent)
    2. Split features into numeric and categorical
    3. Replace inf -> NaN  (numeric only)
    4. Fill NaN  -> median (numeric) / mode (categorical)
    5. LabelEncode every categorical feature column -> integer
    6. StandardScale all features together
    7. Reshape to 3-D for CNN  (samples, features, 1)
    8. Stratified train/test split  (stratify skipped when unlabelled)
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# Priority-ordered label column detection.
# The FIRST list that matches wins — Attack Type beats plain Label.
# All names are matched case-insensitively after stripping whitespace.
_LABEL_PRIORITY = [
    # Tier 1 — specific attack classification (preferred)
    ["attack type", "attacktype", "attack_type"],
    # Tier 2 — generic label columns (fallback)
    ["label", "attack", "class", "target"],
]

# Any column whose stripped-lower name is in this set is a label-related
# column that must be excluded from features entirely.
_LABEL_RELATED = {
    "label", "attack", "class", "target",
    "attack type", "attacktype", "attack_type",
    "attack tool", "attacktool", "attack_tool",
}


class Preprocessor:

    @staticmethod
    def detect_label_column(df):
        """
        Return the best label column using priority order:
          1. 'Attack Type' (or variants) — used when present
          2. 'Label' / 'Attack' / 'Class' / 'Target' — fallback

        Also returns a list of ALL label-related columns found so the
        caller can drop them all from the feature matrix.

        Returns
        -------
        (label_col, cols_to_drop)
            label_col   : str | None  — the chosen label column
            cols_to_drop: list[str]   — all label-related columns to exclude
        """
        # Map stripped-lower -> original column name
        col_map = {col.strip().lower(): col for col in df.columns}

        # Find ALL label-related columns present in this dataset
        cols_to_drop = [
            original
            for lower, original in col_map.items()
            if lower in _LABEL_RELATED
        ]

        # Walk the priority tiers — first tier with a match wins
        for tier in _LABEL_PRIORITY:
            for candidate in tier:
                if candidate in col_map:
                    return col_map[candidate], cols_to_drop

        return None, cols_to_drop  # unlabelled dataset

    @staticmethod
    def preprocess(df):
        """
        Returns
        -------
        X_train, X_test, y_train, y_test, is_labelled, label_encoder
            y_train / y_test are None when the dataset has no label column.
            is_labelled is a bool the caller can branch on.
            label_encoder is the LabelEncoder fitted on the label column (None if unlabelled)
                         Contains .classes_ attribute with original label names
        """
        if df is None or df.empty:
            raise ValueError("Empty dataset received")

        # Detect label — returns (chosen_col, all_label_cols_to_drop)
        label_col, cols_to_drop = Preprocessor.detect_label_column(df)
        is_labelled = label_col is not None

        # Separate X and y — drop ALL label-related columns from features
        if is_labelled:
            y_series = df[label_col].copy()

            # 5G-NIDD issue: 'Attack Type' column has NaN for benign rows
            # (benign traffic has no attack type).  Fill NaN from the 'Label'
            # column if present, otherwise substitute "Normal".
            if y_series.isna().any():
                col_map_lower = {c.strip().lower(): c for c in df.columns}

                # Try to find a fallback label column (e.g. 'Label')
                fallback_col = None
                for candidate in ["label", "class", "target"]:
                    if candidate in col_map_lower and col_map_lower[candidate] != label_col:
                        fallback_col = col_map_lower[candidate]
                        break

                if fallback_col is not None:
                    # Use the fallback column where Attack Type is NaN
                    y_series = y_series.where(y_series.notna(), df[fallback_col])
                else:
                    # No fallback column — substitute "Normal" for NaN rows
                    y_series = y_series.fillna("Normal")

            y_raw = y_series.astype(str)
        else:
            y_raw = None

        # Drop every label-related column so none leaks into features
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()

        if X.shape[1] == 0:
            raise ValueError("No feature columns found after removing label")

        # Split into numeric and categorical feature columns
        numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Clean numeric columns  -  inf -> NaN -> median
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Clean & encode categorical columns  -  NaN -> mode -> LabelEncode
        if categorical_cols:
            for col in categorical_cols:
                # fill missing with the most frequent value
                mode_val = X[col].mode()
                if not mode_val.empty:
                    X[col] = X[col].fillna(mode_val[0])
                # encode: "tcp" -> 0, "udp" -> 1, ...
                le_feat = LabelEncoder()
                X[col]  = le_feat.fit_transform(X[col].astype(str))

        # Encode target labels  (only when labelled)
        le_target = None          # will hold the label encoder if labelled
        if is_labelled:
            le_target = LabelEncoder()
            y         = le_target.fit_transform(y_raw)
        else:
            y = None

        # Scale ALL features together  (now all numeric)
        X_np = X.values.astype(np.float32)

        if X_np.shape[0] == 0:
            raise ValueError("Dataset became empty after cleaning")

        scaler = StandardScaler()
        X_np   = scaler.fit_transform(X_np)

        # Reshape  ->  (samples, features, 1)  for CNN input
        X_np = np.expand_dims(X_np, axis=2)

        # Train / test split
        if is_labelled:
            stratify = y if len(np.unique(y)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y,
                test_size=0.2,
                random_state=42,
                stratify=stratify
            )
        else:
            # No labels -> no stratification, y parts are None
            X_train, X_test = train_test_split(
                X_np,
                test_size=0.2,
                random_state=42
            )
            y_train = None
            y_test  = None

        # Return 6 values: X_train, X_test, y_train, y_test, is_labelled, label_encoder
        return X_train, X_test, y_train, y_test, is_labelled, le_target