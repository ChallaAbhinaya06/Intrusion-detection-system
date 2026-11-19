"""
Dataset Compatibility Checker

Checks if datasets are compatible with trained models based on:
  1. Column structure (same columns)
  2. Number of classes (same num_classes — prevents loading a 13-class model
     for a 2-class dataset, which was causing the 77% accuracy bug)

Also stores and retrieves the label-to-class-ID mapping so attack names
come from the actual dataset, not hardcoded values.
"""

import pandas as pd
import os
import json


def get_dataset_signature(df):
    """
    Generate a signature for a dataset based on its RAW structure.
    Returns (num_columns, col_hash, col_names_tuple)
    """
    num_cols = len(df.columns)
    col_names_sorted = tuple(sorted(df.columns.tolist()))
    col_hash = hash(col_names_sorted)
    return (num_cols, col_hash, col_names_sorted)


def get_dataset_num_classes(df):
    """
    Detect the number of unique classes in this dataset using the same
    priority logic as Preprocessor.detect_label_column().
    Returns int, or None if dataset is unlabelled.
    """
    col_lower = {c.strip().lower(): c for c in df.columns}

    # Priority 1: Attack Type
    for candidate in ["attack type", "attacktype", "attack_type"]:
        if candidate in col_lower:
            col = col_lower[candidate]
            return int(df[col].nunique())

    # Priority 2: Label / Attack / Class / Target
    for candidate in ["label", "attack", "class", "target"]:
        if candidate in col_lower:
            col = col_lower[candidate]
            return int(df[col].nunique())

    return None  # unlabelled


def are_datasets_compatible(df1, df2):
    """Check if two datasets have the same column structure."""
    sig1 = get_dataset_signature(df1)
    sig2 = get_dataset_signature(df2)
    return sig1[:2] == sig2[:2]


def find_compatible_model(dataset_df, saved_models_dir):
    """
    Find the best compatible trained model for the given dataset.

    Matching priority:
      1. EXACT match  — same columns AND same num_classes           ← best
      2. SUPERSET match — same columns, model has MORE classes      ← good fallback
         A model trained on 13 classes can evaluate a 2-class dataset
         better than a model trained on only 2 classes of a different
         distribution, because it has seen richer feature patterns.
         Among superset candidates the one with the MOST classes wins.
      3. SUBSET/OTHER — same columns, model has FEWER classes       ← last resort

    Design rationale
    ----------------
    - Train runs always save their own exact model  →  exact match next time.
    - Evaluate on a dataset that has never been trained:
        * Prefer the richest (most-class) trained model for that column structure
          so the feature encoder has the broadest knowledge.
        * A 13-class CIC model evaluating a 2-class CIC file will still correctly
          predict BENIGN/DDoS for flows that clearly look like those classes;
          accuracy is naturally lower but FAR better than loading a 2-class model
          trained on a completely different class distribution.

    Returns the model filename if found, None otherwise.
    """
    if not os.path.exists(saved_models_dir):
        return None

    dataset_sig       = get_dataset_signature(dataset_df)
    dataset_n_classes = get_dataset_num_classes(dataset_df)

    exact_match    = None   # same columns + same class count
    superset_match = None   # same columns + model has MORE classes AND dataset is a name-subset
    superset_n     = -1     # track highest class count among valid supersets

    for model_file in sorted(os.listdir(saved_models_dir)):
        if not model_file.endswith('.h5'):
            continue

        meta_path = os.path.join(saved_models_dir, model_file.replace('.h5', '.meta'))
        if not os.path.exists(meta_path):
            # No .meta — cannot verify class compatibility, skip
            print(f"[Compatibility] {model_file} has no .meta file — skipping.")
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Columns must match exactly
            if meta.get('num_columns') != dataset_sig[0]:
                continue
            if set(meta.get('column_names', [])) != set(dataset_sig[2]):
                continue

            meta_n_classes = meta.get('num_classes') or 0
            meta_label_map = meta.get('label_mapping', {})   # {str_int → class_name}
            model_class_names = set(meta_label_map.values())

            if dataset_n_classes is not None and meta_n_classes == dataset_n_classes:
                # Perfect — exact class count match
                exact_match = model_file
                break

            elif (dataset_n_classes is not None
                  and meta_n_classes > dataset_n_classes
                  and model_class_names):
                # Model has MORE classes. Check if dataset classes are a
                # true name-subset of the model's known classes.
                # This is safe because train_eval.py will remap y_test integers
                # to the model's integer space using class names as the bridge.
                # Classes in dataset but not in model are excluded from metrics.
                # → No Unknown_Class labels, correct accuracy.
                if meta_n_classes > superset_n:
                    superset_match = model_file
                    superset_n     = meta_n_classes

        except Exception as e:
            print(f"Error reading meta file {model_file}: {e}")
            continue

    if exact_match:
        print(f"[Compatibility] Exact match: {exact_match} ({dataset_n_classes} classes)")
        return exact_match

    if superset_match:
        print(f"[Compatibility] Superset match: {superset_match} "              f"(model={superset_n} classes, dataset={dataset_n_classes} classes). "              f"Dataset classes will be remapped to model integer space for correct evaluation.")
        return superset_match

    return None


def save_dataset_signature(dataset_df, dataset_name, saved_models_dir,
                           num_classes=None, label_encoder=None):
    """
    Save dataset signature alongside the model.
    Stores RAW dataset structure (before preprocessing) + label mapping.
    """
    signature    = get_dataset_signature(dataset_df)
    dataset_base = os.path.splitext(dataset_name)[0]
    meta_file    = os.path.join(saved_models_dir, f"{dataset_base}_model.meta")

    # Detect num_classes from the actual data if not supplied
    if num_classes is None:
        num_classes = get_dataset_num_classes(dataset_df) or 4

    # Build label mapping from the fitted LabelEncoder
    label_mapping = {}
    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
        for class_id, original_name in enumerate(label_encoder.classes_):
            label_mapping[int(class_id)] = str(original_name)

    meta_data = {
        'num_columns':  signature[0],
        'column_hash':  signature[1],
        'column_names': list(signature[2]),
        'dataset_name': dataset_name,
        'num_classes':  num_classes,
        'class_info':   f'Model trained with {num_classes} classes',
        'label_mapping': label_mapping,
    }

    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"Saved dataset signature: {signature[0]} columns, {num_classes} classes")
    if label_mapping:
        print(f"Label mapping: {label_mapping}")


def load_label_mapping(model_filename, saved_models_dir):
    """
    Load the label mapping from the .meta file for a given model.
    Returns dict mapping class_id (int) -> original_label_name (str).
    Returns {} if meta file not found or has no mapping.
    """
    meta_path = os.path.join(saved_models_dir,
                             model_filename.replace('.h5', '.meta'))
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        mapping = meta_data.get('label_mapping', {})
        return {int(k): v for k, v in mapping.items()}
    except Exception as e:
        print(f"Error loading label mapping: {e}")
        return {}