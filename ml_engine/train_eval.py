"""
Training & Evaluation Module

Handles dataset-wise model training and evaluation.
Supports both labelled and unlabelled datasets.
"""

import os
import warnings
import numpy as np
from django.conf import settings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

MODEL_DIR = os.path.join(settings.MEDIA_ROOT, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Disable GUI backend for matplotlib (OS-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras import backend as K

tf.get_logger().setLevel('ERROR')

from ml_engine.model import build_cnn_lstm_attention_model
from ml_engine.preprocess import Preprocessor
from ml_engine.dataset_compatibility import (
    find_compatible_model,
    save_dataset_signature,
    load_label_mapping
)
from ml_engine.attack_knowledge_base import set_label_mapping, get_attack_info

# Reduce CPU pressure
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

RESULTS_DIR = os.path.join(settings.MEDIA_ROOT, "results")


def train_and_evaluate(
    datasets,
    num_classes,
    use_pretrained=False,
    progress_callback=None,
    user=None
):
    """
    Train and/or evaluate model for each dataset.

    use_pretrained=True  -> load saved model, skip training
    Unlabelled datasets  -> predictions only, no accuracy / classification report
    """
    all_results = []

    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        df           = dataset["dataframe"]

        print(f"\nProcessing dataset: {dataset_name}")

        dataset_base       = os.path.splitext(dataset_name)[0]
        dataset_result_dir = os.path.join(RESULTS_DIR, dataset_base)
        os.makedirs(dataset_result_dir, exist_ok=True)

        # PREPROCESSING 
        X_train, X_test, y_train, y_test, is_labelled, label_encoder = Preprocessor.preprocess(df)

        # Derive num_classes from the actual encoded labels — this is the
        # authoritative count and must match what the model output layer uses.
        # The passed-in num_classes is ignored to avoid any off-by-one mismatch.
        if is_labelled and label_encoder is not None:
            num_classes = len(label_encoder.classes_)
        elif is_labelled and y_train is not None:
            num_classes = int(np.max(y_train)) + 1   # fallback if no encoder
        # else: unlabelled — num_classes unused

        model_path    = os.path.join(MODEL_DIR, f"{dataset_base}_model.h5")
        label_mapping = None   # populated in pretrained path from .meta file

        # MODEL
        # y_test_remapped: y_test re-encoded to match the loaded model's integer space.
        # Needed when a superset model is used (e.g. 13-class model on 3-class dataset).
        y_test_remapped = None

        if use_pretrained:
            compatible_model = find_compatible_model(df, MODEL_DIR)
            if compatible_model:
                model_path = os.path.join(MODEL_DIR, compatible_model)
                print(f"Found compatible model: {compatible_model}")
                model = tf.keras.models.load_model(model_path)

                # Load label mapping from .meta
                model_basename = os.path.basename(model_path)
                label_mapping = load_label_mapping(model_basename, MODEL_DIR)

                if not label_mapping and label_encoder is not None:
                    # No .meta (manually placed model) — use dataset encoder as fallback
                    label_mapping = {int(i): str(n) for i, n in enumerate(label_encoder.classes_)}
                    print(f"[Label mapping] No .meta — built from LabelEncoder: {label_mapping}")

                if label_mapping:
                    set_label_mapping(label_mapping)

                # ── Subset evaluation: remap y_test to the model's integer space ──
                # The model was trained with its own LabelEncoder (e.g. 13 classes).
                # The current dataset has its own LabelEncoder (e.g. 3 classes).
                # These produce DIFFERENT integer IDs for the same class name.
                #
                # Example — 13-class model:  BENIGN=0, DDoS=1, Infiltration=5, ...
                #            3-class dataset: BENIGN=0, DDoS=1, Infiltration=2
                #
                # Without remapping: y_test=2 (Infiltration) vs y_pred=5 → WRONG
                # With remapping:    y_test=5 (Infiltration) vs y_pred=5 → CORRECT
                #
                # Classes in the dataset that don't exist in the model are excluded.
                if (is_labelled and label_encoder is not None and label_mapping
                        and model.layers[-1].units != num_classes):

                    # Build name→model_int reverse map
                    model_name_to_int = {str(v): int(k) for k, v in label_mapping.items()}

                    # Map each dataset class name → model integer (None if not in model)
                    dataset_classes = label_encoder.classes_   # e.g. ['BENIGN','DDoS','Infiltration']
                    dataset_to_model = {}
                    missing = []
                    for dataset_int, class_name in enumerate(dataset_classes):
                        if class_name in model_name_to_int:
                            dataset_to_model[dataset_int] = model_name_to_int[class_name]
                        else:
                            missing.append(class_name)
                            dataset_to_model[dataset_int] = None

                    if missing:
                        print(f"[Subset eval] Classes in dataset but NOT in model: {missing} — these rows excluded from metrics.")

                    # Build remapped y_test; mask out rows whose class isn't in model
                    mapped_vals = np.array([dataset_to_model[int(t)] for t in y_test], dtype=object)
                    valid_mask  = np.array([v is not None for v in mapped_vals])
                    y_test_remapped = mapped_vals[valid_mask].astype(int)
                    print(f"[Subset eval] Remapped {valid_mask.sum()}/{len(y_test)} rows to model integer space.")
                    print(f"[Subset eval] Mapping: { {dataset_classes[k]: v for k, v in dataset_to_model.items() if v is not None} }")

            else:
                raise RuntimeError(
                    f"No compatible pretrained model found for this dataset structure. "
                    f"The dataset has {df.shape[1]} columns. "
                    f"Please train a model with this dataset structure first."
                )
        else:
            if not is_labelled:
                raise RuntimeError(
                    "Cannot train on an unlabelled dataset. "
                    "Please provide a dataset with a label column "
                    "(Label / Attack / Class / Target)."
                )
            print("Training new model...")
            model = build_cnn_lstm_attention_model(
                input_shape=X_train.shape[1:],
                num_classes=num_classes
            )

        # CALLBACKS 
        callbacks = [
            # Stop early if validation loss stops improving
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when stuck
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        if progress_callback and not use_pretrained:
            progress_callback(dataset_base, "training_started")

            def on_epoch_end(epoch, logs):
                progress_callback(dataset_base, f"epoch {epoch + 1}/10")

            callbacks.append(
                tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
            )

        # TRAINING 
        if not use_pretrained:
            # Use balanced class weights to handle imbalanced datasets
            # (e.g. many Benign rows vs few attack rows or vice versa)
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = None
            if is_labelled and y_train is not None:
                unique_classes = np.unique(y_train)
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_classes,
                    y=y_train
                )
                class_weights = dict(zip(unique_classes.tolist(), weights.tolist()))
                print(f"Class weights: { {label_encoder.classes_[k]: round(v, 3) for k, v in class_weights.items()} }")

            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weights,
                verbose=1,
                callbacks=callbacks
            )
            model.save(model_path)
            save_dataset_signature(df, dataset_name, MODEL_DIR, num_classes, label_encoder)
            # Set the label mapping from the encoder we just used
            if label_encoder is not None:
                label_mapping_train = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}
                set_label_mapping(label_mapping_train)
            print("Saved model and dataset signature")
            plot_training_curves(history, dataset_result_dir)

        # PREDICTION
        y_pred_full = np.argmax(model.predict(X_test), axis=1)

        # ── Apply remapping mask if subset evaluation is active ──────────────
        if y_test_remapped is not None:
            # Re-compute the valid mask (rows whose class IS in model)
            model_name_to_int_local = {str(v): int(k) for k, v in label_mapping.items()}
            dataset_classes_local   = label_encoder.classes_
            valid_mask_local = np.array([
                dataset_classes_local[int(t)] in model_name_to_int_local
                for t in y_test
            ])
            y_pred = y_pred_full[valid_mask_local]
            y_test_eval = y_test_remapped         # remapped to model's integer space
        else:
            y_pred      = y_pred_full
            y_test_eval = y_test                  # same encoding as model

        # METRICS (labelled only)
        acc            = None
        report         = None
        detection_rate = None   # binary: was the attack detected (regardless of exact name)?

        if is_labelled:
            acc    = accuracy_score(y_test_eval, y_pred)
            report = classification_report(y_test_eval, y_pred, output_dict=True, zero_division=0)

            # ── Detection Rate (binary attack vs normal) ─────────────────
            # Resolves the "77% accuracy" paradox:
            # A 13-class model predicting "DoS GoldenEye" for a "DDoS" row
            # counts as WRONG for exact accuracy but RIGHT for detection rate
            # because it correctly identified an ATTACK (not BENIGN).
            #
            # BENIGN keywords: label contains "benign" or "normal"
            _BENIGN_KW = {"benign", "normal"}

            def _is_benign_name(name):
                return any(kw in str(name).lower() for kw in _BENIGN_KW)

            # Resolve integer IDs → label names using the model's label mapping
            # For subset eval, y_test_eval is already in model integer space
            _active_map = label_mapping if label_mapping else {}
            if not _active_map and label_encoder is not None:
                _active_map = {i: str(n) for i, n in enumerate(label_encoder.classes_)}

            if _active_map:
                actual_is_benign = np.array([
                    _is_benign_name(_active_map.get(int(t), str(t)))
                    for t in y_test_eval
                ])
                pred_is_benign = np.array([
                    _is_benign_name(_active_map.get(int(p), str(p)))
                    for p in y_pred
                ])
                # Detection: actual attack (not benign) AND predicted as attack (not benign)
                actual_attacks  = ~actual_is_benign
                detected        = actual_attacks & ~pred_is_benign
                if actual_attacks.sum() > 0:
                    detection_rate = round(float(detected.sum() / actual_attacks.sum()), 4)
                else:
                    detection_rate = None   # no attack rows — detection rate not applicable

        # DATASET SIZE 
        dataset_size    = df.shape[0]
        dataset_size_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

        # SAVE TEST-RESULTS CSV 
        import pandas as pd

        # Resolve label names directly from the active mapping (never Unknown_Class_X)
        # Priority: label_mapping from .meta → label_encoder from this run → get_attack_info fallback
        _name_map = {}
        if label_mapping:
            _name_map = {int(k): str(v) for k, v in label_mapping.items()}
        elif label_encoder is not None:
            _name_map = {int(i): str(n) for i, n in enumerate(label_encoder.classes_)}

        def _resolve_name(lbl_id):
            return _name_map.get(int(lbl_id), get_attack_info(int(lbl_id))['attack_type'])

        predicted_names = [_resolve_name(lbl) for lbl in y_pred]

        if is_labelled:
            actual_names = [_resolve_name(lbl) for lbl in y_test_eval]
            results_df   = pd.DataFrame({
                'Actual_Label_ID':     y_test_eval,
                'Actual_Label_Name':   actual_names,
                'Predicted_Label_ID':  y_pred,
                'Predicted_Label_Name': predicted_names,
                'Match':               y_test_eval == y_pred,
            
            })
        else:
            # Unlabelled – only predictions, no actuals
            results_df = pd.DataFrame({
                'Sample_Index':        range(len(y_pred)),
                'Predicted_Label_ID':  y_pred,
                'Predicted_Label_Name': predicted_names
            })

        results_csv_path = os.path.join(dataset_result_dir, f"test_results_{dataset_base}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved test results to: {results_csv_path}")

        # PREDICTION-ANALYSIS CHART 
        # Generated for every evaluation run (labelled or not)
        if use_pretrained:
            plt.figure(figsize=(10, 6))
            unique, counts = np.unique(y_pred, return_counts=True)

            plt.subplot(1, 2, 1)
            plt.bar(unique, counts, color='#2563eb', alpha=0.7)
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')
            plt.title('Prediction Distribution')
            plt.grid(axis='y', alpha=0.3)

            plt.subplot(1, 2, 2)
            if is_labelled:
                class_accuracies = []
                class_labels     = []
                for class_id in unique:
                    mask = y_test == class_id
                    if mask.sum() > 0:
                        class_accuracies.append(
                            (y_pred[mask] == y_test[mask]).sum() / mask.sum() * 100
                        )
                        class_labels.append(f'Class {class_id}')
                plt.bar(class_labels, class_accuracies, color='#10b981', alpha=0.7)
                plt.ylabel('Accuracy (%)')
                plt.title('Accuracy by Class')
                plt.ylim(0, 100)
            else:
                # No ground truth – show prediction distribution again as %
                pcts = counts / counts.sum() * 100
                plt.bar([f'Class {c}' for c in unique], pcts, color='#10b981', alpha=0.7)
                plt.ylabel('Percentage (%)')
                plt.title('Prediction Share')
                plt.ylim(0, 100)

            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_result_dir, "prediction_analysis.png"))
            plt.close("all")

        # LOG PREDICTIONS (labelled only) 
        if is_labelled:
            from ml_engine.intrusion_logs import log_predictions
            run_type = 'test' if use_pretrained else 'train'

            # Build the current label mapping to pass explicitly.
            # PRIORITY: use label_mapping from .meta (has ALL model classes, e.g. 14)
            # FALLBACK: use label_encoder from this dataset (may have fewer classes).
            # This prevents Unknown_Class_X on the dashboard when a superset model
            # predicts class IDs that aren't in the small dataset's own encoder.
            current_label_mapping = None
            if label_mapping:
                # .meta mapping has every class the model knows — use it always
                current_label_mapping = label_mapping
            elif label_encoder is not None:
                current_label_mapping = {int(i): str(name) for i, name in enumerate(label_encoder.classes_)}

            log_predictions(
                dataset_name=dataset_name,
                dataset_size=dataset_size,
                y_true=y_test_eval,
                y_pred=y_pred,
                user=user,
                run_type=run_type,
                accuracy=acc,
                label_mapping=current_label_mapping
            )

        # CONFUSION MATRIX (labelled only) 
        if is_labelled:
            cm = confusion_matrix(y_test_eval, y_pred)
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.title(f"Confusion Matrix - {dataset_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_result_dir, "confusion_matrix.png"))
            plt.close("all")

        # PLOT PATHS 
        accuracy_plot           = None
        loss_plot               = None
        confusion_matrix_plot   = None
        prediction_analysis_plot = None

        if not use_pretrained:
            accuracy_plot = f"/media/results/{dataset_base}/accuracy_curve.png"
            loss_plot     = f"/media/results/{dataset_base}/loss_curve.png"

        if is_labelled:
            confusion_matrix_plot = f"/media/results/{dataset_base}/confusion_matrix.png"

        if use_pretrained:
            prediction_analysis_plot = f"/media/results/{dataset_base}/prediction_analysis.png"

        # SUMMARY 
        all_results.append({
            "dataset":                  dataset_base,
            "accuracy":                 acc,
            "detection_rate":           detection_rate,
            "is_labelled":              is_labelled,
            "rows":                     dataset_size,
            "columns":                  df.shape[1],
            "size_mb":                  dataset_size_mb,
            "accuracy_plot":            accuracy_plot,
            "loss_plot":                loss_plot,
            "confusion_matrix_plot":    confusion_matrix_plot,
            "prediction_analysis_plot": prediction_analysis_plot,
            "classification_report":    report,
            "results_csv":              f"/media/results/{dataset_base}/test_results_{dataset_base}.csv"
        })

        K.clear_session()

    return all_results


def plot_training_curves(history, output_dir):
    """Plot Accuracy & Loss curves (non-GUI backend)."""
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"))
    plt.close("all")

    # Loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close("all")