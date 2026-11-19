"""
Metrics Aggregation Module

Computes overall performance metrics from dataset results.
Two metrics are tracked:

  accuracy       — exact label match (strict)
  detection_rate — binary attack-vs-normal match (practical)
                   "Was the intrusion caught, regardless of exact attack name?"
"""

import numpy as np


def compute_overall_metrics(dataset_results):
    """
    Computes aggregate stats across all datasets for both
    exact accuracy and binary detection rate.
    """
    accuracies = [
        res["accuracy"]
        for res in dataset_results
        if res.get("accuracy") is not None
    ]
    detection_rates = [
        res["detection_rate"]
        for res in dataset_results
        if res.get("detection_rate") is not None
    ]

    def _stats(values):
        if not values:
            return {"average": 0.0, "max": 0.0, "min": 0.0}
        return {
            "average": round(float(np.mean(values)), 4),
            "max":     round(float(np.max(values)),  4),
            "min":     round(float(np.min(values)),  4),
        }

    acc_stats = _stats(accuracies)
    det_stats = _stats(detection_rates)

    return {
        # Exact label accuracy
        "average_accuracy": acc_stats["average"],
        "max_accuracy":     acc_stats["max"],
        "min_accuracy":     acc_stats["min"],
        # Binary detection rate
        "average_detection_rate": det_stats["average"],
        "max_detection_rate":     det_stats["max"],
        "min_detection_rate":     det_stats["min"],
    }
