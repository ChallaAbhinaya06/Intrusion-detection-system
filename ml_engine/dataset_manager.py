import os
import pandas as pd

# Default datasets used ONLY when user uploads nothing
DEFAULT_DATASETS = [
    "default_datasets/CIC_IDS2017.csv",
    "default_datasets/5G_NIDD.csv"
]


def get_active_dataset_paths(uploaded_files=None):
    """
    DFD Level-2: Dataset Selection Decision

    If user uploads datasets → use ONLY those
    If user uploads nothing → use default datasets
    """
    if uploaded_files and len(uploaded_files) > 0:
        return uploaded_files

    return DEFAULT_DATASETS


def load_datasets(dataset_paths):
    """
    Load datasets into memory with metadata
    """
    datasets = []

    for path in dataset_paths:
        df = pd.read_csv(path, low_memory=False)

        datasets.append({
            "dataset_name": os.path.basename(path),
            "dataframe": df,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)
        })

    return datasets
