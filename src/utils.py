"""
utils.py: Shared utility functions for SepsisRL.
"""
import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility (Python, NumPy, and PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def dataframe_from_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a full CSV/CSV.GZ into a DataFrame."""
    compression = "gzip" if path.endswith(".gz") else None
    return pd.read_csv(path, compression=compression, **kwargs)


def chunked_read_csv(
    path: str,
    chunk_size: int,
    filter_col: str | None = None,
    filter_values: set | None = None,
    usecols: list | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame:
    """
    Read a large CSV in chunks, optionally filtering rows and columns.

    Parameters
    ----------
    path : str
        Path to CSV or CSV.GZ file.
    chunk_size : int
        Number of rows per chunk.
    filter_col : str, optional
        Column name to filter on.
    filter_values : set, optional
        Accepted values for filter_col.
    usecols : list, optional
        Columns to read.
    dtype : dict, optional
        Column dtypes.

    Returns
    -------
    pd.DataFrame
        Concatenated, filtered DataFrame.
    """
    compression = "gzip" if path.endswith(".gz") else None
    chunks = []
    reader = pd.read_csv(
        path,
        compression=compression,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
    )
    for chunk in reader:
        if filter_col is not None and filter_values is not None:
            chunk = chunk[chunk[filter_col].isin(filter_values)]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        cols = usecols if usecols else []
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


def merge_on_hadm(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Merge two DataFrames on hadm_id."""
    return pd.merge(left, right, on="hadm_id", how="inner", **kwargs)


def merge_on_stay(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Merge two DataFrames on stay_id."""
    return pd.merge(left, right, on="stay_id", how="inner", **kwargs)
