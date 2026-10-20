"""Loading and splitting helpers for the portfolio datasets."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

ULB_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def download_kaggle(slug, dest):
    """Download and unzip a Kaggle dataset unless `dest` already holds files."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        return dest
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=dest, unzip=True)
    return dest


def load_ulb(drop_duplicates=True):
    """ULB credit card data, sorted by time, with hour-of-day and log-amount added.

    Duplicated rows are dropped by default: identical rows on both sides of a
    train/test split would leak and inflate test scores.
    """
    path = download_kaggle("mlg-ulb/creditcardfraud", RAW / "ulb") / "creditcard.csv"
    df = pd.read_csv(path)
    if drop_duplicates:
        df = df.drop_duplicates()
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["Hour"] = (df["Time"] % 86_400) / 3_600
    df["LogAmount"] = np.log1p(df["Amount"])
    return df


def time_split(df, fractions=(0.6, 0.2, 0.2), time_col="Time"):
    """Split a frame into consecutive, non-overlapping time blocks (train, val, test).

    Mimics production: the model only ever sees the past when scoring the future.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    edges = np.cumsum([0, *fractions])
    cuts = (edges * len(df)).round().astype(int)
    return tuple(df.iloc[a:b].reset_index(drop=True) for a, b in pairwise(cuts))
