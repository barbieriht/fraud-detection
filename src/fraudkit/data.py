"""Loading and splitting helpers for the portfolio datasets."""

import warnings
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


IEEE_STRING_COLS = (
    {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"}
    | {f"M{i}" for i in range(1, 10)}
    | {f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)}
)


IEEE_INT_COLS = {"TransactionID": "int32", "isFraud": "int8", "TransactionDT": "int32"}


def _csv_to_parquet(csv_path, parquet_path, string_cols):
    """Stream a wide CSV into parquet batch by batch: float32 numerics, strings kept as text.

    Never holds the whole CSV in memory, which matters for IEEE-CIS (650 MB, 394 columns)
    on a small machine.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    types = {c: pa.string() if c in string_cols else pa.float32() for c in header}
    types.update({c: getattr(pa, t)() for c, t in IEEE_INT_COLS.items() if c in types})
    convert = pacsv.ConvertOptions(column_types=types, strings_can_be_null=True)
    reader = pacsv.open_csv(csv_path, convert_options=convert,
                            read_options=pacsv.ReadOptions(block_size=64 << 20))
    with pq.ParquetWriter(parquet_path, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)


def prepare_ieee():
    """Download IEEE-CIS training data (competition rules must be accepted) and convert to parquet."""
    raw = RAW / "ieee"
    out = {name: raw / f"{name}.parquet" for name in ("train_transaction", "train_identity")}
    if all(p.exists() for p in out.values()):
        return out
    raw.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    for name, parquet in out.items():
        csv = raw / f"{name}.csv"
        if not csv.exists():
            api.competition_download_file("ieee-fraud-detection", f"{name}.csv", path=raw)
            import zipfile

            with zipfile.ZipFile(raw / f"{name}.csv.zip") as z:
                z.extractall(raw)
            (raw / f"{name}.csv.zip").unlink()
        with open(csv) as f:
            header = f.readline().strip().split(",")
        strings = {c for c in header if c in IEEE_STRING_COLS}
        _csv_to_parquet(csv, parquet, strings)
    return out


def load_ieee(columns=None, identity=True):
    """IEEE-CIS training data with identity columns left-joined, sorted by time.

    Text columns come back as pandas categoricals. Pass `columns` to load only what a
    step needs; with all ~430 columns the frame takes about 1.2 GB.
    """
    import pyarrow.parquet as pq

    paths = prepare_ieee()

    def read(path, cols):
        if cols is not None:
            available = set(pq.read_schema(path).names)
            keys = ["TransactionID", "TransactionDT", "isFraud"]   # always needed
            cols = [c for c in dict.fromkeys([*keys, *cols]) if c in available]
        # self_destruct frees each Arrow column as soon as it is converted, so the
        # table and the DataFrame never both sit in memory in full.
        table = pq.read_table(path, columns=cols)
        df = table.to_pandas(self_destruct=True, split_blocks=True)
        del table
        for c in df.columns.intersection(list(IEEE_STRING_COLS)):
            df[c] = df[c].astype("category")
        return df

    df = read(paths["train_transaction"], columns)
    if identity:
        ident = read(paths["train_identity"], columns)
        if ident.shape[1] > 1:
            # Attach identity columns by lookup instead of merge: merge would copy the
            # whole (~1 GB) transaction table.
            ident = ident.set_index("TransactionID").reindex(df.TransactionID.values)
            with warnings.catch_warnings():   # adding columns one by one fragments the frame; fine here
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                for c in ident.columns:
                    df[c] = ident[c].values
            del ident
    if not df.TransactionDT.is_monotonic_increasing:
        df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
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


BAF_STRING_COLS = {"payment_type", "employment_status", "housing_status", "source", "device_os"}


def load_baf(variant="Base"):
    """Bank Account Fraud (Feedzai, NeurIPS 2022): 1M account-opening applications over 8 months.

    Numbers come back as float32 (labels and flags as small integers), text as categoricals.
    `variant` is "Base" or "Variant I" … "Variant V".
    """
    import pyarrow.parquet as pq

    raw = RAW / "baf"
    csv, parquet = raw / f"{variant}.csv", raw / f"{variant.replace(' ', '_')}.parquet"
    if not parquet.exists():
        if not csv.exists():
            raw.mkdir(parents=True, exist_ok=True)
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_file("sgpjesus/bank-account-fraud-dataset-neurips-2022",
                                      f"{variant}.csv", path=raw)
            import zipfile

            for z in raw.glob("*.zip"):
                with zipfile.ZipFile(z) as f:
                    f.extractall(raw)
                z.unlink()
        _csv_to_parquet(csv, parquet, BAF_STRING_COLS)
    table = pq.read_table(parquet)
    df = table.to_pandas(self_destruct=True)
    del table
    for c in BAF_STRING_COLS:
        df[c] = df[c].astype("category")
    small_ints = ["fraud_bool", "month", "email_is_free", "phone_home_valid", "phone_mobile_valid",
                  "has_other_cards", "foreign_request", "keep_alive_session", "device_fraud_count"]
    df[small_ints] = df[small_ints].astype("int8")
    return df
