"""Feature engineering for the IEEE-CIS dataset (project 2).

Every function that looks at other transactions is *causal*: a row only sees rows
that happened before it (or, for label features, labels that were already known).
`transductive_uid_features` is the exception, kept to demonstrate leakage.
"""

import warnings

import numpy as np
import pandas as pd

DAY = 86_400
# D columns that count days since some event; subtracting them from the current day
# turns a value that grows with time into a fixed date (e.g. "card first used on day X").
D_DAYS_SINCE = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15"]


def add_time_features(df):
    df["day"] = df.TransactionDT / DAY
    df["hour"] = (df.TransactionDT % DAY) / 3600
    df["weekday"] = (df.TransactionDT // DAY) % 7
    return df


def add_amount_features(df):
    amt = df.TransactionAmt
    df["amt_log"] = np.log1p(amt)
    df["amt_cents"] = (amt - np.floor(amt)).round(3)
    # Amounts with more than two decimals look like currency conversions.
    df["amt_odd_decimals"] = ((amt * 100).round(4) % 1 != 0).astype("int8")
    return df


def normalize_d_columns(df, drop_original=True, keep_raw=()):
    """Replace drifting 'days since X' columns with the fixed day on which X happened.

    Columns in `keep_raw` are left alone: for "days since the previous transaction" (D3),
    day − D is just a recent date, i.e. a calendar feature (notebooks 01 and 03).
    """
    day = np.floor(df.TransactionDT / DAY)
    cols = [c for c in D_DAYS_SINCE if c in df and c not in keep_raw]
    for c in cols:
        df[f"{c}n"] = (day - df[c]).astype("float32")
    if drop_original:
        df = df.drop(columns=cols)
    return df


def add_uid(df):
    """Proxy user ID: card1 + billing address + the day the card was first used (day − D1).

    Rows missing any part get uid −1 ("unknown user"): pooling them would create
    giant fake users with tens of thousands of transactions.
    """
    d1n = np.floor(df.TransactionDT / DAY) - df["D1"] if "D1" in df else df["D1n"]
    parts = pd.DataFrame({"card1": df.card1, "addr1": df.addr1, "d1n": d1n})
    known = parts.notna().all(axis=1)
    key = pd.Series(pd.NA, index=df.index, dtype="string")
    key[known] = parts[known].astype("int64").astype(str).agg("_".join, axis=1)
    df["uid"] = pd.factorize(key, use_na_sentinel=True)[0].astype("int32")
    return df


def causal_uid_features(df):
    """Aggregates over each user's *past* transactions only. Expects rows sorted by time."""
    g = df.groupby("uid", sort=False)
    out = pd.DataFrame(index=df.index)
    out["uid_n_prev"] = g.cumcount().astype("float32")
    out["uid_secs_since_prev"] = g.TransactionDT.diff().astype("float32")

    amt = df.TransactionAmt.astype("float64")
    csum = amt.groupby(df.uid).cumsum() - amt
    csq = (amt**2).groupby(df.uid).cumsum() - amt**2
    n = out.uid_n_prev.replace(0, np.nan)
    mean = csum / n
    out["uid_amt_mean_prev"] = mean.astype("float32")
    out["uid_amt_std_prev"] = np.sqrt((csq / n - mean**2).clip(lower=0)).astype("float32")
    out["amt_vs_uid_mean"] = (amt / mean).astype("float32")

    # Transactions by the same user in the previous 24 hours and 7 days.
    order = np.lexsort((df.TransactionDT.values, df.uid.values))
    key = df.uid.values[order].astype("int64") * 10**8 + df.TransactionDT.values[order]
    pos = np.arange(len(key))
    for name, window in [("uid_n_24h", DAY), ("uid_n_7d", 7 * DAY)]:
        counts = np.empty(len(key), dtype="float32")
        counts[order] = pos - np.searchsorted(key, key - window, side="left")
        out[name] = counts

    # Distinct e-mail domains and devices the user has shown so far (including now).
    for col, name in [("P_emaildomain", "uid_n_emails"), ("DeviceInfo", "uid_n_devices")]:
        if col in df:
            first = ~df.duplicated(["uid", col]) & df[col].notna()
            out[name] = first.groupby(df.uid).cumsum().astype("float32")
    out.loc[df.uid.values < 0] = np.nan   # unknown users have no history
    return pd.concat([df, out], axis=1)


def transductive_uid_features(df):
    """LEAKY on purpose: aggregates over the user's whole history, future included."""
    g = df.groupby("uid").TransactionAmt
    out = pd.DataFrame({
        "uid_n_total": g.transform("size").astype("float32"),
        "uid_amt_mean_all": g.transform("mean").astype("float32"),
        "uid_amt_std_all": g.transform("std").astype("float32"),
    }, index=df.index)
    out.loc[df.uid.values < 0] = np.nan
    return out


def prior_fraud_count(df, delay_days):
    """Frauds already confirmed for this user, if a label arrives `delay_days` after the transaction."""
    frauds = df.loc[(df.isFraud == 1) & (df.uid >= 0), ["uid", "TransactionDT"]].copy()
    frauds["known_at"] = frauds.TransactionDT + delay_days * DAY
    frauds = frauds.sort_values("known_at")
    frauds["n"] = frauds.groupby("uid").cumcount() + 1
    left = df[["uid", "TransactionDT"]].reset_index().sort_values("TransactionDT")
    merged = pd.merge_asof(left, frauds[["uid", "known_at", "n"]], left_on="TransactionDT",
                           right_on="known_at", by="uid", allow_exact_matches=False)
    n = merged.set_index("index").n.reindex(df.index).fillna(0).astype("float32")
    return n.where(df.uid >= 0)


def frequency_encode(df, cols, fit_mask):
    """How often each value appeared in the training period (fit on `fit_mask` rows only)."""
    for c in cols:
        counts = df.loc[fit_mask, c].value_counts()
        df[f"{c}_freq"] = df[c].map(counts).astype("float32").fillna(0)
    return df


def select_v_columns(df, threshold=0.75):
    """Drop redundant V columns: within each missing-value pattern, keep a column only if its
    absolute correlation with every already-kept column is below `threshold`."""
    v = [c for c in df if c.startswith("V") and c[1:].isdigit()]
    groups = pd.Series(v).groupby(df[v].isna().sum().values)
    keep = []
    for _, cols in groups:
        cols = list(cols)
        corr = df[cols].sample(min(len(df), 100_000), random_state=0).corr().abs()
        kept = []
        for c in sorted(cols, key=lambda c: -df[c].nunique()):   # prefer richer columns
            if all(corr.loc[c, k] < threshold for k in kept):
                kept.append(c)
        keep += kept
    return sorted(keep, key=lambda c: int(c[1:]))


def release_memory():
    """Hand memory freed by pandas back to the OS (glibc keeps it otherwise). Linux only."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def to_float32(frame, columns=None, rows=None):
    """Dense float32 matrix for the models; categoricals become integer codes (missing → NaN).

    Filled column by column (optionally for a boolean `rows` mask), so neither a float64
    copy nor a sliced copy of the whole frame is ever made. Returns the matrix and the
    positions of the categorical columns.
    """
    columns = list(frame.columns if columns is None else columns)
    n = len(frame) if rows is None else int(np.sum(rows))
    out = np.empty((n, len(columns)), dtype=np.float32)
    cats = []
    for j, c in enumerate(columns):
        s = frame[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            col = s.cat.codes.to_numpy(np.float32)
            col[col < 0] = np.nan
            cats.append(j)
        else:
            col = s.to_numpy(np.float32, na_value=np.nan)
        out[:, j] = col if rows is None else col[rows]
    return out, cats


FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain",
             "DeviceInfo", "id_31"]


def build_features(fit_end_day, label_delay=30, normalize_d=True, keep_raw_d=("D3",),
                   groups=("frequency", "confirmed_fraud")):
    """Load IEEE-CIS and build features as in notebook 02.

    `fit_end_day`: frequency encodings are fitted on days before it (the training period).
    `keep_raw_d`: D columns not to normalise (see `normalize_d_columns`).
    `groups`: engineered groups to include, from "amount_time", "frequency", "history" and
    "confirmed_fraud". The default is notebook 02's pruned set.
    Returns (frame, feature names). The frame keeps `uid`, `day` and `isFraud` for analysis.
    """
    import pyarrow.parquet as pq

    from fraudkit.data import load_ieee, prepare_ieee

    paths = prepare_ieee()
    tx_cols = pq.read_schema(paths["train_transaction"]).names
    id_cols = pq.read_schema(paths["train_identity"]).names
    v_cols = [c for c in tx_cols if c.startswith("V")]
    keep_v = select_v_columns(load_ieee(v_cols, identity=False))
    release_memory()
    df = load_ieee([c for c in tx_cols if c not in v_cols] + keep_v + id_cols)
    raw = [c for c in df.columns if c not in ("TransactionID", "TransactionDT", "isFraud")]

    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_uid(df)
    df = frequency_encode(df, FREQ_COLS, fit_mask=(df.day < fit_end_day).values)
    df = causal_uid_features(df)
    df[f"prior_fraud_{label_delay}d"] = prior_fraud_count(df, label_delay)
    if normalize_d:
        df = normalize_d_columns(df, keep_raw=keep_raw_d)
        normalized = [c for c in D_DAYS_SINCE if c not in keep_raw_d]
        raw = [c for c in raw if c not in normalized] + [f"{c}n" for c in normalized]

    engineered = {
        "amount_time": ["amt_log", "amt_cents", "amt_odd_decimals", "hour", "weekday"],
        "frequency": [f"{c}_freq" for c in FREQ_COLS],
        "history": [c for c in df.columns if c.startswith("uid_")] + ["amt_vs_uid_mean"],
        "confirmed_fraud": [f"prior_fraud_{label_delay}d"],
    }
    return df, raw + [f for g in groups for f in engineered[g]]


def cache_feature_matrix(out_dir, extra_columns=None, **build_kwargs):
    """Build features (see `build_features`) and save a float32 matrix plus metadata to `out_dir`.

    Meant to run in its own process (`python -m fraudkit.ieee`): feature building peaks
    above 2 GB, and all of it is returned to the OS when the process exits.
    `extra_columns` maps a new column name to a function of the frame, for columns that
    are kept in the matrix but are not model features.
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, features = build_features(**build_kwargs)
    with warnings.catch_warnings():   # adding columns fragments the wide frame; harmless here
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for name, fn in (extra_columns or {}).items():
            df[name] = fn(df)
    columns = features + list(extra_columns or {})
    X, cat_pos = to_float32(df, columns)
    np.save(out / "X.npy", X)
    df[["uid", "day", "TransactionAmt", "isFraud"]].to_parquet(out / "meta.parquet")
    cats = {c: [str(v) for v in df[c].cat.categories] for c in features
            if isinstance(df[c].dtype, pd.CategoricalDtype)}
    (out / "columns.json").write_text(json.dumps(
        {"features": features, "columns": columns, "cat_pos": cat_pos, "categories": cats}))


def load_feature_matrix(out_dir):
    """Open a cached matrix memory-mapped: pages are read on demand and can be dropped by the OS."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    info = json.loads((out / "columns.json").read_text())
    X = np.load(out / "X.npy", mmap_mode="r")
    meta = pd.read_parquet(out / "meta.parquet")
    return X, meta, info


def cache_model_matrix(out_dir, train_end=120, val_end=150):
    """The matrix used by notebook 03: the pruned feature set (frequency encodings fitted on
    the training period), plus frequency encodings refitted on train + validation for the
    final model (`*_final`), plus the normalised D3n for the D3 check."""
    def final_freq(col):
        def fn(df):
            counts = df.loc[df.day < val_end, col].value_counts()
            return df[col].map(counts).astype("float32").fillna(0)
        return fn

    extra = {f"{c}_freq_final": final_freq(c) for c in FREQ_COLS}
    extra["D3n"] = lambda df: (np.floor(df.TransactionDT / DAY) - df.D3).astype("float32")
    cache_feature_matrix(out_dir, extra_columns=extra, fit_end_day=train_end, label_delay=30,
                         normalize_d=True, keep_raw_d=("D3",), groups=("frequency", "confirmed_fraud"))


if __name__ == "__main__":
    import sys

    cache_model_matrix(sys.argv[1])
