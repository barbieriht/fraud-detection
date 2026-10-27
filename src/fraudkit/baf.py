"""Shared preprocessing and model settings for the Bank Account Fraud project (project 3)."""

import lightgbm as lgb
import numpy as np

from fraudkit.data import load_baf
from fraudkit.ieee import to_float32

# The datasheet encodes "missing" as negative values in these columns.
MISSING_IF_NEGATIVE = ["prev_address_months_count", "current_address_months_count",
                       "intended_balcon_amount", "bank_months_count",
                       "session_length_in_minutes", "device_distinct_emails_8w"]
DROP = ["fraud_bool", "month", "device_fraud_count"]   # label, time, and a constant column
PROTECTED = "customer_age"

LGB_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "num_threads": 12, "seed": 0}


def load(variant="Base"):
    """BAF with missing-value sentinels turned into NaN, plus an `age_group` column (<50 / 50+)."""
    df = load_baf(variant)
    for c in MISSING_IF_NEGATIVE:
        df[c] = df[c].where(df[c] >= 0)
    df["age_group"] = np.where(df.customer_age >= 50, "50+", "<50")
    return df


def features(df, drop_protected=False):
    cols = [c for c in df.columns if c not in DROP and c != "age_group"]
    return [c for c in cols if not (drop_protected and c == PROTECTED)]


def matrix(df, feats, rows=None):
    """float32 matrix (categories as integer codes) and the categorical column positions."""
    return to_float32(df, feats, rows=None if rows is None else np.asarray(rows))


def fit(df, rows, feats, rounds, params=None, valid_rows=None, early_stopping=None):
    """Train LightGBM on `rows` (boolean mask). With `valid_rows` and `early_stopping`,
    stop once validation AUC hasn't improved for that many rounds."""
    X, cats = matrix(df, feats, rows)
    train = lgb.Dataset(X, df.fraud_bool.values[np.asarray(rows)], feature_name=feats,
                        categorical_feature=cats)
    kwargs = {}
    if valid_rows is not None:
        Xv, _ = matrix(df, feats, valid_rows)
        kwargs["valid_sets"] = [lgb.Dataset(Xv, df.fraud_bool.values[np.asarray(valid_rows)], reference=train)]
        kwargs["callbacks"] = [lgb.early_stopping(early_stopping, verbose=False)]
    return lgb.train({**LGB_PARAMS, "metric": "auc", **(params or {})}, train, rounds, **kwargs)


def predict(model, df, rows, feats):
    X, _ = matrix(df, feats, rows)
    return model.predict(X, num_iteration=model.best_iteration or None)
