import numpy as np
import pandas as pd

from fraudkit import ieee

DAY = ieee.DAY


def _frame():
    # Two users (uid 0 and 1) plus one transaction with an unknown user.
    return pd.DataFrame({
        "uid": [0, 1, 0, 0, -1, 1],
        "TransactionDT": [0, 10, DAY // 2, 3 * DAY, 3 * DAY, 40 * DAY],
        "TransactionAmt": [10.0, 50.0, 30.0, 20.0, 99.0, 70.0],
        "isFraud": [0, 1, 1, 0, 1, 0],
    })


def test_causal_features_only_use_the_past():
    out = ieee.causal_uid_features(_frame())
    assert out.uid_n_prev.tolist()[:4] == [0, 0, 1, 2]
    assert np.isnan(out.uid_amt_mean_prev[0])          # no history yet
    assert out.uid_amt_mean_prev[3] == 20.0            # mean of 10 and 30, not including 20
    assert out.uid_n_24h.tolist()[:4] == [0, 0, 1, 0]  # day 3 is > 24h after day 0.5
    assert out.iloc[4].drop(_frame().columns).isna().all()   # unknown user: no history


def test_prior_fraud_respects_label_delay():
    df = _frame()
    assert ieee.prior_fraud_count(df, 0).tolist()[:4] == [0, 0, 0, 1]
    assert ieee.prior_fraud_count(df, 7).tolist()[:4] == [0, 0, 0, 0]   # label not known yet
    assert ieee.prior_fraud_count(df, 7)[5] == 1                       # known 40 days later
    assert np.isnan(ieee.prior_fraud_count(df, 0)[4])


def test_add_uid_marks_incomplete_rows_unknown():
    df = pd.DataFrame({"TransactionDT": [0, DAY, 2 * DAY], "card1": [1.0, 1.0, 2.0],
                       "addr1": [5.0, 5.0, np.nan], "D1": [0.0, 1.0, 0.0]})
    uid = ieee.add_uid(df).uid.tolist()
    assert uid[0] == uid[1] >= 0 and uid[2] == -1


def test_to_float32_encodes_categories_and_missing():
    frame = pd.DataFrame({"a": [1.5, np.nan], "b": pd.Categorical(["x", None])})
    X, cats = ieee.to_float32(frame)
    assert X.dtype == np.float32 and cats == [1]
    assert X[0, 0] == 1.5 and np.isnan(X[1, 0]) and X[0, 1] == 0 and np.isnan(X[1, 1])


def test_to_float32_row_mask_and_column_subset():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    X, _ = ieee.to_float32(frame, columns=["b"], rows=np.array([True, False, True]))
    assert X.tolist() == [[4.0], [6.0]]
