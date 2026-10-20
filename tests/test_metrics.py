import numpy as np

from fraudkit.metrics import best_cost_threshold, recall_at_alert_rate


def test_recall_at_alert_rate_perfect_ranking():
    y = np.array([0] * 95 + [1] * 5)
    s = np.arange(100)
    assert recall_at_alert_rate(y, s, 0.05) == 1.0


def test_cost_threshold_separates_classes():
    y = np.array([0] * 90 + [1] * 10)
    s = np.r_[np.zeros(90), np.ones(10)]
    t, cost = best_cost_threshold(y, s)
    assert cost == 0 and t > 0


def test_bootstrap_ci_brackets_point_estimate():
    from sklearn.metrics import average_precision_score

    from fraudkit.metrics import bootstrap_ci

    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.05
    s = y + rng.normal(0, 0.8, 2000)
    est, lo, hi = bootstrap_ci(average_precision_score, y, s, n_boot=200)
    assert lo <= est <= hi and hi - lo < 0.3


def test_time_split_is_ordered_and_complete():
    import pandas as pd

    from fraudkit.data import time_split

    df = pd.DataFrame({"Time": np.random.default_rng(0).permutation(100)})
    tr, va, te = time_split(df)
    assert (len(tr), len(va), len(te)) == (60, 20, 20)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
