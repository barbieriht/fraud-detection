import numpy as np

from fraudkit import fairness


def test_threshold_at_fpr_flags_the_requested_share_of_negatives():
    rng = np.random.default_rng(0)
    y = rng.random(20_000) < 0.05
    s = rng.normal(size=20_000) + 2 * y
    t = fairness.threshold_at_fpr(y, s, 0.05)
    assert abs((s[~y] >= t).mean() - 0.05) < 0.002


def test_group_rates_and_fpr_ratio():
    y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    flagged = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    groups = np.array(["a"] * 5 + ["b"] * 5)
    g = fairness.group_rates(y, flagged, groups)
    assert g.loc["a", "FPR"] == 0.25 and g.loc["b", "FPR"] == 0.5
    assert g.loc["a", "TPR"] == 1.0 and g.loc["b", "TPR"] == 0.0
    assert fairness.fpr_ratio(y, flagged, groups) == 0.5


def test_group_thresholds_equalise_fpr():
    rng = np.random.default_rng(1)
    groups = np.repeat(["a", "b"], 10_000)
    y = rng.random(20_000) < 0.05
    s = rng.normal(size=20_000) + 2 * y + (groups == "b")   # group b scores higher overall
    th = fairness.group_thresholds_at_fpr(y, s, groups, 0.05)
    flagged = s >= np.vectorize(th.get)(groups)
    assert fairness.fpr_ratio(y, flagged, groups) > 0.9


def test_recall_at_fpr_perfect_and_random():
    y = np.r_[np.zeros(1000), np.ones(100)]
    assert fairness.recall_at_fpr(y, y.astype(float) + np.linspace(0, 0.1, 1100), 0.05) == 1.0
    rng = np.random.default_rng(0)
    y = rng.random(50_000) < 0.05
    assert abs(fairness.recall_at_fpr(y, rng.random(50_000), 0.05) - 0.05) < 0.02
