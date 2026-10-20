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
