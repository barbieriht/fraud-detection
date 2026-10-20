"""Fraud-oriented evaluation metrics shared across projects."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_alert_rate(y_true, scores, alert_rate=0.05):
    """Recall when only the top `alert_rate` fraction of cases can be reviewed."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    k = max(1, int(np.ceil(alert_rate * len(scores))))
    top = np.argsort(-scores)[:k]
    return y_true[top].sum() / max(1, y_true.sum())


def cost_curve(y_true, scores, cost_fn=100.0, cost_fp=5.0, n=501):
    """Total cost at candidate thresholds: missed fraud costs `cost_fn`, each alert reviewed costs `cost_fp`."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, n)))
    costs = np.array([
        cost_fn * ((scores < t) & (y_true == 1)).sum()
        + cost_fp * ((scores >= t) & (y_true == 0)).sum()
        for t in thresholds
    ])
    return thresholds, costs


def best_cost_threshold(y_true, scores, cost_fn=100.0, cost_fp=5.0):
    """Threshold minimizing total cost (missed fraud vs. analyst review)."""
    thresholds, costs = cost_curve(y_true, scores, cost_fn, cost_fp)
    i = int(np.argmin(costs))
    return thresholds[i], costs[i]


def bootstrap_ci(metric, y_true, scores, n_boot=1000, level=0.95, seed=0):
    """Point estimate and percentile bootstrap interval for `metric(y_true, scores)`.

    Resamples rows with replacement; with only a few hundred frauds in a test
    set the interval is wide, and reporting it keeps model comparisons honest.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], scores[idx]))
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return metric(y_true, scores), lo, hi


def fraud_report(y_true, scores, alert_rate=0.05):
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        f"recall@{alert_rate * 100:g}%": recall_at_alert_rate(y_true, scores, alert_rate),
    }
