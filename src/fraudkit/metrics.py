"""Fraud-oriented evaluation metrics shared across projects."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_alert_rate(y_true, scores, alert_rate=0.05):
    """Recall when only the top `alert_rate` fraction of cases can be reviewed."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    k = max(1, int(np.ceil(alert_rate * len(scores))))
    top = np.argsort(-scores)[:k]
    return y_true[top].sum() / max(1, y_true.sum())


def best_cost_threshold(y_true, scores, cost_fn=100.0, cost_fp=5.0):
    """Threshold minimizing total cost (missed fraud vs. analyst review)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    costs = [
        cost_fn * ((scores < t) & (y_true == 1)).sum()
        + cost_fp * ((scores >= t) & (y_true == 0)).sum()
        for t in thresholds
    ]
    i = int(np.argmin(costs))
    return thresholds[i], costs[i]


def fraud_report(y_true, scores, alert_rate=0.05):
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        f"recall@{alert_rate:.0%}": recall_at_alert_rate(y_true, scores, alert_rate),
    }
