"""Operating-point and group-fairness metrics (project 3).

In account-opening fraud the business fixes the share of legitimate applicants it is
willing to reject (the false-positive rate, FPR) and asks how much fraud is caught there.
Fairness is then judged by comparing error rates across groups at that operating point.
"""

import numpy as np
import pandas as pd


def threshold_at_fpr(y_true, scores, fpr=0.05):
    """Score threshold that flags (≥ threshold) a share `fpr` of the legitimate cases."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    return float(np.quantile(scores[y_true == 0], 1 - fpr))


def recall_at_fpr(y_true, scores, fpr=0.05):
    """Share of fraud caught at the threshold that flags a share `fpr` of legitimate cases."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    return float((scores[y_true == 1] >= threshold_at_fpr(y_true, scores, fpr)).mean())


def rates(y_true, flagged):
    """TPR (recall), FPR, flag rate and prevalence for boolean decisions."""
    y_true, flagged = np.asarray(y_true).astype(bool), np.asarray(flagged).astype(bool)
    pos, neg = y_true.sum(), (~y_true).sum()
    return {
        "TPR": (flagged & y_true).sum() / pos if pos else np.nan,
        "FPR": (flagged & ~y_true).sum() / neg if neg else np.nan,
        "flag rate": flagged.mean(),
        "prevalence": y_true.mean(),
        "n": len(y_true),
    }


def group_rates(y_true, flagged, groups):
    """`rates` for every group, plus each group's FPR and TPR relative to the best-treated group."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "f": np.asarray(flagged), "g": np.asarray(groups)})
    out = pd.DataFrame({g: rates(d.y, d.f) for g, d in frame.groupby("g", observed=True)}).T
    # Predictive equality: ratio of the lowest FPR to each group's FPR (1 = parity; the
    # "80 % rule" asks for ≥ 0.8). For TPR, ratio of each group's TPR to the highest TPR.
    out["FPR ratio"] = out.FPR.min() / out.FPR
    out["TPR ratio"] = out.TPR / out.TPR.max()
    return out


def fpr_ratio(y_true, flagged, groups):
    """Smallest group FPR divided by the largest (1 = equal false-positive rates)."""
    g = group_rates(y_true, flagged, groups)
    return float(g.FPR.min() / g.FPR.max())


def group_thresholds_at_fpr(y_true, scores, groups, fpr=0.05):
    """One threshold per group, each giving that group the same FPR."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(scores), "g": np.asarray(groups)})
    return {g: threshold_at_fpr(d.y, d.s, fpr) for g, d in frame.groupby("g", observed=True)}
