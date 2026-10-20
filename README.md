# Fraud Detection Portfolio

A series of machine-learning projects on real fraud datasets, each focused on a different
problem fraud teams face. Every project is a set of commented notebooks that explain the
concepts, the strategies tried and the reasons behind each choice.

| # | Project | Dataset | Focus | Status |
|---|---|---|---|---|
| 1 | [Credit card fraud](projects/01_ulb_creditcard/) | ULB Credit Card Fraud | Class imbalance, anomaly detection, cost-based decisions, SHAP | Done |
| 2 | E-commerce fraud | IEEE-CIS | Feature engineering, gradient boosting, time-based validation | Planned |
| 3 | Account-opening fraud | Bank Account Fraud (Feedzai) | Fairness, distribution shift, alert budgets | Planned |
| 4 | Money laundering | Elliptic Bitcoin | Graph ML vs. graph features | Planned |
| 5 | Transaction behaviour | PaySim / Sparkov | Sequence features, demo app | Optional |

## Setup

```bash
uv sync                       # create the environment
export KAGGLE_API_TOKEN=...   # Kaggle API token, used to download the data
uv run jupyter lab
```

Each project's first notebook downloads its own data into `data/raw/` (not committed).

## Shared code

`src/fraudkit/` holds code reused across projects, for example `metrics.py`: PR-AUC,
recall when only a fixed share of cases can be reviewed, and the cost-minimising threshold.
Run its tests with `uv run pytest`.
