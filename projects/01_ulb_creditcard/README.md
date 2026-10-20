# Project 1 · Credit card fraud (ULB)

284,807 European card transactions over two days. 0.17 % are fraud, and 28 of the 30
features are anonymised PCA components. The project uses this dataset to study **extreme class
imbalance**, **anomaly detection** and **turning scores into business decisions**.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | [Exploratory analysis](01_eda.ipynb) | Imbalance, amount and time-of-day patterns, KS feature screen, within-fraud correlation, t-SNE map |
| 02 | [Imbalance strategies](02_imbalance_strategies.ipynb) | Time-based split, PR-AUC vs ROC-AUC, class weights, undersampling, SMOTE, calibration damage, rolling-origin CV |
| 03 | [Anomaly detection](03_anomaly_detection.ipynb) | Isolation Forest, Mahalanobis, LOF, autoencoder vs. supervised; overlap analysis; how many labels beat no labels |
| 04 | [Thresholds, costs, explainability](04_threshold_cost_explainability.ipynb) | Optuna tuning, Platt calibration, € cost-based decision rules, final test evaluation, SHAP |

## Key results

* **Final model** (LightGBM, untouched test period): PR-AUC **0.81** (95 % CI 0.73–0.89).
  It catches **85 %** of frauds when 0.5 % of transactions are reviewed.
* **Imbalance handling depends on the model.** Class weights lift logistic regression
  from 0.63 to 0.76 PR-AUC, while LightGBM is best unweighted. Undersampling and SMOTE
  never paid off, and weighting or resampling inflated predicted risk by up to 90×.
* **Labels beat anomaly detection.** LOF trained on legit history catches 74 % of fraud at
  a 0.5 % budget without fraud labels, but LightGBM trained on just **5** labelled frauds
  out-ranks it.
* **Decisions matter as much as models.** On the test period, cost-based thresholds on
  calibrated probabilities cost **~24 % less** than a fixed review budget. An
  amount-aware rule matched that cost while ignoring small card-testing frauds. That is
  "optimal" only under a naive cost model.
* **Hyperparameter tuning didn't help.** Optuna improved cross-validation and lost on
  validation, all within noise.

## Methodology notes

* Duplicate rows (1,081) are dropped before any split to prevent leakage.
* The data is split **by time**: 60 % train, 20 % validation, 20 % test. Test is used only in notebook 04.
* Every headline metric comes with a bootstrap confidence interval, because validation and
  test hold only 57 and 74 frauds.

## Reproduce

```bash
uv sync
export KAGGLE_API_TOKEN=...                       # data downloads on first run
uv run jupyter nbconvert --to notebook --execute --inplace projects/01_ulb_creditcard/*.ipynb
```

The four notebooks take about 5 minutes in total on a 16-core CPU.
