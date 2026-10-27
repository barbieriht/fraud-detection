# Project 3 · Account-opening fraud and fairness (Bank Account Fraud, Feedzai)

1 million synthetic applications to open a bank account online over 8 months, 1.1 %
fraudulent, from the BAF suite (NeurIPS 2022). The data keeps the statistical structure
and biases of a real dataset and includes protected attributes such as age. This
project is about **operating a model under a business constraint** and **auditing who
bears its errors**.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | [Exploratory analysis](01_eda.ipynb) | Rising fraud rate, missing-value sentinels, weak individual signals, prevalence by age and income, age proxies, feature drift |
| 02 | [A model at a fixed false-positive rate](02_model_at_fixed_fpr.ipynb) | Recall at 5 % FPR, logistic regression vs. LightGBM, final test months, threshold drift and how to fix it, SHAP |
| 03 | [Fairness audit](03_fairness.ipynb) | Predictive equality across age groups, where the gap comes from, removing age, group thresholds, the fairness–recall trade-off, thresholds set in advance |

## Key results

* **Model:** LightGBM catches **54.5 %** of fraud at 5 % FPR on test months 6–7 (95 % CI
  52.7–56.4 %), ROC-AUC 0.895, in line with the BAF paper. It is 2.5 points ahead of
  logistic regression.
* **Thresholds drift.** A threshold set for 5 % FPR in month 5 flags **6.5 %** of
  legitimate applicants in month 6, 30 % above the agreed limit. Setting it as a fixed
  **flag rate** (a quantile of current scores) holds 4.8–4.9 % with no new labels.
  Dropping the most-drifting features doesn't help.
* **The model is unequal across ages.** At one threshold, legitimate applicants aged 50+
  are wrongly flagged **11.4 %** of the time vs. **3.8 %** under 50 (FPR ratio **0.33**).
  Fraud really is more common among older identities, but the cost lands on innocent
  older applicants.
* **Removing the age column isn't enough.** The other features predict "50+" with AUC
  0.84 and carry ~60 % of the score gap. The ratio only improves to 0.43.
* **Group thresholds** reach full parity at a cost of **2.6 points of recall**, or about
  2 points for the 80 % rule. Set in advance on month 5, they still give a ratio of 0.92
  on the test months.
* The notebooks discuss the legal caveat (group thresholds use age at decision time) and
  alternatives (in-processing, re-weighting).

## Reproduce

```bash
uv sync
export KAGGLE_API_TOKEN=...   # Base.csv downloads on first run
uv run jupyter nbconvert --to notebook --execute --inplace projects/03_bank_account_fraud/*.ipynb
```

The three notebooks take about 10 minutes on a 16-core CPU.
