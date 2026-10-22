# Project 2 · E-commerce fraud (IEEE-CIS / Vesta)

590,540 online card payments over six months, with 3.5 % fraud and 434 columns: amounts,
cards, addresses, e-mail domains, counts, time deltas, Vesta's engineered V features and
device fingerprints. Unlike project 1, most features have names and types, so this project
is about **feature engineering, validation over time and model maintenance**.

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | [Exploratory analysis](01_eda.ipynb) | Time split, informative missing values, category risk with Wilson intervals, odd-decimal amounts, V-column redundancy (339 → 130), drift and adversarial validation |
| 02 | [Feature engineering](02_feature_engineering.ipynb) | User-ID reconstruction, causal user history, confirmed fraud under label delay, 3-seed ablation with paired bootstrap tests, a leakage demonstration, new vs. returning users |
| 03 | [Models and validation](03_models_and_validation.ipynb) | A disguised calendar feature, LightGBM vs. XGBoost vs. CatBoost, blending, final test month, model staleness, SHAP |

## Key results

* **Final model (LightGBM, test month 6):** PR-AUC **0.687** (95 % CI 0.671–0.700),
  ROC-AUC 0.938. It catches **50 %** of fraud when 2 % of transactions are reviewed and
  **70 %** at 5 %.
* **Feature engineering on month 5:** PR-AUC rose from 0.639 to 0.718. The biggest gains
  came from a reconstructed **user ID** (card + address + account-opening date) and the
  user's **already-confirmed frauds**. That feature is only usable with a realistic label
  delay: with instant labels it would reach 0.84, which is unattainable in practice.
* **Label delay is a business lever.** Confirming fraud in 7 days instead of 30 is worth
  about +0.04 PR-AUC, more than most feature groups.
* **Leakage corrupts decisions.** Aggregates over each user's *whole* history (including
  the future) inflated validation by +0.03 and made a feature group look useful when its
  honest, past-only version slightly hurt.
* **Some sensible features were redundant.** Amount/time and user-history aggregates
  slightly *hurt* (confirmed with 3 seeds and paired bootstrap), because Vesta's own count
  and V columns already capture them.
* **Models go stale fast.** Retraining with one more month of data was worth +0.07 PR-AUC
  on the test month. Fresh models lose 0.08–0.14 within 15–30 days.
* **LightGBM beat XGBoost and CatBoost** on this pipeline, and a rank-average blend added
  nothing.

## Methodology notes

* Split by time: months 1–4 train, month 5 validation (all model and feature decisions),
  month 6 test (opened once). The competition's test set has no labels and isn't used.
* Every feature is causal: computed only from information available when the
  transaction is scored. Frequency encodings are fitted on the training period.
* Small differences are judged with paired bootstrap intervals and multiple seeds, not by
  comparing third decimals.

## Running on a small machine

The notebooks run on a 7 GB WSL machine. Getting there took some care, all of it in
`src/fraudkit`:

* CSV → parquet conversion streams in batches, with float32 numbers and text kept as
  categories (650 MB CSV → 58 MB parquet, 1 GB in memory).
* Arrow-to-pandas conversion with `self_destruct`, identity columns attached without a
  merge copy, and `malloc_trim` to return freed memory to the OS.
* Models receive float32 NumPy matrices built column by column (`to_float32`), not
  pandas slices, which LightGBM would convert to float64.
* Notebook 03 builds its feature matrix in a separate process and opens it memory-mapped.

## Reproduce

```bash
uv sync
export KAGGLE_API_TOKEN=...   # and accept the competition rules on kaggle.com/c/ieee-fraud-detection
uv run jupyter nbconvert --to notebook --execute --inplace projects/02_ieee_cis/*.ipynb
```

Runtime is about 45 minutes in total on a 16-core CPU (notebook 02's 3-seed ablation is the
longest).
