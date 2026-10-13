# Rossmann Store Sales Prediction

Predicts daily sales for Rossmann drug stores with tree-based models (Random Forest and XGBoost), using the Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset (about 1M training rows, 1,115 stores).

## Problem

Given a store, a date, and promotion/holiday information, predict that day's `Sales`. The competition metric is **RMSPE**; during development I tracked **RMSE** in sales units and checked RMSPE on the best models.

## Results

### Kaggle leaderboard (RMSPE, lower is better)

| Model | Private score | Public score |
|---|---|---|
| Random Forest | 0.14857 | 0.13414 |
| **XGBoost (final)** | **0.13729** | **0.12863** |

### Validation (chronological split: last 25% of dates)

| Model | Val RMSE | Val RMSPE |
|---|---|---|
| Mean baseline (measured on train) | 3081 | |
| Linear Regression / Ridge / Lasso | ~2785 | |
| Decision Tree | 1514 | |
| Random Forest (default, 100 trees) | 1249 | |
| Random Forest (200 trees, `min_samples_leaf=2`, `max_features=0.75`) | 1234 | |
| XGBoost (`Store` as numeric, `max_depth=8`) | 1266 | |
| **XGBoost (`Store` as category, tuned)** | **1051** | **0.161** |
| XGBoost, `log1p(Sales)` target | 1127 | 0.162 |

Final XGBoost parameters: `learning_rate=0.2`, `max_depth=4`, `min_child_weight=2`, `subsample=0.8`, `colsample_bytree=0.9`, early stopping on validation RMSE (stopped at 618 rounds).

## Key Findings

1. **Treating `Store` as a native categorical was the biggest gain.** A numeric store ID has no meaningful order. As a category, the model learns a separate effect per store, which moved validation RMSE from about 1266 to 1051 together with tuning.
2. **Shallow trees generalized better.** `max_depth=4` beat `max_depth=8`. The validation set is the future, so deep trees that memorize period-specific patterns transfer worse.
3. **Random Forest hyperparameters were nearly flat.** `min_samples_leaf` and `max_features` moved RMSE by only a few percent. Larger leaves hurt, since each store appears ~800 times and fine-grained store patterns are real signal.
4. **`log1p` target did not help.** It tied on RMSPE (0.162 vs 0.161) and lost on RMSE, so the raw target was kept.
5. **Feature importance is stable at the top:** `Promo` (~0.46) and `Store` (~0.22) dominate across default and tuned models. The ranking of weaker features changed between runs, so only the top two are reported as a finding. Gain importance is biased toward high-cardinality features like `Store`.
6. **Validation scores are worse than leaderboard scores** (0.161 vs 0.137 private). The validation window and the test period differ, so compare models against each other, not across sets.

## Approach

**Data cleaning**
- Merged train/test with `store.csv` (left join on `Store`).
- Structural nulls (`Promo2Since*`, `PromoInterval`) occur exactly when `Promo2 == 0`; filled with `0` / `'None'` instead of imputed.
- `StateHoliday` mixed types normalized to strings.
- 11 test rows with missing `Open` (one store, consecutive days, no holiday flags) set to open.
- Closed-store rows removed from training; test predictions for closed days set to 0.
- `Customers` is not used (it does not exist at prediction time).

**Features**
- Date: `Day`, `Month`, `Year`, `WeekOfYear`, `DayOfWeek`
- Derived: `CompetitionOpenMonths`, `Promo2OpenWeeks`, `IsPromo2Month`
- Store attributes: `StoreType`, `Assortment`, `CompetitionDistance`, promo and holiday flags
- XGBoost: `Store`, `DayOfWeek`, `StateHoliday`, `StoreType`, `Assortment` as native categoricals (categories defined from train only)
- Random Forest notebook: median imputation, one-hot encoding, and scaling fit on train only
- XGBoost notebook: missing `CompetitionDistance` filled with the maximum distance (no nearby competitor)

**Validation:** data sorted by date; first 75% train, last 25% validation. A random split would leak future information on time-ordered data.

## Repository Structure

```
.
├── Rossmann_store_sales.ipynb   # EDA, cleaning, linear/tree/Random Forest models, RF submission
├── XGBoost.ipynb                # categorical Store, tuning, final XGBoost model, submission
├── submission.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/Rossmann-store-sales-prediction-model.git
cd Rossmann-store-sales-prediction-model
pip install -r requirements.txt
```

The notebooks download the data with `opendatasets`, which asks for your Kaggle username and API key on first run. Accept the competition rules on the Kaggle page first. Data is saved to `./rossmann-store-sales/` and is not committed.

Run `Rossmann_store_sales.ipynb` for the Random Forest pipeline and `XGBoost.ipynb` for the final model. The Random Forest takes about 8 minutes to train on 4 cores.

## Limitations and Next Steps

- The final XGBoost model is trained on the first 75% of the data only. Refitting on all data with the best number of rounds should improve the submission.
- Early stopping and reporting use the same validation set, so the validation score is slightly optimistic.
- Add store-level features (average sales per store and per store × weekday, computed from the training period only) and holiday-proximity features.
- Compare with LightGBM and an ensemble.
- The mean baseline is weak; a per-store, per-weekday, per-promo median would be a fairer benchmark.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Plotly, Seaborn, Jupyter

## Data

The dataset belongs to Rossmann/Kaggle and is subject to the competition's terms. It is not included in this repository.