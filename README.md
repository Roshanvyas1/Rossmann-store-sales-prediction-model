# Rossmann Store Sales Forecasting

Predicts daily sales for Rossmann drug stores with a Random Forest regressor, using the Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset.

## Problem

Given a store, a date, and promotion/holiday information, predict that day's `Sales`.

## Results

Evaluated on a held-out validation set. Metric: **RMSE** (in sales units, lower is better).

| Model | Validation RMSE |
|---|---|
| Baseline: predict the mean sales | 3081 |
| Random Forest (default parameters) | 1249 |
| Random Forest (regularized parameters) | 1235 |

- Both Random Forests cut the baseline error by about 60%.
- The regularized forest is only ~1% better than default, which is within what a single validation split can show as noise. Its main advantage is **speed and smaller model size**, not accuracy.
- Validation split: `[describe: date-based or random, and the size]`.

## Approach

### Data cleaning
- Merged train/test with `store.csv` (left join on `Store`).
- `Open` NaN in the test set (one store, consecutive days) filled from that store's weekday history and holiday flags.
- `StateHoliday` mixed types (`0` / `'0'`) normalized to strings.
- Structural nulls (`PromoInterval`, `Promo2Since*`) treated as "no Promo2" rather than imputed with a median.
- `CompetitionDistance` imputed with the train median, applied to test.
- Closed-store rows removed from training; test predictions for closed days set to 0.
- `Customers` dropped (not available at prediction time).

### Features
- Date: `Day`, `Month`, `Year`, `WeekOfYear`, `DayOfWeek`
- Store: `StoreType`, `Assortment`, `CompetitionDistance`, promo flags
- Categoricals encoded with an encoder fit on train only

### Modeling
- `RandomForestRegressor`, default vs regularized (`min_samples_leaf`, `max_features`, `max_samples`).
- Training time: default `[X]` min, regularized `[Y]` min.

## Repository Structure

```
.
├── rossmann_sales.ipynb   # full pipeline: cleaning, features, models, evaluation
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Download `train.csv`, `test.csv`, and `store.csv` from the [Kaggle data page](https://www.kaggle.com/c/rossmann-store-sales/data) into a `data/` folder, then run the notebook top to bottom.

## Limitations and Next Steps

- The mean baseline is weak. A per-store, per-weekday, per-promo median baseline would be a fairer benchmark.
- Evaluate with RMSPE, the competition metric.
- Add store-level features, a `log1p(Sales)` target, and a LightGBM comparison.
- Random Forest is slow and memory-heavy at ~1M rows.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Jupyter

## Data

The dataset belongs to Rossmann/Kaggle and is subject to the competition's terms. It is not included in this repository.