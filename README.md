# Rossmann Store Sales Prediction

Predicts daily sales for Rossmann drug stores with a Random Forest regressor, using the Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset (about 1M training rows, 1,115 stores).

## Problem

Given a store, a date, and promotion/holiday information, predict that day's `Sales`.

## Results

Metric: **RMSE** in sales units (lower is better), on a chronological validation set (the last 25% of dates).

| Model | Validation RMSE |
|---|---|
| Mean baseline (predict average sales; measured on train) | 3081 |
| Linear Regression / Ridge / Lasso | ~2785 |
| Decision Tree (default) | 1514 |
| Random Forest (default, 100 trees) | 1249 |
| **Random Forest (final: 200 trees, `min_samples_leaf=2`, `max_features=0.75`)** | **1234** |

- Tree models cut linear-model error by about 55%, which points to strong non-linear interactions (store, promo, day of week).
- The final forest is ~1% better than default. A single validation split cannot separate that from noise, so treat the two forests as roughly equal.
- Hyperparameter sweeps (30 trees each, validation RMSE):

| `min_samples_leaf` | RMSE | `max_features` | RMSE |
|---|---|---|---|
| 1 | 1265 | 0.3 | 1308 |
| 2 | 1257 | 0.5 | 1254 |
| 3 | 1269 | 0.7 | 1240 |
| 5 | 1296 | 1.0 | 1265 |

  Larger leaves hurt (each store appears ~800 times, so fine-grained store patterns are real signal). `max_features` around 0.7 is the best of the values tested.

## Approach

**Data cleaning**
- Merged train/test with `store.csv` (left join on `Store`).
- Structural nulls (`Promo2Since*`, `PromoInterval`) confirmed to occur exactly when `Promo2 == 0`; filled with `0` / `'None'` instead of imputing a typical value.
- `StateHoliday` mixed types normalized to strings.
- The 11 test rows with missing `Open` (one store, consecutive days) were set to open.
- Closed-store rows removed from training; predictions for closed test days are set to 0.
- `Customers` is not used (it does not exist at prediction time).

**Preprocessing (fit on train only, applied to validation and test)**
- `SimpleImputer(median)` for `CompetitionDistance`
- `OneHotEncoder(handle_unknown='ignore')` for categorical columns
- `StandardScaler` for numeric columns

**Features:** `Store`, `CompetitionDistance`, `Day`, `Month`, `Year`, `WeekOfYear`, `DayOfWeek`, promo and holiday flags, `StoreType`, `Assortment`, `PromoInterval`.

**Validation:** data sorted by date; first 75% train, last 25% validation, so the model is always evaluated on the future. A random split would leak information on time-ordered data.

## Repository Structure

```
.
├── Rossmann_store_sales.ipynb   # full pipeline: EDA, cleaning, models, tuning, submission
├── submission.csv               # Kaggle submission from the final model
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

The notebook downloads the data with `opendatasets`, which asks for your Kaggle username and API key on first run. You must also accept the competition rules on the Kaggle page. The data is saved to `./rossmann-store-sales/` and is not committed. Then run the notebook top to bottom.

Note: the final forest takes about 8 minutes to train on 4 cores.

## Limitations and Next Steps

- The mean baseline is weak. A per-store, per-weekday, per-promo median is a fairer benchmark.
- Evaluate with RMSPE, the competition's metric. RMSE is dominated by high-sales stores.
- `Store` is treated as a numeric value. Store-level aggregates (computed from the training period only) or a native-categorical model would use it better.
- Try `log1p(Sales)` as the target, duration features (months since competition opened / Promo2 started), and LightGBM.
- Random Forest is slow and memory-heavy at this size.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Plotly, Seaborn, Jupyter

## Data

The dataset belongs to Rossmann/Kaggle and is subject to the competition's terms. It is not included in this repository.