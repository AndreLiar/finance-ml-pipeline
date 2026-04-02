"""
Cash Flow Forecasting — Regression Pipeline
=============================================
Predict next month's total spend using time-series features.

Models:
  - Baseline    : Last-value (naive) — always predict previous month's spend
  - Linear      : Ridge Regression
  - Intermediate: Random Forest Regressor
  - Advanced    : XGBoost Regressor
  - Gradient    : Gradient Boosting Regressor

Evaluation Metrics:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R²   (Coefficient of Determination)
  - MAPE (Mean Absolute Percentage Error)

Output:
  data/cashflow_results.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection    import cross_val_score, KFold
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import Ridge
from sklearn.ensemble           import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection  import SelectKBest, f_regression
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import mean_absolute_error, mean_squared_error, r2_score
from xgboost                    import XGBRegressor

from src.config import (
    CREDITWORTHINESS_XLSX   as INPUT_MONTHLY,
    FEATURES_XLSX           as INPUT_FULL,
    CASHFLOW_RESULTS_XLSX   as OUTPUT_EXCEL,
)
from src.db import read_table, write_table, table_exists
from src.logger import get_logger

log = get_logger(__name__)

print("=" * 60)
print("CASH FLOW FORECASTING — REGRESSION PIPELINE")
print("=" * 60)

# ── 1. LOAD MONTHLY DATA ──────────────────────────────────────────────────────

if table_exists("monthly_credit"):
    monthly = read_table("monthly_credit")
else:
    monthly = pd.read_excel(INPUT_MONTHLY, sheet_name='Monthly Credit Profile')
monthly = monthly.sort_values('year_month').reset_index(drop=True)

# Load full transaction data for category-level features
if table_exists("features_full"):
    full = read_table("features_full", parse_dates=["date_operation"])
else:
    full = pd.read_excel(INPUT_FULL, sheet_name='Full Data')
    full['date_operation'] = pd.to_datetime(full['date_operation'])
full['year_month'] = full['date_operation'].dt.to_period('M').astype(str)

print(f"\nMonths available : {len(monthly)}")
print(f"Spend range      : €{monthly['spend'].min():.0f} – €{monthly['spend'].max():.0f}")
print(f"Avg monthly spend: €{monthly['spend'].mean():.0f}")

# ── 2. ENGINEER FORECASTING FEATURES ─────────────────────────────────────────

# Category-level monthly spend breakdown
cat_monthly = (
    full[full['type'] == 'DEBIT']
    .groupby(['year_month', 'category'])['debit']
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)

monthly = monthly.merge(cat_monthly, on='year_month', how='left').fillna(0)

# Lag features — previous months' spend (what we know at prediction time)
for lag in [1, 2, 3]:
    monthly[f'spend_lag{lag}'] = monthly['spend'].shift(lag)
    monthly[f'income_lag{lag}'] = monthly['income'].shift(lag)

# Rolling statistics
monthly['spend_roll3_mean'] = monthly['spend'].shift(1).rolling(3).mean()
monthly['spend_roll3_std']  = monthly['spend'].shift(1).rolling(3).std()
monthly['spend_roll6_mean'] = monthly['spend'].shift(1).rolling(6).mean()
monthly['spend_roll6_max']  = monthly['spend'].shift(1).rolling(6).max()

# Trend — direction of spend over last 3 months
monthly['spend_trend'] = (
    monthly['spend'].shift(1) - monthly['spend'].shift(3)
)

# Month-of-year (seasonality)
monthly['month_num'] = pd.to_datetime(
    monthly['year_month'], format='%Y-%m'
).dt.month
monthly['is_year_end']   = monthly['month_num'].isin([11, 12]).astype(int)
monthly['is_year_start'] = monthly['month_num'].isin([1, 2]).astype(int)
monthly['quarter']       = ((monthly['month_num'] - 1) // 3) + 1

# Rolling overdraft & volatility context
monthly['overdraft_lag1']     = monthly['overdraft'].shift(1)
monthly['volatility_lag1']    = monthly['expense_volatility'].shift(1)
monthly['savings_rate_lag1']  = monthly['savings_rate'].shift(1)
monthly['tx_count_lag1']      = monthly['tx_count'].shift(1)

# Drop rows with NaN lags (first 3 months unusable)
monthly_clean = monthly.dropna().reset_index(drop=True)
print(f"Usable months after lag features: {len(monthly_clean)}")

# ── 3. DEFINE FEATURES & TARGET ───────────────────────────────────────────────

# Base forecasting features
BASE_FEATURES = [
    'spend_lag1', 'spend_lag2', 'spend_lag3',
    'income_lag1', 'income_lag2',
    'spend_roll3_mean', 'spend_roll3_std',
    'spend_roll6_mean', 'spend_roll6_max',
    'spend_trend',
    'month_num', 'is_year_end', 'is_year_start', 'quarter',
    'overdraft_lag1', 'volatility_lag1',
    'savings_rate_lag1', 'tx_count_lag1',
    'dscr', 'debt_payments',
]

# Add category columns that exist
cat_cols = [c for c in cat_monthly.columns if c != 'year_month']
available_cats = [c for c in cat_cols if c in monthly_clean.columns]
# Use lag-1 of category spend as features
for c in available_cats:
    monthly_clean[f'{c}_lag1'] = monthly_clean[c].shift(1)
cat_lag_features = [f'{c}_lag1' for c in available_cats if f'{c}_lag1' in monthly_clean.columns]

monthly_clean = monthly_clean.dropna().reset_index(drop=True)

ALL_FEATURES = BASE_FEATURES + [f for f in cat_lag_features if f in monthly_clean.columns]
TARGET = 'spend'

X = monthly_clean[ALL_FEATURES].values
y = monthly_clean[TARGET].values

print(f"Feature count : {len(ALL_FEATURES)}")
print(f"Final samples : {len(X)}")

# ── 4. SPLIT — TIME-SERIES AWARE (no shuffling) ───────────────────────────────
# Use last 8 months as test set, rest for training

n_test  = 8
n_train = len(X) - n_test

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
dates_test = monthly_clean['year_month'].iloc[n_train:].values

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# SelectKBest: cap features at min(10, n_train//2) to avoid overfitting
# with few training months. Fitted on train only to prevent leakage.
K_BEST = min(10, max(5, n_train // 2))
selector = SelectKBest(f_regression, k=K_BEST)
selector.fit(X_train_s, y_train)
X_train_sel = selector.transform(X_train_s)
X_test_sel  = selector.transform(X_test_s)
selected_features = [ALL_FEATURES[i] for i in selector.get_support(indices=True)]

print(f"\nTime-series split:")
print(f"  Train: {n_train} months ({monthly_clean['year_month'].iloc[0]} – {monthly_clean['year_month'].iloc[n_train-1]})")
print(f"  Test : {n_test}  months ({monthly_clean['year_month'].iloc[n_train]} – {monthly_clean['year_month'].iloc[-1]})")
print(f"\nFeature selection: {len(ALL_FEATURES)} -> {K_BEST} features (SelectKBest)")
print(f"  Selected: {selected_features}")

# ── 5. EVALUATION HELPER ──────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    print(f"  {name:<35} MAE: €{mae:7.0f} | RMSE: €{rmse:7.0f} | R²: {r2:6.3f} | MAPE: {mape:5.1f}%")
    return {'Model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2),
            'R2': round(r2,4), 'MAPE': round(mape,2)}

# ── 6. BASELINE — Naive last-value ────────────────────────────────────────────

print("\n" + "-" * 60)
print("TRAINING MODELS")
print("-" * 60)

y_naive = np.concatenate([[y_train[-1]], y_test[:-1]])   # shift by 1, same length as y_test
results_list = [evaluate("Baseline (Naive last-value)", y_test, y_naive)]
pred_dict    = {"Baseline": y_naive}

# ── 7. RIDGE REGRESSION (SelectKBest features) ────────────────────────────────
# Primary model: Ridge on top-K features avoids the curse of dimensionality
# that causes negative R² when features >> training samples.

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_sel, y_train)
y_ridge = ridge.predict(X_test_sel)
results_list.append(evaluate("Ridge Regression", y_test, y_ridge))
pred_dict["Ridge"] = y_ridge

# ── 8. RANDOM FOREST ─────────────────────────────────────────────────────────

rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
results_list.append(evaluate("Random Forest", y_test, y_rf))
pred_dict["Random Forest"] = y_rf

# ── 9. XGBOOST ───────────────────────────────────────────────────────────────

xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
results_list.append(evaluate("XGBoost", y_test, y_xgb))
pred_dict["XGBoost"] = y_xgb

# ── 10. GRADIENT BOOSTING ────────────────────────────────────────────────────

gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_gb = gb.predict(X_test)
results_list.append(evaluate("Gradient Boosting", y_test, y_gb))
pred_dict["Gradient Boosting"] = y_gb

# ── 11. CROSS-VALIDATION (time-series walk-forward) ───────────────────────────

print("\n" + "-" * 60)
print("WALK-FORWARD CROSS VALIDATION (5 folds, no shuffle)")
print("-" * 60)

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for name, model, Xall, scl in [
    ("Ridge Regression",    ridge, X_train_sel, True),
    ("Random Forest",       rf,    X_train,     False),
    ("XGBoost",             xgb,   X_train,     False),
    ("Gradient Boosting",   gb,    X_train,     False),
]:
    neg_mae = cross_val_score(model, Xall, y_train, cv=tscv,
                              scoring='neg_mean_absolute_error')
    print(f"  {name:<30} CV MAE: €{-neg_mae.mean():.0f} (+/- €{neg_mae.std():.0f})")

# ── 12. FEATURE IMPORTANCE ────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("TOP 10 FEATURES — Random Forest")
print("-" * 60)

fi_df = pd.DataFrame({
    'Feature':    ALL_FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in fi_df.head(10).iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f}")

# ── 13. PREDICTIONS TABLE ─────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("ACTUAL vs PREDICTED (test months)")
print("-" * 60)

pred_df = pd.DataFrame({
    'Month':            dates_test,
    'Actual Spend (€)': y_test.round(0),
    'Baseline':         y_naive.round(0),
    'Ridge':            y_ridge.round(0),
    'Random Forest':    y_rf.round(0),
    'XGBoost':          y_xgb.round(0),
    'Gradient Boosting':y_gb.round(0),
})
print(pred_df.to_string(index=False))

# ── 14. NEXT MONTH FORECAST ───────────────────────────────────────────────────

print("\n" + "-" * 60)
print("NEXT MONTH FORECAST")
print("-" * 60)

last_row   = monthly_clean[ALL_FEATURES].iloc[[-1]].values
last_row_s = scaler.transform(last_row)
last_row_sel = selector.transform(last_row_s)   # selected features for Ridge
next_month = pd.to_datetime(monthly_clean['year_month'].iloc[-1], format='%Y-%m') + pd.DateOffset(months=1)

forecasts = {
    "Ridge Regression":    ridge.predict(last_row_sel)[0],
    "Random Forest":       rf.predict(last_row)[0],
    "XGBoost":             xgb.predict(last_row)[0],
    "Gradient Boosting":   gb.predict(last_row)[0],
}
ensemble_forecast = np.mean(list(forecasts.values()))

print(f"  Forecasting for: {next_month.strftime('%Y-%m')}")
for m, v in forecasts.items():
    print(f"  {m:<30} €{v:,.0f}")
print(f"  {'Ensemble Average':<30} €{ensemble_forecast:,.0f}")

# ── 15. SUMMARY ──────────────────────────────────────────────────────────────

summary_df = pd.DataFrame(results_list)
print("\n" + "=" * 60)
print("REGRESSION METRICS SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

best = summary_df.loc[summary_df['MAE'].idxmin(), 'Model']
print(f"\nBest model (lowest MAE): {best}")

# ── 16. PERSIST MODEL ARTIFACTS ───────────────────────────────────────────────

from src.model_store import save_artifacts, data_hash as _data_hash

_best_cf = summary_df.loc[summary_df['MAE'].idxmin(), 'Model']
save_artifacts(
    "cashflow_forecast",
    {
        "ridge":             ridge,
        "rf":                rf,
        "xgb":               xgb,
        "gb":                gb,
        "scaler":            scaler,
        "selector":          selector,          # SelectKBest fitted on train
        "selected_features": selected_features, # names of the K chosen features
        "feature_cols":      ALL_FEATURES,
        "last_row":          monthly_clean[ALL_FEATURES].iloc[[-1]].values,
        "next_month":        next_month.strftime('%Y-%m'),
    },
    metrics={
        "best_model": _best_cf,
        "best_mae":   float(summary_df.loc[summary_df['MAE'].idxmin(), 'MAE']),
        "ridge_r2":   float(summary_df.loc[summary_df['Model'] == 'Ridge Regression', 'R2'].values[0]),
    },
    data_hash=_data_hash(monthly_clean),
)

# ── 17. SAVE ─────────────────────────────────────────────────────────────────

forecast_df = pd.DataFrame({
    'Month':    [next_month.strftime('%Y-%m')],
    'Ridge':    [round(forecasts["Ridge Regression"], 0)],
    'Random Forest': [round(forecasts["Random Forest"], 0)],
    'XGBoost':  [round(forecasts["XGBoost"], 0)],
    'Gradient Boosting': [round(forecasts["Gradient Boosting"], 0)],
    'Ensemble Average':  [round(ensemble_forecast, 0)],
})

sel_df = pd.DataFrame({
    'Rank':      range(1, len(selected_features) + 1),
    'Feature':   selected_features,
    'Score':     selector.scores_[selector.get_support(indices=True)].round(4),
})

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Regression Metrics', index=False)
    pred_df.to_excel(writer, sheet_name='Actual vs Predicted', index=False)
    forecast_df.to_excel(writer, sheet_name='Next Month Forecast', index=False)
    fi_df.to_excel(writer, sheet_name='Feature Importance', index=False)
    sel_df.to_excel(writer, sheet_name='Selected Features (Ridge)', index=False)
    monthly_clean[['year_month', 'spend', 'income', 'net'] + BASE_FEATURES[:8]].to_excel(
        writer, sheet_name='Monthly Features', index=False
    )

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(summary_df,  "cashflow_metrics")
write_table(pred_df,     "cashflow_predictions")
write_table(forecast_df, "cashflow_forecast")
write_table(fi_df,       "cashflow_feature_imp")
write_table(sel_df,      "cashflow_selected_features")
log.info("Cashflow forecast complete -> %s", OUTPUT_EXCEL)

print(f"\nResults saved to: {OUTPUT_EXCEL}")
print("\nSheets:")
print("  - Regression Metrics         (MAE, RMSE, R², MAPE)")
print("  - Actual vs Predicted        (test period)")
print("  - Next Month Forecast")
print("  - Feature Importance         (Random Forest)")
print(f"  - Selected Features (Ridge) (top {K_BEST} by F-score)")
print("  - Monthly Features")
