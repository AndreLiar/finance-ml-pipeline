"""
Personal Finance Anomaly Detection
====================================
Detect unusual transactions using 3 unsupervised models:
  - Isolation Forest    (tree-based, good for tabular data)
  - One-Class SVM       (kernel-based boundary)
  - Local Outlier Factor (density-based neighborhood)

Ensemble: a transaction is flagged if >= 2 of 3 models agree it's anomalous.

Features used:
  - abs_amount, log_amount       (transaction size)
  - day_of_week, is_weekend      (timing)
  - month, quarter               (seasonality)
  - is_round_number              (round amounts are suspicious)
  - rolling_7d_spend             (recent spending context)
  - rolling_30d_spend            (monthly context)
  - category_encoded             (merchant type)
  - type_encoded                 (debit/credit)
  - tx_count, avg_tx_amount      (activity level)
  - amount_vs_monthly_avg        (how far from personal baseline)
  - amount_vs_category_avg       (how far from category baseline)

Output:
  data/anomaly_results.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.preprocessing      import StandardScaler
from sklearn.ensemble           import IsolationForest
from sklearn.svm                import OneClassSVM
from sklearn.neighbors          import LocalOutlierFactor
from sklearn.metrics            import classification_report

from config import FEATURES_XLSX as INPUT_EXCEL, ANOMALY_RESULTS_XLSX as OUTPUT_EXCEL
from db import read_table, write_table, table_exists
from model_store import save_artifacts, data_hash as compute_data_hash
from logger import get_logger

log = get_logger(__name__)

print("=" * 60)
print("PERSONAL FINANCE ANOMALY DETECTION")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

if table_exists("features_full"):
    df = read_table("features_full", parse_dates=["date_operation"])
else:
    df = pd.read_excel(INPUT_EXCEL, sheet_name="Full Data")
    df['date_operation'] = pd.to_datetime(df['date_operation'])
df = df.sort_values('date_operation').reset_index(drop=True)

# Focus on DEBIT transactions (spending anomalies)
# But keep CREDIT for income anomaly detection too
debits = df[df['type'] == 'DEBIT'].copy()

print(f"\nTotal transactions : {len(df)}")
print(f"Debit transactions : {len(debits)}")

# ── 2. ENGINEER ANOMALY FEATURES ──────────────────────────────────────────────

# Amount vs personal monthly average (how unusual is this tx in context)
monthly_avg = debits.groupby('year_month')['debit'].transform('mean')
debits['amount_vs_monthly_avg'] = debits['debit'] / monthly_avg.replace(0, 1)

# Amount vs category average (how unusual in its own category)
cat_avg = debits.groupby('category')['debit'].transform('mean')
cat_std = debits.groupby('category')['debit'].transform('std').fillna(1)
debits['amount_vs_cat_avg']  = debits['debit'] / cat_avg.replace(0, 1)
debits['amount_z_in_cat']    = (debits['debit'] - cat_avg) / cat_std.replace(0, 1)

# Days since last transaction (gaps can indicate unusual activity)
debits['days_since_last'] = debits['date_operation'].diff().dt.days.fillna(0).clip(0, 30)

# Number of transactions on same day
daily_count = debits.groupby('date_operation')['debit'].transform('count')
debits['tx_same_day'] = daily_count

FEATURES = [
    'abs_amount',
    'log_amount',
    'day_of_week',
    'is_weekend',
    'month',
    'quarter',
    'is_round_number',
    'rolling_7d_spend',
    'rolling_30d_spend',
    'category_encoded',
    'tx_count',
    'avg_tx_amount',
    'amount_vs_monthly_avg',
    'amount_vs_cat_avg',
    'amount_z_in_cat',
    'days_since_last',
    'tx_same_day',
]

X = debits[FEATURES].fillna(0).values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures used : {len(FEATURES)}")
print(f"Samples       : {len(X)}")

# ── 3. TRAIN MODELS ───────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("TRAINING ANOMALY DETECTORS")
print("-" * 60)

# Contamination = expected % of anomalies (~5% is typical for fraud detection)
CONTAMINATION = 0.05

# Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                      random_state=42, n_jobs=-1)
iso_labels  = iso.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
iso_scores  = iso.decision_function(X_scaled)    # lower = more anomalous
iso_anomaly = (iso_labels == -1).astype(int)
print(f"  Isolation Forest     : {iso_anomaly.sum()} anomalies detected")

# One-Class SVM (nu = contamination rate)
ocsvm = OneClassSVM(nu=CONTAMINATION, kernel='rbf', gamma='scale')
ocsvm_labels  = ocsvm.fit_predict(X_scaled)
ocsvm_scores  = ocsvm.decision_function(X_scaled)
ocsvm_anomaly = (ocsvm_labels == -1).astype(int)
print(f"  One-Class SVM        : {ocsvm_anomaly.sum()} anomalies detected")

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
lof_labels  = lof.fit_predict(X_scaled)
lof_scores  = lof.negative_outlier_factor_       # more negative = more anomalous
lof_anomaly = (lof_labels == -1).astype(int)
print(f"  Local Outlier Factor : {lof_anomaly.sum()} anomalies detected")

# ── 4. ENSEMBLE — flag if >= 2 models agree ───────────────────────────────────

vote_sum         = iso_anomaly + ocsvm_anomaly + lof_anomaly
ensemble_anomaly = (vote_sum >= 2).astype(int)

# Anomaly score: normalise and combine (higher = more anomalous)
iso_norm   = (iso_scores   - iso_scores.min())   / (iso_scores.max()   - iso_scores.min() + 1e-9)
ocsvm_norm = (ocsvm_scores - ocsvm_scores.min()) / (ocsvm_scores.max() - ocsvm_scores.min() + 1e-9)
lof_norm   = (lof_scores   - lof_scores.min())   / (lof_scores.max()   - lof_scores.min() + 1e-9)

# Invert so higher score = more anomalous
ensemble_score = 1 - (iso_norm + ocsvm_norm + lof_norm) / 3

print(f"\n  Ensemble (>=2 agree) : {ensemble_anomaly.sum()} anomalies flagged "
      f"({ensemble_anomaly.sum()/len(ensemble_anomaly)*100:.1f}%)")

# ── 5. BUILD RESULTS DATAFRAME ────────────────────────────────────────────────

debits['iso_anomaly']      = iso_anomaly
debits['ocsvm_anomaly']    = ocsvm_anomaly
debits['lof_anomaly']      = lof_anomaly
debits['vote_count']       = vote_sum
debits['is_anomaly']       = ensemble_anomaly
debits['anomaly_score']    = (ensemble_score * 100).round(1)   # 0-100, higher = weirder
debits['iso_score']        = iso_scores
debits['ocsvm_score']      = ocsvm_scores
debits['lof_score']        = lof_scores

anomalies = debits[debits['is_anomaly'] == 1].sort_values('anomaly_score', ascending=False)

# ── 6. ANALYSIS ───────────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("TOP 20 ANOMALOUS TRANSACTIONS")
print("-" * 60)

cols = ['date_operation', 'description', 'category', 'debit', 'anomaly_score', 'vote_count']
for _, row in anomalies[cols].head(20).iterrows():
    print(f"  {str(row['date_operation'])[:10]}  €{row['debit']:8.2f}  "
          f"[score:{row['anomaly_score']:.0f}  votes:{int(row['vote_count'])}]  "
          f"{row['category']:<20}  {str(row['description'])[:45]}")

# Category breakdown of anomalies
print("\n" + "-" * 60)
print("ANOMALIES BY CATEGORY")
print("-" * 60)
cat_anom = anomalies.groupby('category').agg(
    count=('debit', 'count'),
    total=('debit', 'sum'),
    avg=('debit', 'mean'),
    max=('debit', 'max')
).sort_values('count', ascending=False)
print(cat_anom.to_string())

# Monthly anomaly rate
print("\n" + "-" * 60)
print("MONTHLY ANOMALY RATE")
print("-" * 60)
monthly_anom = debits.groupby('year_month').agg(
    total_tx=('is_anomaly', 'count'),
    anomaly_tx=('is_anomaly', 'sum')
).reset_index()
monthly_anom['anomaly_rate'] = (monthly_anom['anomaly_tx'] / monthly_anom['total_tx'] * 100).round(1)
print(monthly_anom[monthly_anom['anomaly_tx'] > 0].to_string(index=False))

# ── 7. SAVE ───────────────────────────────────────────────────────────────────

result_cols = [
    'date_operation', 'description', 'category', 'debit',
    'abs_amount', 'day_of_week', 'is_weekend', 'is_round_number',
    'rolling_7d_spend', 'rolling_30d_spend',
    'amount_vs_monthly_avg', 'amount_vs_cat_avg', 'amount_z_in_cat',
    'iso_anomaly', 'ocsvm_anomaly', 'lof_anomaly',
    'vote_count', 'is_anomaly', 'anomaly_score',
    'year_month'
]

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    debits[result_cols].sort_values('date_operation').to_excel(
        writer, sheet_name='All Transactions', index=False)

    anomalies[result_cols].sort_values('anomaly_score', ascending=False).to_excel(
        writer, sheet_name='Flagged Anomalies', index=False)

    cat_anom.reset_index().to_excel(
        writer, sheet_name='By Category', index=False)

    monthly_anom.to_excel(
        writer, sheet_name='Monthly Rate', index=False)

    # Model summary
    summary = pd.DataFrame({
        'Model':           ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor', 'Ensemble (>=2)'],
        'Anomalies Found': [iso_anomaly.sum(), ocsvm_anomaly.sum(), lof_anomaly.sum(), ensemble_anomaly.sum()],
        'Rate (%)':        [round(iso_anomaly.mean()*100,1), round(ocsvm_anomaly.mean()*100,1),
                            round(lof_anomaly.mean()*100,1), round(ensemble_anomaly.mean()*100,1)],
        'Method':          ['Tree-based isolation', 'Kernel boundary', 'Density neighborhood', 'Majority vote (>=2/3)'],
    })
    summary.to_excel(writer, sheet_name='Model Summary', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
_result_cols = [c for c in debits.columns if c in
                ['date_operation','description','amount','debit','category',
                 'anomaly_score','is_anomaly','if_score','svm_score','lof_score']]
write_table(debits[_result_cols].sort_values('date_operation'),           "anomaly_results")
write_table(anomalies[_result_cols].sort_values('anomaly_score',
            ascending=False),                                              "anomaly_flagged")
write_table(cat_anom.reset_index(),                                        "anomaly_by_category")
write_table(monthly_anom,                                                  "anomaly_monthly_rate")
write_table(summary,                                                       "anomaly_summary")

# ── Persist anomaly model artifacts ───────────────────────────────────────────
save_artifacts(
    "anomaly_detection",
    {
        "isolation_forest": iso,
        "ocsvm":            ocsvm,
        "scaler":           scaler,
        "feature_cols":     FEATURES,
        "contamination":    CONTAMINATION,
    },
    metrics={
        "n_flagged_ensemble": int(debits['is_anomaly'].sum()),
        "contamination":      CONTAMINATION,
    },
    data_hash=compute_data_hash(debits[FEATURES].fillna(0)),
)

log.info("Anomaly detection complete → %s", OUTPUT_EXCEL)

print(f"\nResults saved to: {OUTPUT_EXCEL}")
print("\nSheets:")
print("  - All Transactions   (with anomaly scores)")
print("  - Flagged Anomalies  (ensemble flagged)")
print("  - By Category")
print("  - Monthly Rate")
print("  - Model Summary")
