"""
Step 3 — Model Training & Evaluation
Tasks:
  1. Label transactions with auto-tagged categories (keyword rules as baseline)
  2. Split: 70% Train / 15% Validation / 15% Test
  3. Train 3 models:
       - Baseline  : Logistic Regression
       - Intermediate: Random Forest
       - Advanced  : XGBoost
  4. Evaluate: Accuracy, F1, Confusion Matrix, ROC-AUC
  5. Save results to data/model_results.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

from src.config import FEATURES_XLSX as INPUT_EXCEL, MODEL_RESULTS_XLSX as OUTPUT_EXCEL
from src.db import read_table, write_table, table_exists
from src.logger import get_logger

log = get_logger(__name__)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — MODEL TRAINING & EVALUATION")
print("=" * 60)

if table_exists("feature_matrix"):
    df = read_table("feature_matrix", parse_dates=["date_operation"])
else:
    df = pd.read_excel(INPUT_EXCEL, sheet_name="Feature Matrix")
print(f"\nLoaded {len(df)} transactions, {df.shape[1]} columns")

# ── 2. PREPARE FEATURES & LABELS ──────────────────────────────────────────────

# Drop rare classes (< 5 samples) — not enough to train/test on
cat_counts   = df['category'].value_counts()
valid_cats   = cat_counts[cat_counts >= 5].index
df           = df[df['category'].isin(valid_cats)].copy()
print(f"After dropping rare categories: {len(df)} transactions")
print(f"Classes: {sorted(df['category'].unique())}")

FEATURE_COLS = [
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'is_weekend', 'quarter', 'month_part_encoded',
    'abs_amount', 'log_amount', 'is_round_number',
    'rolling_7d_spend', 'rolling_30d_spend',
    'monthly_income', 'monthly_spend', 'monthly_net',
    'tx_count', 'avg_tx_amount', 'max_tx_amount', 'savings_rate',
    'type_encoded',
]

X = df[FEATURE_COLS].values

# Encode target label
le = LabelEncoder()
y  = le.fit_transform(df['category'])
class_names = le.classes_

print(f"\nFeatures : {len(FEATURE_COLS)}")
print(f"Samples  : {len(X)}")
print(f"Classes  : {len(class_names)}")

# ── 3. CHRONOLOGICAL SPLIT (no shuffle — avoids temporal leakage) ─────────────
# Data is already sorted by date. Use the last 15% of time as test,
# the preceding 15% as validation, and the rest as train.
# This mirrors real deployment: train on past, evaluate on future.

df = df.sort_values('date_operation').reset_index(drop=True)
X  = df[FEATURE_COLS].values
y  = le.transform(df['category'])

n       = len(X)
n_test  = max(int(n * 0.15), 1)
n_val   = max(int(n * 0.15), 1)
n_train = n - n_val - n_test

X_train, y_train = X[:n_train],              y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

cutoff_val  = df['date_operation'].iloc[n_train].strftime('%Y-%m-%d')
cutoff_test = df['date_operation'].iloc[n_train+n_val].strftime('%Y-%m-%d')

print(f"\nChronological split (no shuffle):")
print(f"  Train      : {len(X_train)} transactions  (up to {cutoff_val})")
print(f"  Validation : {len(X_val)} transactions  ({cutoff_val} – {cutoff_test})")
print(f"  Test       : {len(X_test)} transactions  ({cutoff_test} – present)")

# Scale for Logistic Regression
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── 4. TRAIN MODELS ───────────────────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        random_state=42, class_weight='balanced', n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    ),
}

results      = {}
val_results  = {}
all_reports  = {}

print("\n" + "-" * 60)
print("TRAINING MODELS")
print("-" * 60)

for name, model in models.items():
    print(f"\n  [{name}]")

    # Use scaled data for Logistic Regression, raw for tree models
    Xtr = X_train_s if name == "Logistic Regression" else X_train
    Xvl = X_val_s   if name == "Logistic Regression" else X_val
    Xts = X_test_s  if name == "Logistic Regression" else X_test

    # Train
    model.fit(Xtr, y_train)

    # Validation performance (for hyperparameter tuning reference)
    y_val_pred  = model.predict(Xvl)
    val_acc     = accuracy_score(y_val, y_val_pred)
    val_f1      = f1_score(y_val, y_val_pred, average='weighted')
    val_results[name] = {'val_accuracy': val_acc, 'val_f1': val_f1}
    print(f"    Val Accuracy : {val_acc:.4f}  |  Val F1 (weighted): {val_f1:.4f}")

    # Final test performance
    y_test_pred = model.predict(Xts)
    test_acc    = accuracy_score(y_test, y_test_pred)
    test_f1_w   = f1_score(y_test, y_test_pred, average='weighted')
    test_f1_m   = f1_score(y_test, y_test_pred, average='macro')

    # ROC-AUC (one-vs-rest, requires predict_proba)
    if hasattr(model, 'predict_proba'):
        y_prob    = model.predict_proba(Xts)
        # Only compute if all classes represented in test set
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        except Exception:
            roc_auc = None
    else:
        roc_auc = None

    results[name] = {
        'accuracy':    round(test_acc,  4),
        'f1_weighted': round(test_f1_w, 4),
        'f1_macro':    round(test_f1_m, 4),
        'roc_auc':     round(roc_auc,   4) if roc_auc else 'N/A',
    }
    roc_str = f"{roc_auc:.4f}" if roc_auc else "N/A"
    print(f"    Test Accuracy: {test_acc:.4f}  |  F1 Weighted: {test_f1_w:.4f}  |  ROC-AUC: {roc_str}")

    # Full classification report — use labels= to handle classes absent from test split
    present_labels = sorted(set(y_test) | set(y_test_pred))
    present_names  = [class_names[i] for i in present_labels]
    report = classification_report(
        y_test, y_test_pred,
        labels=present_labels,
        target_names=present_names,
        output_dict=True
    )
    all_reports[name] = report

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    results[name]['confusion_matrix'] = cm
    results[name]['model']            = model

# ── 5. CROSS-VALIDATION (Train set only) ─────────────────────────────────────

print("\n" + "-" * 60)
print("5-FOLD CROSS VALIDATION (on training set)")
print("-" * 60)

cv_results = {}
for name, model in models.items():
    Xtr = X_train_s if name == "Logistic Regression" else X_train
    cv  = cross_val_score(model, Xtr, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = {'cv_mean': round(cv.mean(), 4), 'cv_std': round(cv.std(), 4)}
    print(f"  {name:<25} CV F1: {cv.mean():.4f} (+/- {cv.std():.4f})")

# ── 6. FEATURE IMPORTANCE (Random Forest & XGBoost) ───────────────────────────

rf_importance = pd.DataFrame({
    'feature':   FEATURE_COLS,
    'importance': results['Random Forest']['model'].feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

xgb_importance = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': results['XGBoost']['model'].feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\n" + "-" * 60)
print("TOP 10 FEATURES — Random Forest")
print("-" * 60)
print(rf_importance.head(10).to_string(index=False))

print("\n" + "-" * 60)
print("TOP 10 FEATURES — XGBoost")
print("-" * 60)
print(xgb_importance.head(10).to_string(index=False))

# ── 7. SUMMARY TABLE ─────────────────────────────────────────────────────────

summary = pd.DataFrame([
    {
        'Model':        name,
        'Val Accuracy': val_results[name]['val_accuracy'],
        'Val F1':       val_results[name]['val_f1'],
        'Test Accuracy':results[name]['accuracy'],
        'Test F1 (weighted)': results[name]['f1_weighted'],
        'Test F1 (macro)':    results[name]['f1_macro'],
        'ROC-AUC':      results[name]['roc_auc'],
        'CV F1 Mean':   cv_results[name]['cv_mean'],
        'CV F1 Std':    cv_results[name]['cv_std'],
    }
    for name in models.keys()
])

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))

# ── 8. PER-CLASS REPORT (best model) ─────────────────────────────────────────

best_model_name = summary.loc[summary['Test F1 (weighted)'].idxmax(), 'Model']
print(f"\nBest model: {best_model_name}")

best_report_df = pd.DataFrame(all_reports[best_model_name]).T.reset_index()
best_report_df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
best_report_df = best_report_df[~best_report_df['Class'].isin(['accuracy', 'macro avg', 'weighted avg'])]

print(f"\nPer-class report ({best_model_name}):")
print(best_report_df.to_string(index=False))

# ── 9. CONFUSION MATRIX TABLE ────────────────────────────────────────────────

best_model    = results[best_model_name]['model']
Xts_best      = X_test_s if best_model_name == "Logistic Regression" else X_test
y_test_pred      = best_model.predict(Xts_best)
present_labels   = sorted(set(y_test) | set(y_test_pred))
present_names    = [class_names[i] for i in present_labels]
cm               = confusion_matrix(y_test, y_test_pred, labels=present_labels)
cm_df            = pd.DataFrame(cm, index=present_names, columns=present_names)

# ── 10. GROUND TRUTH EVALUATION (manual_labels.csv) ──────────────────────────

from src.config import DATA_DIR as _DATA_DIR

_manual_path = _DATA_DIR / "manual_labels.csv"
_gt_f1 = None

if _manual_path.exists():
    _manual = pd.read_csv(_manual_path)
    _manual = _manual[_manual['category'].astype(str).str.strip() != ''].copy()
    _manual = _manual[_manual['category'].isin(valid_cats)].copy()

    if len(_manual) >= 10:
        print("\n" + "-" * 60)
        print("GROUND TRUTH EVALUATION (manual_labels.csv)")
        print("-" * 60)

        # Re-use feature engineering from the main df by matching on description
        _gt_merged = _manual.merge(
            df[['description', 'date_operation'] + FEATURE_COLS + ['category']].rename(
                columns={'category': '_cat_orig'}
            ),
            on='description', how='inner'
        ).drop_duplicates(subset='description')

        if len(_gt_merged) >= 5:
            _Xgt = _gt_merged[FEATURE_COLS].values
            _ygt = le.transform(_gt_merged['category'])

            _Xgt_s = scaler.transform(_Xgt)
            _gt_results = {}
            for _name, _model in models.items():
                _Xin = _Xgt_s if _name == "Logistic Regression" else _Xgt
                _ypred = _model.predict(_Xin)
                _f1 = f1_score(_ygt, _ypred, average='weighted', zero_division=0)
                _gt_results[_name] = _f1
                print(f"  {_name:<25} Ground Truth F1 (weighted): {_f1:.4f}  (n={len(_ygt)})")

            _gt_f1 = _gt_results.get(best_model_name)
            print(f"\n  Best model ({best_model_name}) ground truth F1: {_gt_f1:.4f}")
        else:
            print(f"  Only {len(_gt_merged)} labeled rows matched feature matrix — need >= 5, skipping.")
    else:
        print(f"\n[Ground truth] {len(_manual)} labeled rows in manual_labels.csv — need >= 10, skipping evaluation.")
else:
    print("\n[Ground truth] data/manual_labels.csv not found — skipping. Label it and re-run to get ground truth F1.")

# ── 11. PERSIST MODEL ARTIFACTS ──────────────────────────────────────────────

from src.model_store import save_artifacts, data_hash as _data_hash

_metrics = {
    "best_model":    best_model_name,
    "test_accuracy": results[best_model_name]['accuracy'],
    "f1_weighted":   results[best_model_name]['f1_weighted'],
}
if _gt_f1 is not None:
    _metrics["gt_f1_weighted"] = round(_gt_f1, 4)

save_artifacts(
    "category_classifier",
    {
        "models":        {n: r['model'] for n, r in results.items()},
        "scaler":        scaler,
        "label_encoder": le,
        "feature_cols":  FEATURE_COLS,
        "class_names":   list(class_names),
    },
    metrics=_metrics,
    data_hash=_data_hash(df),
)

# ── 11. SAVE TO EXCEL ─────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    # Model comparison
    summary.to_excel(writer, sheet_name='Model Comparison', index=False)

    # Per-class reports for each model
    for name in models.keys():
        report_df = pd.DataFrame(all_reports[name]).T.reset_index()
        report_df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        sheet_name = name[:28] + ' Report'
        report_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Confusion matrix of best model
    cm_df.to_excel(writer, sheet_name='Confusion Matrix')

    # Feature importances
    rf_importance.to_excel(writer, sheet_name='RF Feature Importance', index=False)
    xgb_importance.to_excel(writer, sheet_name='XGB Feature Importance', index=False)

    # Cross-validation results
    cv_df = pd.DataFrame(cv_results).T.reset_index()
    cv_df.columns = ['Model', 'CV F1 Mean', 'CV F1 Std']
    cv_df.to_excel(writer, sheet_name='Cross Validation', index=False)

    # Ground truth evaluation (if available)
    if _gt_f1 is not None and '_gt_results' in dir():
        gt_df = pd.DataFrame([
            {'Model': n, 'Ground Truth F1 (weighted)': round(v, 4)}
            for n, v in _gt_results.items()
        ])
        gt_df.to_excel(writer, sheet_name='Ground Truth Eval', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(summary,         "model_comparison")
write_table(rf_importance,   "rf_feature_importance")
write_table(xgb_importance,  "xgb_feature_importance")
write_table(cv_df,           "cross_validation")
write_table(cm_df.reset_index().rename(columns={"index": "Class"}), "confusion_matrix")

log.info("Training complete -> %s", OUTPUT_EXCEL)
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Results saved to: {OUTPUT_EXCEL}")
print(f"\nSheets:")
print(f"  - Model Comparison")
print(f"  - Logistic Regression Report")
print(f"  - Random Forest Report")
print(f"  - XGBoost Report")
print(f"  - Confusion Matrix (best model: {best_model_name})")
print(f"  - RF Feature Importance")
print(f"  - XGB Feature Importance")
print(f"  - Cross Validation")
