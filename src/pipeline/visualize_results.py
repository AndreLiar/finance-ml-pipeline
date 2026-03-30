"""
Step 4 - Evaluation & Visualizations
Produces 8 charts saved to data/charts/:
  1.  model_comparison.png         - Accuracy / F1 / ROC-AUC bar chart
  2.  confusion_matrix.png         - XGBoost confusion matrix heatmap
  3.  per_class_f1.png             - Per-class F1 score bar chart
  4.  feature_importance_rf.png    - Random Forest feature importances
  5.  feature_importance_xgb.png   - XGBoost feature importances
  6.  monthly_spend_income.png     - Monthly spend vs income line chart
  7.  category_spend.png           - Spending by category bar chart
  8.  spending_trends.png          - Rolling 30-day spend trend
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    confusion_matrix, f1_score, accuracy_score, roc_auc_score,
    classification_report
)
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_EXCEL = Path("data/features.xlsx")
RESULTS_EXCEL  = Path("data/model_results.xlsx")
CHARTS_DIR     = Path("data/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE   = "#1a1a2e"
ACCENT    = "#e94560"
BLUE      = "#0f3460"
GREEN     = "#16213e"
FONT_CLR  = "#2d2d2d"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   '#cccccc',
    'grid.color':       '#eeeeee',
    'font.family':      'DejaVu Sans',
})

print("=" * 60)
print("STEP 4 - EVALUATION & VISUALIZATIONS")
print("=" * 60)

# ── Re-train models (needed for predictions) ──────────────────────────────────
df = pd.read_excel(FEATURES_EXCEL, sheet_name="Feature Matrix")
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 5].index
df         = df[df['category'].isin(valid_cats)].copy()

FEATURE_COLS = [
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'is_weekend', 'quarter', 'month_part_encoded',
    'abs_amount', 'log_amount', 'is_round_number',
    'rolling_7d_spend', 'rolling_30d_spend',
    'monthly_income', 'monthly_spend', 'monthly_net',
    'tx_count', 'avg_tx_amount', 'max_tx_amount', 'savings_rate',
    'type_encoded',
]

X  = df[FEATURE_COLS].values
le = LabelEncoder()
y  = le.fit_transform(df['category'])
class_names = le.classes_

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr  = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
rf  = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                              random_state=42, class_weight='balanced', n_jobs=-1)
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8,
                     use_label_encoder=False, eval_metric='mlogloss',
                     random_state=42, verbosity=0)

print("\nRe-training models for visualization...")
lr.fit(X_train_s, y_train);  print("  Logistic Regression done")
rf.fit(X_train,   y_train);  print("  Random Forest done")
xgb.fit(X_train,  y_train);  print("  XGBoost done")

models_map = {
    "Logistic Regression": (lr,  X_test_s),
    "Random Forest":       (rf,  X_test),
    "XGBoost":             (xgb, X_test),
}

metrics = {}
for name, (model, Xts) in models_map.items():
    yp    = model.predict(Xts)
    yprob = model.predict_proba(Xts)
    try:
        auc = roc_auc_score(y_test, yprob, multi_class='ovr', average='weighted')
    except Exception:
        auc = None
    metrics[name] = {
        'Accuracy':    accuracy_score(y_test, yp),
        'F1 Weighted': f1_score(y_test, yp, average='weighted'),
        'ROC-AUC':     auc,
    }

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Model Comparison (Accuracy / F1 / ROC-AUC)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating charts...")

model_names  = list(metrics.keys())
metric_names = ['Accuracy', 'F1 Weighted', 'ROC-AUC']
colors       = ['#0f3460', '#e94560', '#533483']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
w = 0.25

for i, (metric, color) in enumerate(zip(metric_names, colors)):
    vals = [metrics[m][metric] for m in model_names]
    bars = ax.bar(x + i * w, vals, w, label=metric, color=color, alpha=0.88, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x + w)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — Accuracy / F1 / ROC-AUC', fontsize=14, fontweight='bold', pad=15)
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "1_model_comparison.png", dpi=150)
plt.close()
print("  1. model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Confusion Matrix (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
y_pred_xgb = xgb.predict(X_test)
cm         = confusion_matrix(y_test, y_pred_xgb)
cm_norm    = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    cm_norm, annot=True, fmt='.0%', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    linewidths=0.5, linecolor='#dddddd',
    annot_kws={'size': 8}, ax=ax
)
ax.set_title('Confusion Matrix — XGBoost (normalised by true class)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual',    fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0,  fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "2_confusion_matrix.png", dpi=150)
plt.close()
print("  2. confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Per-Class F1 Score (all 3 models)
# ─────────────────────────────────────────────────────────────────────────────
per_class = {}
for name, (model, Xts) in models_map.items():
    report = classification_report(y_test, model.predict(Xts),
                                   target_names=class_names, output_dict=True)
    per_class[name] = {cls: report[cls]['f1-score']
                       for cls in class_names if cls in report}

pc_df = pd.DataFrame(per_class).reindex(class_names)

fig, ax = plt.subplots(figsize=(13, 7))
x  = np.arange(len(class_names))
w  = 0.26
for i, (name, color) in enumerate(zip(models_map.keys(), colors)):
    vals = [pc_df.loc[c, name] if c in pc_df.index else 0 for c in class_names]
    ax.bar(x + i * w, vals, w, label=name, color=color, alpha=0.88, edgecolor='white')

ax.set_xticks(x + w)
ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Per-Class F1 Score — All Models', fontsize=14, fontweight='bold', pad=15)
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "3_per_class_f1.png", dpi=150)
plt.close()
print("  3. per_class_f1.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Feature Importance (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
fi_rf = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(fi_rf.index, fi_rf.values, color='#0f3460', alpha=0.85, edgecolor='white')
for bar, v in zip(bars, fi_rf.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', fontsize=8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "4_feature_importance_rf.png", dpi=150)
plt.close()
print("  4. feature_importance_rf.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Feature Importance (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────
fi_xgb = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(fi_xgb.index, fi_xgb.values, color='#e94560', alpha=0.85, edgecolor='white')
for bar, v in zip(bars, fi_xgb.values):
    ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', fontsize=8)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance — XGBoost', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(CHARTS_DIR / "5_feature_importance_xgb.png", dpi=150)
plt.close()
print("  5. feature_importance_xgb.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — Monthly Spend vs Income
# ─────────────────────────────────────────────────────────────────────────────
monthly = pd.read_excel(FEATURES_EXCEL, sheet_name="Monthly Aggregates")
monthly = monthly.sort_values('year_month').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(monthly))
ax.bar(x, monthly['monthly_spend'],  label='Spend',  color='#e94560', alpha=0.75, width=0.4, align='edge')
ax.bar(x - 0.4, monthly['monthly_income'], label='Income', color='#0f3460', alpha=0.75, width=0.4, align='edge')
ax.plot(x, monthly['monthly_net'], color='#533483', marker='o', linewidth=2,
        markersize=5, label='Net Flow', zorder=5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels(monthly['year_month'], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Amount (EUR)', fontsize=12)
ax.set_title('Monthly Income vs Spend vs Net Flow', fontsize=14, fontweight='bold', pad=15)
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f} EUR'))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "6_monthly_spend_income.png", dpi=150)
plt.close()
print("  6. monthly_spend_income.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 7 — Spending by Category
# ─────────────────────────────────────────────────────────────────────────────
full_df  = pd.read_excel(FEATURES_EXCEL, sheet_name="Full Data")
cat_spend = (
    full_df[full_df['type'] == 'DEBIT']
    .groupby('category')['debit'].sum()
    .sort_values(ascending=True)
)

cmap   = plt.cm.get_cmap('tab20', len(cat_spend))
colors_cat = [cmap(i) for i in range(len(cat_spend))]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(cat_spend.index, cat_spend.values, color=colors_cat, edgecolor='white', alpha=0.9)
for bar, v in zip(bars, cat_spend.values):
    ax.text(v + 50, bar.get_y() + bar.get_height() / 2,
            f'{v:,.0f} EUR', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Total Spend (EUR)', fontsize=12)
ax.set_title('Total Spending by Category (Oct 2022 - Feb 2026)', fontsize=14, fontweight='bold', pad=15)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "7_category_spend.png", dpi=150)
plt.close()
print("  7. category_spend.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 8 — Rolling 30-Day Spend Trend
# ─────────────────────────────────────────────────────────────────────────────
trend_df = full_df[full_df['type'] == 'DEBIT'].copy()
trend_df['date_operation'] = pd.to_datetime(trend_df['date_operation'])
trend_df = trend_df.set_index('date_operation').sort_index()
daily    = trend_df['debit'].resample('D').sum()
rolling  = daily.rolling(30).mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(daily.index, daily.values, alpha=0.2, color='#0f3460', label='Daily Spend')
ax.plot(rolling.index, rolling.values, color='#e94560', linewidth=2.5,
        label='30-Day Rolling Avg')
ax.set_ylabel('Amount (EUR)', fontsize=12)
ax.set_title('Daily Spend with 30-Day Rolling Average', fontsize=14, fontweight='bold', pad=15)
ax.legend(framealpha=0.9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f} EUR'))
plt.tight_layout()
plt.savefig(CHARTS_DIR / "8_spending_trends.png", dpi=150)
plt.close()
print("  8. spending_trends.png")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"ALL CHARTS SAVED TO: {CHARTS_DIR.resolve()}")
print(f"{'='*60}")
print(f"""
CHARTS GENERATED:
  1. model_comparison.png       - Accuracy / F1 / ROC-AUC bar chart
  2. confusion_matrix.png       - XGBoost confusion matrix heatmap
  3. per_class_f1.png           - Per-class F1 score (all 3 models)
  4. feature_importance_rf.png  - Random Forest feature importances
  5. feature_importance_xgb.png - XGBoost feature importances
  6. monthly_spend_income.png   - Monthly spend vs income
  7. category_spend.png         - Total spend by category
  8. spending_trends.png        - 30-day rolling spend trend
""")
