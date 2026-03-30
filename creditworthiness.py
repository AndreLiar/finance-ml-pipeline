"""
Bank Creditworthiness Scoring — Simulating what a bank does
=============================================================
Features engineered (per month):
  - DSCR             : Debt Service Coverage Ratio = income / total_debt_payments
  - savings_rate     : (income - spend) / income
  - expense_volatility: std deviation of monthly spend
  - overdraft_freq   : % of months where spend > income
  - income_stability : coefficient of variation of monthly income
  - avg_monthly_income
  - avg_monthly_spend
  - max_single_tx    : largest single transaction
  - essential_ratio  : essential spend (groceries, rent, insurance) / total spend
  - discretionary_ratio: non-essential spend / total spend

Target:
  LOW_RISK   — financially healthy
  MEDIUM_RISK — some concerns
  HIGH_RISK   — significant financial stress

Models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - VotingClassifier (ensemble of all 3)  ← main deliverable

Output:
  data/creditworthiness_results.xlsx
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, VotingClassifier
from sklearn.metrics            import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

from config import (
    FEATURES_XLSX           as INPUT_EXCEL,
    LIVRET_A_XLSX           as INPUT_LIVRET,
    CREDITWORTHINESS_XLSX   as OUTPUT_EXCEL,
)
from db import read_table, write_table, table_exists
from logger import get_logger

log = get_logger(__name__)

print("=" * 60)
print("BANK CREDITWORTHINESS SCORING")
print("=" * 60)

# ── 1. LOAD & BUILD MONTHLY PROFILE ───────────────────────────────────────────

if table_exists("features_full"):
    df = read_table("features_full", parse_dates=["date_operation"])
else:
    df = pd.read_excel(INPUT_EXCEL, sheet_name="Full Data")
    df['date_operation'] = pd.to_datetime(df['date_operation'])
df['year_month'] = df['date_operation'].dt.to_period('M').astype(str)

# Load Livret A data for true income picture
livret_income = {}           # month -> salary deposited into Livret A
livret_transfers_in = {}     # month -> amount Livret A sent to Compte Cheques (funds spending)
if INPUT_LIVRET.exists():
    livret_monthly = pd.read_excel(INPUT_LIVRET, sheet_name="Monthly Livret A")
    for _, row in livret_monthly.iterrows():
        ym = row['year_month']
        livret_income[ym]       = row['savings_deposits']      # real salary
        livret_transfers_in[ym] = row['transfers_to_checking'] # funding the checking account
    print(f"Livret A income loaded for {len(livret_income)} months")
    # Estimate average monthly salary for gap months (Feb-Aug 2025 not in statements)
    known_salaries = [v for v in livret_income.values() if v > 0]
    avg_salary = np.mean(known_salaries) if known_salaries else 0.0
    print(f"  Known avg monthly salary: EUR{avg_salary:,.0f}")
else:
    avg_salary = 0.0
    print("No Livret A data found — income will be Compte Cheques credits only")

print(f"\nLoaded {len(df)} transactions across {df['year_month'].nunique()} months")

# ── 2. ENGINEER CREDITWORTHINESS FEATURES PER MONTH ──────────────────────────

ESSENTIAL_CATS     = {'GROCERIES', 'INSURANCE', 'RENT', 'UTILITIES', 'HEALTH', 'TELECOM'}
DISCRETIONARY_CATS = {'RESTAURANTS', 'ENTERTAINMENT', 'SHOPPING', 'TRAVEL', 'AI_SERVICES'}
DEBT_CATS          = {'INSURANCE', 'TELECOM', 'BANKING_FEES', 'UTILITIES'}

monthly_rows = []

for ym, grp in df.groupby('year_month'):

    debits  = grp[grp['type'] == 'DEBIT']
    credits = grp[grp['type'] == 'CREDIT']

    # ── True income logic ───────────────────────────────────────────────────
    # Compte Cheques CREDIT rows = Livret A transfers + occasional direct credits.
    # We separate "real salary" (Livret A deposits) from "internal transfers"
    # (Livret A → Checking, which just fund spending, not new income).
    #
    # For months WITH Livret A statement data: use savings_deposits as income.
    # For months WITHOUT Livret A data but with checking transfers: use transfers
    #   as a lower bound on income (salary was received; some went to Livret A, rest
    #   came to checking — we don't know the split, so treat transfers as income).
    # For pre-Livret-A months (before Sep 2024): use Compte Cheques credits as-is.

    checking_credits = credits['credit'].sum()
    livret_transfers = livret_transfers_in.get(ym, 0.0)  # what Livret A sent us

    if ym in livret_income and livret_income[ym] > 0:
        # We have actual salary data — Livret A deposits = gross salary saved
        # Total income = salary + anything directly credited to checking (not from Livret A)
        direct_credits = max(0, checking_credits - livret_transfers)
        income = livret_income[ym] + direct_credits
    elif ym >= '2024-09':
        # Gap months (Feb–Aug 2025): Livret A exists but statement not available.
        # Checking credits ARE the Livret A transfers (spending money sent over).
        # Use avg_salary as income estimate since we know salary was received.
        income = avg_salary if avg_salary > 0 else checking_credits
    else:
        # Pre-Livret-A period: checking credits = all income
        income = checking_credits

    spend   = debits['debit'].sum()
    net     = income - spend

    # DSCR — Debt Service Coverage Ratio
    debt_payments = debits[debits['category'].isin(DEBT_CATS)]['debit'].sum()
    dscr = income / debt_payments if debt_payments > 0 else (10.0 if income > 0 else 0.0)

    # Savings rate (clipped to [-1, 1])
    savings_rate = (income - spend) / income if income > 0 else -1.0

    # Essential vs discretionary spend
    essential_spend     = debits[debits['category'].isin(ESSENTIAL_CATS)]['debit'].sum()
    discretionary_spend = debits[debits['category'].isin(DISCRETIONARY_CATS)]['debit'].sum()
    essential_ratio     = essential_spend / spend if spend > 0 else 0.0
    discretionary_ratio = discretionary_spend / spend if spend > 0 else 0.0

    # Overdraft flag
    overdraft = 1 if net < 0 else 0

    # Transaction frequency
    tx_count  = len(grp)
    avg_tx    = debits['debit'].mean() if len(debits) > 0 else 0.0
    max_tx    = debits['debit'].max()  if len(debits) > 0 else 0.0

    # Cash withdrawals ratio
    cash_spend  = debits[debits['category'] == 'CASH']['debit'].sum()
    cash_ratio  = cash_spend / spend if spend > 0 else 0.0

    # Transfer ratio (money sent abroad)
    transfer_spend = debits[debits['category'] == 'TRANSFER']['debit'].sum()
    transfer_ratio = transfer_spend / spend if spend > 0 else 0.0

    monthly_rows.append({
        'year_month':          ym,
        'income':              round(income, 2),
        'spend':               round(spend, 2),
        'net':                 round(net, 2),
        'dscr':                round(min(dscr, 20.0), 4),   # cap at 20x
        'savings_rate':        round(np.clip(savings_rate, -1, 1), 4),
        'overdraft':           overdraft,
        'essential_ratio':     round(essential_ratio, 4),
        'discretionary_ratio': round(discretionary_ratio, 4),
        'cash_ratio':          round(cash_ratio, 4),
        'transfer_ratio':      round(transfer_ratio, 4),
        'tx_count':            tx_count,
        'avg_tx_amount':       round(avg_tx, 2),
        'max_tx_amount':       round(max_tx, 2),
        'debt_payments':       round(debt_payments, 2),
    })

monthly_df = pd.DataFrame(monthly_rows).sort_values('year_month').reset_index(drop=True)

# ── 3. ROLLING / STABILITY FEATURES ──────────────────────────────────────────

monthly_df['expense_volatility'] = (
    monthly_df['spend'].rolling(3, min_periods=1).std().fillna(0)
)
monthly_df['income_stability'] = (
    monthly_df['income'].rolling(3, min_periods=1).std().fillna(0)
    / monthly_df['income'].rolling(3, min_periods=1).mean().replace(0, 1)
)
monthly_df['overdraft_freq'] = (
    monthly_df['overdraft'].rolling(6, min_periods=1).mean()
)
monthly_df['avg_3m_income'] = monthly_df['income'].rolling(3, min_periods=1).mean()
monthly_df['avg_3m_spend']  = monthly_df['spend'].rolling(3, min_periods=1).mean()
monthly_df['spend_trend']   = (
    monthly_df['spend'].diff().rolling(3, min_periods=1).mean().fillna(0)
)

print(f"\nMonthly feature matrix: {monthly_df.shape}")
print("\nSample monthly profile:")
print(monthly_df[['year_month','income','spend','dscr','savings_rate',
                   'overdraft','expense_volatility','overdraft_freq']].head(8).to_string(index=False))

# ── 4. CREDIT RISK LABEL (Rule-based, transparent) ────────────────────────────

def compute_credit_score(row):
    score = 50

    # DSCR — most important signal
    if row['dscr'] >= 3.0:    score += 20
    elif row['dscr'] >= 1.5:  score += 10
    elif row['dscr'] < 1.0:   score -= 15

    # Savings rate
    if row['savings_rate'] >= 0.10:   score += 15
    elif row['savings_rate'] >= 0.0:  score += 5
    elif row['savings_rate'] < -0.3:  score -= 15
    else:                             score -= 5

    # Overdraft frequency
    if row['overdraft_freq'] == 0:      score += 10
    elif row['overdraft_freq'] <= 0.3:  score -= 5
    else:                               score -= 15

    # Expense volatility
    if row['expense_volatility'] < 300:   score += 5
    elif row['expense_volatility'] > 900: score -= 8

    # Income stability
    if row['income_stability'] < 0.3:  score += 5
    elif row['income_stability'] > 1.0: score -= 8

    # Cash & transfer ratios
    if row['cash_ratio'] > 0.3:     score -= 5
    if row['transfer_ratio'] > 0.2: score -= 5

    return max(0, min(100, score))

monthly_df['credit_score'] = monthly_df.apply(compute_credit_score, axis=1)

# Use tercile-based labels so classes are balanced (each ~33%)
terciles = monthly_df['credit_score'].quantile([0.33, 0.66])
low_thresh  = terciles[0.33]
high_thresh = terciles[0.66]

def assign_credit_label(score):
    if score >= high_thresh: return 'LOW_RISK'
    elif score >= low_thresh: return 'MEDIUM_RISK'
    else:                     return 'HIGH_RISK'

monthly_df['credit_label'] = monthly_df['credit_score'].apply(assign_credit_label)

label_counts = monthly_df['credit_label'].value_counts()
print(f"\nCredit label distribution:\n{label_counts.to_string()}")

# ── 5. FEATURES & SPLIT ───────────────────────────────────────────────────────

CREDIT_FEATURES = [
    'dscr', 'savings_rate', 'overdraft_freq', 'expense_volatility',
    'income_stability', 'essential_ratio', 'discretionary_ratio',
    'cash_ratio', 'transfer_ratio', 'avg_tx_amount', 'max_tx_amount',
    'tx_count', 'avg_3m_income', 'avg_3m_spend', 'spend_trend',
    'debt_payments',
]

le = LabelEncoder()
X  = monthly_df[CREDIT_FEATURES].values
y  = le.fit_transform(monthly_df['credit_label'])
class_names = le.classes_

print(f"\nFeatures : {len(CREDIT_FEATURES)}")
print(f"Samples  : {len(X)} months")
print(f"Classes  : {list(class_names)}")

# Chronological split — last 6 months = test, rest = train.
# monthly_df is already sorted by year_month (see sort_values above).
# This prevents future months from leaking into training via rolling features.
n_test  = min(6, max(1, len(X) // 5))
n_train = len(X) - n_test

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

cutoff = monthly_df['year_month'].iloc[n_train]
print(f"\nChronological split — cutoff: {cutoff}")
print(f"  Train: {n_train} months | Test: {n_test} months (last {n_test} months)")

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 6. TRAIN MODELS ───────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("TRAINING MODELS")
print("-" * 60)

lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2,
                             random_state=42, class_weight='balanced', n_jobs=-1)
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     use_label_encoder=False, eval_metric='mlogloss',
                     random_state=42, verbosity=0)

# Ensemble — soft voting (average predicted probabilities)
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[1, 2, 2],   # RF and XGBoost weighted higher
)

models = {
    'Logistic Regression': (lr,       X_train_s, X_test_s),
    'Random Forest':       (rf,       X_train,   X_test),
    'XGBoost':             (xgb,      X_train,   X_test),
    'Ensemble (Voting)':   (ensemble, X_train_s, X_test_s),
}

results   = {}
all_reports = {}

for name, (model, Xtr, Xts) in models.items():
    print(f"\n  [{name}]")
    model.fit(Xtr, y_train)
    yp   = model.predict(Xts)
    acc  = accuracy_score(y_test, yp)
    f1w  = f1_score(y_test, yp, average='weighted')
    f1m  = f1_score(y_test, yp, average='macro')
    try:
        yprob = model.predict_proba(Xts)
        auc   = roc_auc_score(y_test, yprob, multi_class='ovr', average='weighted')
    except Exception:
        auc = None
    report = classification_report(y_test, yp, target_names=class_names, output_dict=True)
    cm     = confusion_matrix(y_test, yp)
    results[name] = {
        'accuracy': round(acc,4), 'f1_weighted': round(f1w,4),
        'f1_macro': round(f1m,4), 'roc_auc': round(auc,4) if auc else None,
        'cm': cm, 'model': model,
    }
    all_reports[name] = report
    auc_s = f"{auc:.4f}" if auc else "N/A"
    print(f"    Accuracy: {acc:.4f} | F1 Weighted: {f1w:.4f} | ROC-AUC: {auc_s}")

# ── 7. CROSS-VALIDATION ───────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("5-FOLD CROSS-VALIDATION")
print("-" * 60)

cv_results = {}
for name, (model, Xtr, _) in models.items():
    cv = cross_val_score(model, Xtr, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = {'cv_mean': round(cv.mean(),4), 'cv_std': round(cv.std(),4)}
    print(f"  {name:<25} CV F1: {cv.mean():.4f} (+/- {cv.std():.4f})")

# ── 8. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("-" * 60)

fi_df = pd.DataFrame({
    'Feature':    CREDIT_FEATURES,
    'Importance': rf.feature_importances_,
}).sort_values('Importance', ascending=False)

for _, row in fi_df.head(10).iterrows():
    print(f"  {row['Feature']:<25} {row['Importance']:.4f}")

# ── 9. SUMMARY ────────────────────────────────────────────────────────────────

summary = pd.DataFrame([
    {
        'Model':        name,
        'Accuracy':     r['accuracy'],
        'F1 Weighted':  r['f1_weighted'],
        'F1 Macro':     r['f1_macro'],
        'ROC-AUC':      r['roc_auc'] if r['roc_auc'] else 'N/A',
        'CV F1 Mean':   cv_results[name]['cv_mean'],
        'CV F1 Std':    cv_results[name]['cv_std'],
    }
    for name, r in results.items()
])

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))

# ── 10. MONTHLY CREDIT PROFILE ────────────────────────────────────────────────

print("\n" + "-" * 60)
print("MONTHLY CREDIT PROFILE (last 12 months)")
print("-" * 60)

profile_cols = ['year_month','income','spend','dscr','savings_rate',
                'overdraft_freq','expense_volatility','credit_score','credit_label']
print(monthly_df[profile_cols].tail(12).to_string(index=False))

# ── 11. PERSIST MODEL ARTIFACTS ───────────────────────────────────────────────

from model_store import save_artifacts, data_hash as _data_hash

_best_name = summary.loc[summary['Accuracy'].idxmax(), 'Model']
_best      = results[_best_name]
save_artifacts(
    "creditworthiness",
    {
        "ensemble":      results['Ensemble (Voting)']['model'],
        "rf":            results['Random Forest']['model'],
        "scaler":        scaler,
        "label_encoder": le,
        "feature_cols":  CREDIT_FEATURES,
        "class_names":   list(class_names),
        "monthly_df":    monthly_df,
    },
    metrics={
        "best_model":   _best_name,
        "accuracy":     _best['accuracy'],
        "f1_weighted":  _best['f1_weighted'],
        "roc_auc":      _best['roc_auc'] if _best['roc_auc'] else 0.0,
    },
    data_hash=_data_hash(monthly_df),
)

# ── 12. SAVE ──────────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    summary.to_excel(writer, sheet_name='Model Comparison', index=False)

    monthly_df.to_excel(writer, sheet_name='Monthly Credit Profile', index=False)

    for name, report in all_reports.items():
        rdf = pd.DataFrame(report).T.reset_index()
        rdf.columns = ['Class', 'Precision', 'Recall', 'F1', 'Support']
        rdf.to_excel(writer, sheet_name=name[:28] + ' Rpt', index=False)

    fi_df.to_excel(writer, sheet_name='Feature Importance', index=False)

    cv_df = pd.DataFrame(cv_results).T.reset_index()
    cv_df.columns = ['Model', 'CV F1 Mean', 'CV F1 Std']
    cv_df.to_excel(writer, sheet_name='Cross Validation', index=False)

    # Credit score explanation
    explain = pd.DataFrame({
        'Feature': CREDIT_FEATURES,
        'Description': [
            'Debt Service Coverage: income / debt payments (>1.5 = good)',
            'Savings rate: (income - spend) / income',
            'Overdraft frequency: % of last 6 months with deficit',
            'Expense volatility: rolling 3-month std of spend',
            'Income stability: coefficient of variation of income',
            'Essential spend ratio: groceries/insurance/health/telecom',
            'Discretionary spend ratio: restaurants/shopping/travel',
            'Cash withdrawal ratio: % of spend as cash',
            'Transfer ratio: % of spend sent as transfers',
            'Average transaction amount',
            'Largest single transaction',
            'Number of transactions per month',
            '3-month rolling average income',
            '3-month rolling average spend',
            'Spend trend: direction of spending (positive = increasing)',
            'Total fixed debt payments',
        ]
    })
    explain.to_excel(writer, sheet_name='Feature Definitions', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(summary,      "credit_model_comparison")
write_table(monthly_df,   "monthly_credit")
write_table(fi_df,        "credit_feature_importance")
write_table(cv_df,        "credit_cross_validation")
write_table(explain,      "credit_feature_definitions")
log.info("Creditworthiness scoring complete → %s", OUTPUT_EXCEL)

print(f"\nResults saved to: {OUTPUT_EXCEL}")
print("\nSheets:")
print("  - Model Comparison")
print("  - Monthly Credit Profile")
print("  - Per-model reports (LR, RF, XGBoost, Ensemble)")
print("  - Feature Importance")
print("  - Cross Validation")
print("  - Feature Definitions")
