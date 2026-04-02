"""
Step 2 — Data Preparation & Feature Engineering
Banque Populaire - Compte Cheques 23192700536

Produces:
  data/features.xlsx  — cleaned + engineered features ready for ML
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

from src.config import TRANSACTIONS_XLSX as INPUT_EXCEL, FEATURES_XLSX as OUTPUT_EXCEL
from src.db import read_table, write_table, table_exists
from src.logger import get_logger

log = get_logger(__name__)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────

print("=" * 55)
print("STEP 2 — DATA PREPARATION & FEATURE ENGINEERING")
print("=" * 55)

if table_exists("transactions"):
    df = read_table("transactions", parse_dates=["date_operation"])
else:
    df = pd.read_excel(INPUT_EXCEL, sheet_name="Transactions")
log.info("Loaded %d transactions", len(df))
print(f"\nLoaded {len(df)} transactions")

# ── Validate input before proceeding ──────────────────────────────────────────
from src.schemas import validate_transactions
df = validate_transactions(df, source="feature_engineering input")

# ── 2. CLEAN ──────────────────────────────────────────────────────────────────

# Drop duplicate rows from old-format parser (RECAPITULATIF lines)
df = df[~df['description'].str.contains('RECAPITULATIF', case=False, na=False)]

# Drop BOTH type rows (parsing artefacts from old format — mixed debit+credit)
df = df[df['type'] != 'BOTH']
df = df[df['type'] != 'UNKNOWN']

# Fill nulls
df['debit']  = df['debit'].fillna(0.0)
df['credit'] = df['credit'].fillna(0.0)

# Ensure amount is signed correctly
df['amount'] = df.apply(
    lambda r: r['credit'] if r['type'] == 'CREDIT' else -r['debit'],
    axis=1
)

# Clean description — strip pipe separators from old format
df['description'] = df['description'].str.replace(r'\s*\|\s*', ' ', regex=True).str.strip()

print(f"After cleaning: {len(df)} transactions")

# ── 3. SPENDING CATEGORY LABELS ───────────────────────────────────────────────
# Rule-based auto-labeling using keywords in description

CATEGORY_RULES = {
    # SAVINGS and INVESTMENT must come before SALARY — virements to Livret A / Trade Republic
    # would otherwise match the broad VIREMENT pattern in SALARY
    'SAVINGS':      [r'LIVRET', r'EPARGNE', r'VIREMENT INTERNE'],
    'INVESTMENT':   [r'TRADE REPUBLIC', r'TRADEREPUBLIC', r'DEGIRO', r'BOURSO',
                     r'INVESTISSEMENT', r'BOURSE'],
    'FINES':        [r'AMENDE\.GOUV', r'AMENDE GOV', r'AMENDEFR'],
    'FINANCE':      [r'3X 4X ONEY', r'ONEY\b', r'CETELEM', r'COFIDIS'],
    'RENT':         [r'LOYER', r'BAIL', r'PROPRIETE', r'IMMOBILIERE', r'IMMOBILIER\b'],
    'SALARY':       [r'VIREMENT', r'SALAIRE', r'LEOPOLD', r'PAIE', r'VIR\s+INST',
                     r'VIR INST'],
    'GROCERIES':    [r'CARREFOUR', r'LIDL', r'AUCHAN', r'MONOPRIX', r'SUPERMARCHE',
                     r'MARKE', r'CASINO', r'LECLERC', r'INTERMARCHE', r'SUPEFR',
                     r'SC-VS\s', r'SC\.', r'SUPE\d', r'NEW FRUITS', r'ACTION\s',
                     r'FOURNIL', r'PAUL KIOSQUE', r'BOULANGER', r'FRANPRIX'],
    'RESTAURANTS':  [r'KFC', r'MCDO', r'ANTALYA', r'GRILLE', r'YOGURT',
                     r'BURGER', r'PIZZA', r'RESTAURANT', r'KOKIES', r'DELICE',
                     r'BRASSERIE', r'CAFE', r'SNACK', r'VAPIANO', r'DEL ARTE',
                     r'BOSPHORE', r'SOGERES', r'PAK 786', r'KEBAB', r'SUSHI',
                     r'BRIOCHE', r'PAUL\s', r'SANDWI', r'TRAITEUR', r'CANTEEN',
                     r'SELF\s', r'RESTO', r'REPAS'],
    'TRANSPORT':    [r'IMAGINE R', r'NAVIGO', r'RATP', r'SNCF', r'UBER',
                     r'TAXI', r'PARKING', r'CARBURANT', r'TOTAL\s', r'BP\s',
                     r'FLIXBUS', r'BLABLACAR', r'PARK\s', r'PARK$', r'STATIONNEMENT',
                     r'ESSO', r'SHELL', r'AUTOROUTE', r'PEAGE', r'BOLT\s'],
    'INSURANCE':    [r'COTIS', r'CRISTAL', r'ASSURANCE', r'CONFORT', r'CNV',
                     r'TRINITY', r'TRINITYFR', r'RUMFIC'],
    'BANKING_FEES': [r'FRAIS', r'COMMISSION', r'AGIOS', r'INTERETS',
                     r'PERCEPTION', r'MIN PERCEPTION', r'FMIN', r'TRIMESTRE',
                     r'IMMEUBLFR', r'SIX\s'],
    'SHOPPING':     [r'ZARA', r'HM\b', r'PRIMARK', r'FNAC', r'AMAZON', r'SHOPPING',
                     r'VETEMENT', r'PARADIS', r'DEFENSE', r'TEMU', r'SHEIN',
                     r'ALIEXPRESS', r'DECATHLON', r'IKEA', r'DARTY', r'BOULANGER',
                     r'CASHKORNER', r'SC-COMFORT', r'COMFORT SERV'],
    'HEALTH':       [r'PHARMACIE', r'MEDECIN', r'DOCTEUR', r'SANTE', r'MUTUELLE',
                     r'CLINIQUE', r'HOPITAL', r'DENTISTE', r'OPTIQUE', r'SECU'],
    'TELECOM':      [r'ORANGE', r'SFR', r'BOUYGUES', r'FREE', r'MOBILE', r'INTERNET',
                     r'RECHARGE'],
    'CASH':         [r'RETRAIT', r'DAB', r'REM CHQ', r'CHEQUE'],
    'ENTERTAINMENT':[r'CINEMA', r'NETFLIX', r'SPOTIFY', r'GAMING', r'SPORT',
                     r'LOISIR', r'BetM', r'PARI', r'BET', r'GYM', r'FITNESS',
                     r'SALLE\s', r'SUNLIGHT', r'MUSCULATION', r'PISCINE',
                     r'THEATRE', r'CONCERT', r'DISNEY', r'PRIME VIDEO'],
    'UTILITIES':    [r'EDF', r'GDF', r'ENGIE', r'EAU\b', r'ELECTRICITE', r'GAZ',
                     r'COURBEN', r'COURBEX', r'COUDFR', r'IMMEUB'],
    'AI_SERVICES':  [r'OPENAI', r'CHATGPT', r'ANTHROPIC', r'CLAUDE', r'MIDJOURNEY',
                     r'HTTPSOPENAI', r'SUBSCUS'],
    'TRANSFER':     [r'TAPTAP', r'TAPTAP SEND', r'WERO', r'WESTERN UNION',
                     r'MONEYGRAM', r'WISE', r'ENVOI', r'HODAS', r'NYAFR',
                     r'NYA\s', r'KAIS\s'],
    'PAYPAL':       [r'PAYPAL'],
    'TRAVEL':       [r'HOTEL', r'AIRBNB', r'BOOKING', r'BERLIN', r'AMSTERDAM',
                     r'BRUXELLES', r'TALLINN', r'EUROPE', r'VOYAGE', r'VOL\s',
                     r'AIRPORT', r'AEROPORT'],
}

def assign_category(desc: str) -> str:
    desc_upper = desc.upper()
    for category, patterns in CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, desc_upper):
                return category
    return 'OTHER'

df['category'] = df['description'].apply(assign_category)

# ── 3b. APPLY GROUND TRUTH CORRECTIONS ───────────────────────────────────────
# Overrides rule-based labels with verified corrections from data/labels/.
# Pattern-based corrections fix systematic errors at scale (e.g. all Livret A
# transfers were SALARY, now SAVINGS). tx_id-level corrections fix individual
# transactions. Labels are read from data/labels/category_corrections.csv.
try:
    from src.pipeline.label_loader import apply_category_corrections
    n_before = len(df)
    df = apply_category_corrections(df)
    n_corrected = int(df.get('correction_applied', pd.Series([False]*len(df))).sum())
    log.info("Ground truth corrections applied: %d / %d transactions relabeled", n_corrected, n_before)
    print(f"Ground truth corrections applied: {n_corrected} transactions relabeled")
    # Drop the helper column before continuing
    if 'correction_applied' in df.columns:
        df = df.drop(columns=['correction_applied'])
    if 'tx_id' in df.columns:
        df = df.drop(columns=['tx_id'])
except Exception as _exc:
    log.warning("Could not apply ground truth corrections: %s", _exc)
    print(f"Warning: ground truth corrections skipped ({_exc})")

cat_counts = df['category'].value_counts()
print(f"\nCategory distribution (after corrections):\n{cat_counts.to_string()}")

# ── 4. DATE FEATURES ──────────────────────────────────────────────────────────

df['year']        = df['date_operation'].dt.year
df['month']       = df['date_operation'].dt.month
df['day']         = df['date_operation'].dt.day
df['day_of_week'] = df['date_operation'].dt.dayofweek   # 0=Mon, 6=Sun
df['week_of_year']= df['date_operation'].dt.isocalendar().week.astype(int)
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['quarter']     = df['date_operation'].dt.quarter
df['year_month']  = df['date_operation'].dt.to_period('M').astype(str)

# Start / mid / end of month
df['month_part'] = pd.cut(
    df['day'],
    bins=[0, 10, 20, 31],
    labels=['start', 'mid', 'end']
)

# ── 5. AMOUNT FEATURES ────────────────────────────────────────────────────────

df['abs_amount']     = df['amount'].abs()
df['log_amount']     = np.log1p(df['abs_amount'])   # log-scale for skewed amounts
df['is_round_number']= (df['abs_amount'] % 10 == 0).astype(int)  # 50, 100, 200...

# ── 6. ROLLING / AGGREGATED FEATURES (per transaction) ───────────────────────

df = df.sort_values('date_operation').reset_index(drop=True)

# Rolling 7-day and 30-day spend (sum of debits)
# Use expanding window on sorted index to handle duplicate dates
debit_df = df[['date_operation', 'debit']].copy()
debit_df = debit_df.set_index('date_operation').sort_index()

rolling_7d  = debit_df['debit'].rolling('7D').sum()
rolling_30d = debit_df['debit'].rolling('30D').sum()

# Group by date to get one value per date (handle duplicates)
r7  = rolling_7d.groupby(level=0).last()
r30 = rolling_30d.groupby(level=0).last()

df['rolling_7d_spend']  = df['date_operation'].map(r7).fillna(0)
df['rolling_30d_spend'] = df['date_operation'].map(r30).fillna(0)

# ── 7. MONTHLY AGGREGATES ─────────────────────────────────────────────────────

monthly = df.groupby('year_month').agg(
    monthly_income  = ('credit',    'sum'),
    monthly_spend   = ('debit',     'sum'),
    monthly_net     = ('amount',    'sum'),
    tx_count        = ('amount',    'count'),
    avg_tx_amount   = ('abs_amount','mean'),
    max_tx_amount   = ('abs_amount','max'),
).reset_index()

monthly['savings_rate'] = (
    (monthly['monthly_income'] - monthly['monthly_spend'])
    / monthly['monthly_income'].replace(0, np.nan)
).fillna(0).clip(-1, 1)

monthly['overdraft_risk'] = (monthly['monthly_net'] < 0).astype(int)

# Merge monthly features back
df = df.merge(monthly, on='year_month', how='left')

# ── 8. CREDITWORTHINESS SCORE (target for supervised ML) ─────────────────────
# Rule-based scoring to generate a label for training
# Score 0-100 -> mapped to Low / Medium / High

def credit_score(row):
    score = 50  # baseline

    # Positive signals
    if row['monthly_income'] > 1500:  score += 15
    if row['savings_rate']   > 0.1:   score += 10
    if row['monthly_net']    > 0:     score += 10

    # Negative signals
    if row['overdraft_risk'] == 1:    score -= 20
    if row['monthly_spend']  > row['monthly_income'] * 0.9: score -= 10
    if row['max_tx_amount']  > 500:   score -= 5

    return max(0, min(100, score))

df['credit_score']     = df.apply(credit_score, axis=1)
df['credit_risk_label']= pd.cut(
    df['credit_score'],
    bins=[0, 40, 70, 100],
    labels=['HIGH_RISK', 'MEDIUM_RISK', 'LOW_RISK'],
    include_lowest=True
)

print(f"\nCredit risk distribution:\n{df['credit_risk_label'].value_counts().to_string()}")

# ── 9. ENCODE CATEGORICALS ────────────────────────────────────────────────────

df['type_encoded']       = df['type'].map({'DEBIT': 0, 'CREDIT': 1})
df['category_encoded']   = df['category'].astype('category').cat.codes
df['month_part_encoded'] = df['month_part'].map({'start': 0, 'mid': 1, 'end': 2})

# ── 10. FINAL FEATURE MATRIX ──────────────────────────────────────────────────

ML_FEATURES = [
    # Date features
    'year', 'month', 'day', 'day_of_week', 'week_of_year',
    'is_weekend', 'quarter', 'month_part_encoded',
    # Amount features
    'abs_amount', 'log_amount', 'is_round_number',
    # Rolling features
    'rolling_7d_spend', 'rolling_30d_spend',
    # Monthly aggregates
    'monthly_income', 'monthly_spend', 'monthly_net',
    'tx_count', 'avg_tx_amount', 'max_tx_amount', 'savings_rate',
    # Encoded categoricals
    'type_encoded', 'category_encoded',
]

feature_matrix = df[ML_FEATURES + ['credit_score', 'credit_risk_label', 'category', 'description', 'date_operation', 'amount', 'type']]

# ── 11. SAVE TO EXCEL ─────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
    feature_matrix.to_excel(writer, sheet_name='Feature Matrix', index=False)
    df.to_excel(writer, sheet_name='Full Data', index=False)
    monthly.to_excel(writer, sheet_name='Monthly Aggregates', index=False)

    # Category summary
    cat_summary = df.groupby('category').agg(
        count       = ('amount', 'count'),
        total_spend = ('debit',  'sum'),
        avg_amount  = ('abs_amount', 'mean'),
    ).sort_values('total_spend', ascending=False).reset_index()
    cat_summary.to_excel(writer, sheet_name='Category Summary', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(feature_matrix, "feature_matrix")
write_table(df,             "features_full")
write_table(monthly,        "monthly_aggregates")
write_table(cat_summary,    "category_summary")
log.info("Feature engineering complete: %d transactions, %d features -> %s",
         len(df), len(ML_FEATURES), OUTPUT_EXCEL)

print(f"\n{'='*55}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*55}")
print(f"Transactions       : {len(df)}")
print(f"Features built     : {len(ML_FEATURES)}")
print(f"Output saved to    : {OUTPUT_EXCEL}")
print(f"\nSheets:")
print(f"  - Feature Matrix     : ML-ready features + labels")
print(f"  - Full Data          : All columns")
print(f"  - Monthly Aggregates : Month-level stats")
print(f"  - Category Summary   : Spend by category")
print(f"\nFeature list:\n  " + "\n  ".join(ML_FEATURES))
