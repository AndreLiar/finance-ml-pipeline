"""
NLP Spending Category Classifier
=================================
Uses TF-IDF on transaction descriptions + numeric features
to classify transactions into spending categories.

Models compared:
  - Baseline  : TF-IDF only + Logistic Regression
  - Intermediate: TF-IDF + Numeric features + Random Forest
  - Advanced  : TF-IDF + Numeric features + XGBoost

Outputs:
  data/nlp_results.xlsx  — metrics, per-class report, feature importance
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, VotingClassifier
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

from config import FEATURES_XLSX as INPUT_EXCEL, NLP_RESULTS_XLSX as OUTPUT_EXCEL
from db import read_table, write_table, table_exists
from model_store import save_artifacts, data_hash as compute_data_hash
from logger import get_logger

log = get_logger(__name__)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("NLP SPENDING CATEGORY CLASSIFIER")
print("=" * 60)

if table_exists("feature_matrix"):
    df = read_table("feature_matrix", parse_dates=["date_operation"])
else:
    df = pd.read_excel(INPUT_EXCEL, sheet_name="Feature Matrix")

# Drop rare classes
cat_counts = df['category'].value_counts()
valid_cats = cat_counts[cat_counts >= 5].index
df = df[df['category'].isin(valid_cats)].copy()
print(f"\nSamples : {len(df)}")
print(f"Classes : {sorted(df['category'].unique())}")

# ── 2. TEXT PREPROCESSING ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize transaction description for TF-IDF."""
    import re
    text = str(text).upper()
    # Remove transaction codes (alphanumeric 6+ chars like 'EMKZ9EI')
    text = re.sub(r'\b[A-Z0-9]{6,}\b', '', text)
    # Remove dates embedded in description (e.g. 111122)
    text = re.sub(r'\b\d{6}\b', '', text)
    # Remove card numbers
    text = re.sub(r'CB\*+\d+', '', text)
    # Remove postal codes / dept codes (2-5 digits)
    text = re.sub(r'\b\d{2,5}\b', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['description_clean'] = df['description'].apply(clean_text)
print(f"\nSample cleaned descriptions:")
for _, row in df[['description', 'description_clean', 'category']].head(5).iterrows():
    print(f"  [{row['category']}] {row['description_clean'][:60]}")

# ── 3. FEATURE MATRIX ────────────────────────────────────────────────────────

NUMERIC_COLS = [
    'abs_amount', 'log_amount', 'is_round_number',
    'month', 'day_of_week', 'is_weekend', 'quarter',
    'rolling_7d_spend', 'rolling_30d_spend',
    'monthly_income', 'monthly_spend', 'monthly_net',
    'tx_count', 'avg_tx_amount', 'savings_rate', 'type_encoded',
]

le = LabelEncoder()
y  = le.fit_transform(df['category'])
class_names = le.classes_

# TF-IDF on cleaned descriptions
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=500,     # top 500 terms
    min_df=2,             # must appear in at least 2 docs
    sublinear_tf=True,    # log-scale TF
)
X_text = tfidf.fit_transform(df['description_clean'])

# Numeric features (scaled)
scaler  = StandardScaler()
X_num   = scaler.fit_transform(df[NUMERIC_COLS].values)
X_num_s = csr_matrix(X_num)

# Combined feature matrix: TF-IDF + Numeric
X_combined = hstack([X_text, X_num_s])
X_text_only = X_text

print(f"\nTF-IDF vocabulary size : {X_text.shape[1]}")
print(f"Numeric features       : {len(NUMERIC_COLS)}")
print(f"Combined feature shape : {X_combined.shape}")

# ── 4. SPLIT ─────────────────────────────────────────────────────────────────

X_train_c, X_temp_c, y_train, y_temp = train_test_split(
    X_combined, y, test_size=0.30, random_state=42, stratify=y
)
X_val_c, X_test_c, y_val, y_test = train_test_split(
    X_temp_c, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Text-only splits for baseline
X_train_t = X_text[:X_train_c.shape[0]]
X_test_t  = X_text[X_train_c.shape[0] + X_val_c.shape[0]:]

# Rebuild proper splits for text-only (same indices)
indices = np.arange(len(y))
idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])

X_train_t = X_text[idx_train]
X_test_t  = X_text[idx_test]
X_train_c2 = X_combined[idx_train]
X_test_c2  = X_combined[idx_test]
y_train2   = y[idx_train]
y_test2    = y[idx_test]

print(f"\nSplit — Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

# ── 5. TRAIN MODELS ───────────────────────────────────────────────────────────

print("\n" + "-" * 60)
print("TRAINING MODELS")
print("-" * 60)

models = {
    "LR — TF-IDF only": (
        LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        X_train_t, X_test_t
    ),
    "LR — TF-IDF + Numeric": (
        LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
        X_train_c2, X_test_c2
    ),
    "Random Forest — TF-IDF + Numeric": (
        RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                               class_weight='balanced', n_jobs=-1),
        X_train_c2, X_test_c2
    ),
    "XGBoost — TF-IDF + Numeric": (
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      use_label_encoder=False, eval_metric='mlogloss',
                      random_state=42, verbosity=0),
        X_train_c2, X_test_c2
    ),
}

results = {}
for name, (model, Xtr, Xts) in models.items():
    print(f"\n  [{name}]")
    model.fit(Xtr, y_train2)
    yp   = model.predict(Xts)
    acc  = accuracy_score(y_test2, yp)
    f1w  = f1_score(y_test2, yp, average='weighted')
    f1m  = f1_score(y_test2, yp, average='macro')
    try:
        yprob = model.predict_proba(Xts)
        auc   = roc_auc_score(y_test2, yprob, multi_class='ovr', average='weighted')
    except Exception:
        auc = None
    report = classification_report(y_test2, yp, target_names=class_names, output_dict=True)
    results[name] = {
        'accuracy': round(acc, 4), 'f1_weighted': round(f1w, 4),
        'f1_macro': round(f1m, 4), 'roc_auc': round(auc, 4) if auc else None,
        'report': report, 'model': model,
    }
    auc_str = f"{auc:.4f}" if auc else "N/A"
    print(f"    Accuracy: {acc:.4f} | F1 Weighted: {f1w:.4f} | ROC-AUC: {auc_str}")

# ── 6. TOP TF-IDF TERMS PER CATEGORY ─────────────────────────────────────────

print("\n" + "-" * 60)
print("TOP TF-IDF TERMS PER CATEGORY (from LR coefficients)")
print("-" * 60)

lr_model   = results["LR — TF-IDF only"]['model']
vocab      = tfidf.get_feature_names_out()
top_terms  = {}

for i, cls in enumerate(class_names):
    coef = lr_model.coef_[i] if len(class_names) > 2 else lr_model.coef_[0]
    top_idx = np.argsort(coef)[-6:][::-1]
    top_terms[cls] = [vocab[j] for j in top_idx]
    print(f"  {cls:<18}: {', '.join(top_terms[cls])}")

# ── 7. COMPARISON SUMMARY ─────────────────────────────────────────────────────

summary = pd.DataFrame([
    {
        'Model':             name,
        'Accuracy':          r['accuracy'],
        'F1 Weighted':       r['f1_weighted'],
        'F1 Macro':          r['f1_macro'],
        'ROC-AUC':           r['roc_auc'] if r['roc_auc'] else 'N/A',
    }
    for name, r in results.items()
])

print("\n" + "=" * 60)
print("MODEL COMPARISON — NLP vs Numeric-only (previous)")
print("=" * 60)
print(summary.to_string(index=False))

# Previous best (numeric only from train_models.py)
print("\nPrevious best (numeric features only):")
print("  XGBoost — Accuracy: 0.5192 | F1 Weighted: 0.5128 | ROC-AUC: 0.8677")

best_name = summary.loc[summary['F1 Weighted'].idxmax(), 'Model']
best_f1   = summary['F1 Weighted'].max()
print(f"\nBest NLP model : {best_name}")
print(f"Best F1        : {best_f1:.4f}")

# ── 8. SAVE ───────────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:

    summary.to_excel(writer, sheet_name='Model Comparison', index=False)

    for name, r in results.items():
        sheet = name[:28] + ' Rpt'
        rdf = pd.DataFrame(r['report']).T.reset_index()
        rdf.columns = ['Class', 'Precision', 'Recall', 'F1', 'Support']
        rdf.to_excel(writer, sheet_name=sheet, index=False)

    # Top TF-IDF terms
    terms_df = pd.DataFrame(
        [(cls, ', '.join(terms)) for cls, terms in top_terms.items()],
        columns=['Category', 'Top TF-IDF Terms']
    )
    terms_df.to_excel(writer, sheet_name='Top TF-IDF Terms', index=False)

    # Vocabulary sample
    vocab_df = pd.DataFrame({'term': vocab})
    vocab_df.to_excel(writer, sheet_name='TF-IDF Vocabulary', index=False)

# ── Write to SQLite ────────────────────────────────────────────────────────────
write_table(summary, "nlp_model_comparison")

# ── Persist best NLP model artifacts ──────────────────────────────────────────
_best_name  = summary.loc[summary['F1 Weighted'].idxmax(), 'Model']
_best_model = results[_best_name]['model']
_best_f1    = float(summary['F1 Weighted'].max())
_best_acc   = float(summary.loc[summary['F1 Weighted'].idxmax(), 'Accuracy'])
save_artifacts(
    "nlp_classifier",
    {
        "model":         _best_model,
        "tfidf":         tfidf,
        "scaler":        scaler,
        "label_encoder": le,
        "numeric_cols":  NUMERIC_COLS,
        "best_name":     _best_name,
    },
    metrics={"f1_weighted": _best_f1, "accuracy": _best_acc},
    data_hash=compute_data_hash(df),
)

log.info("NLP classifier complete → %s", OUTPUT_EXCEL)

print(f"\nResults saved to: {OUTPUT_EXCEL}")
print("\nSheets: Model Comparison | Per-class Reports | Top TF-IDF Terms | Vocabulary")
