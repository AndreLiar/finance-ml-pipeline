# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML pipeline for personal finance analysis. Processes French bank statement PDFs (Banque Populaire) into structured data, then applies multiple ML models for classification, anomaly detection, forecasting, and credit scoring. Culminates in an interactive Streamlit dashboard.

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Scripts must be run in order — each stage depends on the previous stage's output files in `data/`:

```bash
# 1. Parse PDFs into transactions
python src/pipeline/parse_statements.py       # → data/transactions.xlsx
python parse_livret_a.py         # → data/livret_a_transactions.xlsx (optional)

# 2. Feature engineering
python src/pipeline/feature_engineering.py   # → data/features.xlsx

# 3. Train models (can run independently after step 2)
python src/pipeline/train_models.py           # → data/model_results.xlsx
python src/pipeline/nlp_classifier.py         # → data/nlp_results.xlsx
python src/pipeline/creditworthiness.py       # → data/creditworthiness_results.xlsx
python src/pipeline/loan_report.py            # → data/loan_report.txt + new sheet in creditworthiness_results.xlsx (requires Ollama + mistral)
python src/pipeline/cashflow_forecast.py      # → data/cashflow_results.xlsx
python src/pipeline/anomaly_detection.py      # → data/anomaly_results.xlsx

# 4. Generate static charts
python src/pipeline/visualize_results.py      # → data/charts/

# 5. Launch interactive dashboard
streamlit run src/dashboard/dashboard.py       # http://localhost:8501
```

## Architecture

### Data Flow
```
PDF Bank Statements → parse_statements.py → data/transactions.xlsx
                                          → feature_engineering.py → data/features.xlsx
                                                                    → train_models.py
                                                                    → nlp_classifier.py
                                                                    → creditworthiness.py
                                                                    → cashflow_forecast.py
                                                                    → anomaly_detection.py
                                                                    → dashboard.py
```

### Input Data
Bank statement PDFs live in `Documents/Documents Banque/`. Two PDF formats are supported:
- Old format (2022–2023): table-based layout
- New format (2024+): different structure

`parse_statements.py` handles both via separate regex/parsing paths.

### Feature Engineering (`feature_engineering.py`)
- Rule-based category assignment mapping description keywords → 25+ categories (SALARY, GROCERIES, TRANSPORT, etc.)
- Temporal features: year, month, day_of_week, week_of_year, quarter
- Transaction features: abs_amount, log_amount, is_round_number, rolling averages
- `data/features.xlsx` has 4 sheets: Feature Matrix, Full Data, Monthly Aggregates, Category Summary

### ML Models
| Module | Task | Algorithms |
|--------|------|-----------|
| `train_models.py` | Category classification | Logistic Regression, Random Forest, XGBoost |
| `nlp_classifier.py` | Text-based classification | TF-IDF + LR/RF/XGBoost |
| `creditworthiness.py` | Risk scoring (LOW/MEDIUM/HIGH_RISK) | Ensemble VotingClassifier (LR+RF+XGBoost) |
| `loan_report.py` | LLM-generated bank loan report from creditworthiness output | Mistral via Ollama (`ollama serve` must be running) |
| `cashflow_forecast.py` | Monthly spending regression | Ridge, RF, Gradient Boosting, XGBoost |
| `anomaly_detection.py` | Unusual transaction detection | Isolation Forest + One-Class SVM + LOF (flag if ≥2/3 agree) |

### Dashboard (`dashboard.py`)
Large (~103KB) Streamlit app. Re-trains models on load (cached with `@st.cache_data`). Sections: overview, model comparison, anomalies, credit scoring, cash flow forecasting, Livret A savings. Uses Plotly for interactive charts.

## Data Persistence

Each script writes to both `.xlsx` (backwards-compat) **and** `data/finance.db` (SQLite). Scripts prefer reading from SQLite if the table exists, falling back to `.xlsx`.

```bash
# One-time migration after first full pipeline run:
python migrate_xlsx_to_sqlite.py   # → data/finance.db

# Helpers:
# db.py        — read_table(), write_table(), table_exists()
# model_store.py — save_artifacts(), load_artifacts() for joblib models
```

SQLite table naming: `transactions`, `features_full`, `feature_matrix`, `monthly_aggregates`, `model_comparison`, `monthly_credit`, `cashflow_metrics`, `cashflow_forecast`, `anomaly_results`, etc. See `migrate_xlsx_to_sqlite.py` for full mapping.

## Key Design Decisions
- Scripts write to both `.xlsx` and `data/finance.db`; reads prefer SQLite via `db.py::read_table()` with `.xlsx` fallback.
- Model artifacts persisted via `model_store.py` (joblib) to `models/` — dashboard loads pre-trained models instead of retraining on every load.
- `anomaly_detection.py` uses an ensemble consensus approach (majority vote across 3 unsupervised models) to reduce false positives.
- `cashflow_forecast.py` uses `SelectKBest(f_regression, k=min(10, n_train//2))` before Ridge to prevent overfitting when feature count exceeds training months.
- Chronological (no-shuffle) train/test splits in all classifiers to prevent temporal leakage from monthly aggregate features.
