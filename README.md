# Personal Finance ML Pipeline

End-to-end machine learning pipeline for personal finance analysis. Processes French bank statement PDFs (Banque Populaire) into structured data, then applies multiple ML models for classification, anomaly detection, forecasting, and credit scoring. Culminates in an interactive Streamlit dashboard.

> **Data privacy:** All processing is local. Bank statement PDFs and personal data never leave the machine. The LLM component uses Ollama (local Mistral) — no external API calls.

---

## Project Structure

```
FinanceProjects/
├── config.py                   # All paths and constants — single source of truth
├── db.py                       # SQLite read/write helpers
├── logger.py                   # Rotating file logger (logs/pipeline.log)
├── model_store.py              # Joblib artifact persistence (save/load trained models)
├── schemas.py                  # Pandera validation schemas for DataFrames
│
├── parse_statements.py         # [Stage 1] PDF → transactions.xlsx + SQLite
├── parse_livret_a.py           # [Stage 1b] Livret A savings PDF parser (optional)
├── feature_engineering.py      # [Stage 2] Feature engineering + category labelling
├── train_models.py             # [Stage 3] Category classifier (LR / RF / XGBoost)
├── nlp_classifier.py           # [Stage 4] TF-IDF text classifier
├── creditworthiness.py         # [Stage 5] Credit risk scoring ensemble
├── loan_report.py              # [Stage 6] LLM-generated loan report (Ollama/Mistral)
├── cashflow_forecast.py        # [Stage 7] Monthly spend forecasting (Ridge + SelectKBest)
├── anomaly_detection.py        # [Stage 8] Ensemble anomaly detection (IF + OCSVM + LOF)
├── visualize_results.py        # [Stage 9] Static chart generation
├── dashboard.py                # [Stage 10] Interactive Streamlit dashboard
│
├── run_pipeline.py             # Orchestrator: runs all stages, writes pipeline_status.json
├── migrate_xlsx_to_sqlite.py   # One-time migration from .xlsx to SQLite
├── drift_check.py              # Data drift detection using evidently
├── synthetic_augmentation.py   # Synthetic monthly records for cashflow augmentation
│
├── prompts/
│   └── loan_report_v1.txt      # Versioned LLM prompt template
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic DataFrames)
│   ├── test_utils.py           # Unit tests (parse_amount, assign_category, validators)
│   ├── test_pipeline.py        # Integration tests (schema contract, cleaning steps)
│   └── test_models.py          # Regression/snapshot tests (joblib round-trip)
├── docs/
│   ├── gap.md                  # Production upgrade plan (phases 1-3)
│   ├── ML_System_BankData.md   # System design notes
│   └── nextsession.md          # Session notes
│
├── data/                       # gitignored — generated outputs
│   ├── finance.db              # SQLite database (all pipeline outputs)
│   ├── *.xlsx                  # Excel outputs (backwards-compat)
│   ├── charts/                 # Static PNG charts
│   ├── drift_reports/          # Evidently HTML drift reports
│   └── pipeline_status.json    # Last pipeline run status (read by dashboard)
├── models/                     # gitignored — joblib model artifacts
├── logs/                       # gitignored — rotating pipeline.log
├── Documents/                  # gitignored — bank statement PDFs (personal data)
│
├── dvc.yaml                    # DVC pipeline definition (data + model versioning)
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place bank statement PDFs

Put your Banque Populaire PDFs in:
```
Documents/Documents Banque/Comptes/Compte De Cheques - <account_number>/
```

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

Or run stages individually in order:

```bash
python src/pipeline/parse_statements.py       # → data/transactions.xlsx
python src/pipeline/feature_engineering.py    # → data/features.xlsx
python src/pipeline/train_models.py           # → data/model_results.xlsx + models/
python src/pipeline/nlp_classifier.py         # → data/nlp_results.xlsx + models/
python src/pipeline/creditworthiness.py       # → data/creditworthiness_results.xlsx + models/
python src/pipeline/cashflow_forecast.py      # → data/cashflow_results.xlsx + models/
python src/pipeline/anomaly_detection.py      # → data/anomaly_results.xlsx + models/
python src/pipeline/loan_report.py            # → data/loan_report.txt (requires Ollama)
python src/pipeline/visualize_results.py      # → data/charts/
```

### 4. Launch dashboard

```bash
streamlit run src/dashboard/dashboard.py
```

Navigate to `http://localhost:8501`.

---

## ML Models

| Script | Task | Algorithms | Output |
|--------|------|-----------|--------|
| `train_models.py` | Spending category classification | Logistic Regression, Random Forest, XGBoost | `model_results.xlsx` |
| `nlp_classifier.py` | Text-based classification | TF-IDF + LR / RF / XGBoost | `nlp_results.xlsx` |
| `creditworthiness.py` | Credit risk scoring | Ensemble VotingClassifier (LR+RF+XGBoost) | `creditworthiness_results.xlsx` |
| `cashflow_forecast.py` | Monthly spend forecasting | Ridge + SelectKBest, RF, Gradient Boosting | `cashflow_results.xlsx` |
| `anomaly_detection.py` | Unusual transaction detection | Isolation Forest + One-Class SVM + LOF | `anomaly_results.xlsx` |
| `loan_report.py` | LLM-generated bank loan report | Mistral via Ollama (local) | `loan_report.txt` |

---

## Data Persistence

All pipeline scripts write to **both** `.xlsx` and `data/finance.db` (SQLite). Reads prefer SQLite via `db.py::read_table()` with `.xlsx` fallback.

```bash
# One-time migration (after first full run):
python migrate_xlsx_to_sqlite.py
```

---

## Testing

```bash
pytest tests/ -v
```

Three test layers:
- **Unit** (`test_utils.py`) — pure functions: `parse_amount()`, `assign_category()`, validators
- **Integration** (`test_pipeline.py`) — schema contracts, cleaning steps with synthetic fixtures
- **Regression** (`test_models.py`) — joblib round-trip, snapshot prediction stability

---

## Production Hardening (Phases 1–3)

See [`docs/gap.md`](docs/gap.md) for the full production upgrade plan. Implemented:

| Phase | Gap | Fix |
|-------|-----|-----|
| 1 | Hardcoded paths | `config.py` — single source of truth |
| 1 | No validation | `schemas.py` — pandera DataFrame schemas |
| 1 | No auth | `credentials.yaml` — streamlit-authenticator |
| 1 | No error recovery | `run_pipeline.py` — orchestrator + status JSON |
| 2 | No model persistence | `model_store.py` — joblib artifacts + metadata |
| 2 | Temporal leakage | Chronological splits in all classifiers |
| 2 | Circular labels | `data/manual_labels.csv` — ground truth eval set |
| 2 | .xlsx database | SQLite via `db.py` — all scripts migrated |
| 2 | Negative R² | SelectKBest + Ridge in `cashflow_forecast.py` |
| 3 | No tests | `tests/` — 3-layer pytest suite |
| 3 | No drift detection | `drift_check.py` — evidently DataDriftPreset |
| 3 | Prompt versioning | `prompts/loan_report_v1.txt` |
| 3 | LLM hallucination | `validate_report_numbers()` in `loan_report.py` |
| 3 | LLM contradicts ML | `build_constraints()` — hard prompt constraints |
| 3 | No logging | `logger.py` — RotatingFileHandler across all scripts |
| 3 | No data versioning | `dvc.yaml` — DVC pipeline definition |
| 3 | Small dataset | `synthetic_augmentation.py` — bootstrap augmentation |

---

## LLM Component (Loan Report)

Requires Ollama running locally with Mistral:

```bash
ollama serve
ollama pull mistral
python src/pipeline/loan_report.py
```

All inference is local — personal financial data is never sent to any external service.

---

## Configuration

All paths and constants are in `config.py`. Edit there to change:
- PDF input directories
- Account identifiers
- LLM model / temperature / token limit
- Data date bounds for validation
