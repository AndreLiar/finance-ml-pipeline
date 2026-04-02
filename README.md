# Personal Finance ML Pipeline

End-to-end machine learning pipeline for personal finance analysis. Processes French bank statement PDFs (Banque Populaire) into structured data, then applies multiple ML models for classification, anomaly detection, forecasting, and credit scoring. Culminates in an interactive Streamlit dashboard with an AI financial advisor chat powered by LangChain and Mistral (local).

> **Data privacy:** All processing is local. Bank statement PDFs and personal data never leave the machine. All LLM components use Ollama (local Mistral) — no external API calls.

---

## Project Structure

```
FinanceProjects/
├── src/
│   ├── config.py                        # All paths and constants — single source of truth
│   ├── db.py                            # SQLite read/write helpers (read_table, write_table)
│   ├── logger.py                        # Rotating file logger (logs/pipeline.log, 5MB × 3)
│   ├── model_store.py                   # Joblib artifact persistence (save/load + metadata JSON)
│   ├── schemas.py                       # Pandera validation schemas for DataFrames
│   │
│   ├── pipeline/
│   │   ├── parse_statements.py          # [Stage 1]  PDF → transactions.xlsx + SQLite
│   │   ├── parse_livret_a.py            # [Stage 1b] Livret A savings PDF parser (optional)
│   │   ├── feature_engineering.py       # [Stage 2]  Feature engineering + category labelling
│   │   ├── label_loader.py              # Ground truth label management (anomaly + category)
│   │   ├── train_models.py              # [Stage 3]  Category classifier (LR / RF / XGBoost)
│   │   ├── nlp_classifier.py            # [Stage 4]  TF-IDF text classifier
│   │   ├── creditworthiness.py          # [Stage 5]  Credit risk scoring ensemble
│   │   ├── loan_report.py               # [Stage 6]  LLM-generated loan report (Ollama/Mistral)
│   │   ├── cashflow_forecast.py         # [Stage 7]  Monthly spend forecasting (Ridge + SelectKBest)
│   │   ├── anomaly_detection.py         # [Stage 8]  Ensemble anomaly detection (IF + OCSVM + LOF)
│   │   ├── supervised_anomaly.py        # [Stage 8b] Supervised anomaly classifier (GBM on ground truth)
│   │   ├── retrain_with_labels.py       # Retrain category classifier on corrected labels
│   │   ├── visualize_results.py         # [Stage 9]  Static chart generation
│   │   ├── financial_advisor.py         # [Stage 10] LangChain RAG financial advisor (Ollama/Mistral)
│   │   ├── drift_check.py               # Data drift detection using Evidently
│   │   └── synthetic_augmentation.py    # Bootstrap synthetic monthly records for augmentation
│   │
│   ├── vectorstore/
│   │   ├── embedder.py                  # 3-tier embedding: nomic-embed-text → MiniLM → TF-IDF
│   │   ├── store.py                     # Qdrant local vector store + numpy cosine fallback
│   │   ├── retriever.py                 # semantic_search(), build_context(), store_status()
│   │   └── indexer.py                   # Indexes transactions, anomalies, summaries into Qdrant
│   │
│   ├── mcp/
│   │   └── finance_mcp_server.py        # FastMCP server — 10 tools over stdio + HTTP (port 8052)
│   │
│   ├── agents/
│   │   └── financial_advisor_agent.py   # Tool-calling financial advisor agent (Anthropic SDK)
│   │
│   ├── dashboard/
│   │   └── dashboard.py                 # Interactive Streamlit dashboard (12+ pages)
│   │
│   └── prompts/
│       ├── loan_report_v1.txt           # Versioned LLM prompt — loan report
│       └── financial_advisor_v1.txt     # Versioned LLM prompt — financial advisor chat
│
├── run_pipeline.py                      # Orchestrator: runs all stages, writes pipeline_status.json
├── migrate_xlsx_to_sqlite.py            # One-time migration from .xlsx to SQLite
├── claude_desktop_config.json           # Claude Desktop MCP server config
│
├── tests/
│   ├── conftest.py                      # Shared fixtures (synthetic DataFrames)
│   ├── test_utils.py                    # Unit tests (parse_amount, assign_category, validators)
│   ├── test_pipeline.py                 # Integration tests (schema contract, cleaning steps)
│   └── test_models.py                   # Regression/snapshot tests (joblib round-trip)
│
├── docs/
│   ├── gap.md                           # Production upgrade plan (phases 1–3)
│   ├── ML_System_BankData.md            # System design notes
│   └── nextsession.md                   # Session notes
│
├── data/                                # gitignored — generated outputs
│   ├── finance.db                       # SQLite database (all pipeline outputs)
│   ├── *.xlsx                           # Excel outputs (backwards-compat)
│   ├── charts/                          # Static PNG charts
│   ├── drift_reports/                   # Evidently HTML drift reports
│   ├── vectorstore/                     # Qdrant local vector store files
│   ├── labels/
│   │   ├── anomaly_labels.csv           # Ground truth: 44 labeled anomalies (10 real, 34 false pos.)
│   │   └── category_corrections.csv     # Ground truth: 57 tx-level + 10 pattern category corrections
│   └── pipeline_status.json            # Last pipeline run status (read by dashboard)
│
├── models/                              # gitignored — joblib model artifacts
├── logs/                                # gitignored — rotating pipeline.log
├── Documents/                           # gitignored — bank statement PDFs (personal data)
│
├── dvc.yaml                             # DVC pipeline definition (data + model versioning)
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama (required for LLM features)

```bash
# Install from https://ollama.com
ollama serve
ollama pull mistral
ollama pull nomic-embed-text   # optional — upgrades embedding quality
```

### 3. Place bank statement PDFs

```
Documents/Documents Banque/Comptes/Compte De Cheques - <account_number>/
```

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

Or run stages individually in order:

```bash
py -3 -m src.pipeline.parse_statements       # → data/transactions.xlsx
py -3 -m src.pipeline.feature_engineering    # → data/features.xlsx
py -3 -m src.pipeline.train_models           # → data/model_results.xlsx + models/
py -3 -m src.pipeline.nlp_classifier         # → data/nlp_results.xlsx + models/
py -3 -m src.pipeline.creditworthiness       # → data/creditworthiness_results.xlsx + models/
py -3 -m src.pipeline.loan_report            # → data/loan_report.txt (requires Ollama)
py -3 -m src.pipeline.cashflow_forecast      # → data/cashflow_results.xlsx + models/
py -3 -m src.pipeline.anomaly_detection      # → data/anomaly_results.xlsx + models/
py -3 -m src.pipeline.visualize_results      # → data/charts/
```

### 5. Build the vector store (enables semantic search + RAG)

```bash
py -3 -m src.vectorstore.indexer
```

### 6. Launch dashboard

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
| `creditworthiness.py` | Credit risk scoring (LOW / MEDIUM / HIGH_RISK) | Ensemble VotingClassifier (LR + RF + XGBoost) | `creditworthiness_results.xlsx` |
| `cashflow_forecast.py` | Monthly spend forecasting | Ridge + SelectKBest, RF, Gradient Boosting, XGBoost | `cashflow_results.xlsx` |
| `anomaly_detection.py` | Unusual transaction detection | Isolation Forest + One-Class SVM + LOF (≥2/3 consensus) | `anomaly_results.xlsx` |
| `supervised_anomaly.py` | Supervised anomaly classification | GradientBoostingClassifier on ground truth labels + augmented negatives | `supervised_anomaly_results.xlsx` |
| `retrain_with_labels.py` | Category retraining on corrected labels | RF (baseline) vs GBM (corrected) comparison | `corrected_model_results.xlsx` |
| `loan_report.py` | LLM-generated bank loan report | Mistral via Ollama (local) | `loan_report.txt` |
| `financial_advisor.py` | RAG financial advisor chat | LangChain LCEL + Mistral via Ollama (local) | real-time answers |

---

## Ground Truth Labels

The pipeline includes a ground truth labeling system that converts unsupervised outputs into properly validated data.

### Label files (`data/labels/`)

| File | Purpose |
|------|---------|
| `anomaly_labels.csv` | 44 labeled transactions: 10 genuine anomalies, 34 false positives from the unsupervised ensemble |
| `category_corrections.csv` | 57 tx-level category overrides + 10 pattern-based rules applied to all transactions |

### How corrections are applied

1. **Pattern rules** (`label_loader.py::get_pattern_corrections`) — regex applied to all transactions at feature engineering time. Fixes systematic errors at scale (e.g. Livret A transfers mis-labelled as SALARY).
2. **tx_id overrides** — stable 12-char MD5 hash of `date|description[:60]|amount` links individual corrections to specific transactions durably.
3. **Integration point** — `feature_engineering.py` calls `apply_category_corrections()` immediately after rule-based labelling, so all downstream scripts (training, dashboard, forecasting) receive corrected categories automatically.

### Supervised anomaly detection

`supervised_anomaly.py` replaces the unsupervised ensemble for scoring:
- 44 labeled examples + 200 high-confidence augmented negatives
- GradientBoostingClassifier with stratified cross-validation
- Top features: `abs_amount` (38%), `amount_z_in_cat` (17%), `category_enc` (12%)
- Flags ~11 transactions (vs 44 from unsupervised ensemble) — far fewer false positives

---

## Creditworthiness Scoring

`creditworthiness.py` engineers 19 features per month and trains a VotingClassifier (LR + RF + XGBoost) to label each month as LOW / MEDIUM / HIGH_RISK.

### Feature set

| Feature | Description |
|---------|-------------|
| `dscr` | Debt Service Coverage Ratio: income / fixed debt payments |
| `savings_rate` | (income − spend) / income |
| `overdraft_freq` | % of last 6 months where spend exceeded income |
| `expense_volatility` | Rolling 3-month std of monthly spend |
| `income_stability` | Coefficient of variation of monthly income |
| `essential_ratio` | Essential spend (groceries/rent/health/telecom) / total spend |
| `discretionary_ratio` | Non-essential spend / total spend |
| `cash_ratio` | Cash withdrawals / total spend |
| `transfer_ratio` | International transfers / total spend |
| `avg_3m_income` | 3-month rolling average income |
| `avg_3m_spend` | 3-month rolling average spend |
| `spend_trend` | Direction of spending over last 3 months |
| `debt_payments` | Total fixed obligation payments |
| `livret_net` | Livret A net monthly flow (positive = saving, negative = drawing down) |
| `livret_drawdown_freq` | % of last 6 months with net negative Livret A flow |
| `savings_buffer_ratio` | Cumulative Livret A net / avg monthly spend (months of runway) |

The Livret A features make the model aware of when savings are being consumed to cover a spending deficit — a signal invisible to models that only see the checking account.

---

## Vector Store & Semantic Search

Transactions, anomalies, and monthly summaries are embedded and stored in a local Qdrant vector database for semantic retrieval.

### Embedding backends (auto-detected, cascading fallback)

1. **nomic-embed-text** via Ollama (best quality, requires `ollama pull nomic-embed-text`)
2. **all-MiniLM-L6-v2** via sentence-transformers (384-dim, pure Python)
3. **TF-IDF** cosine similarity (offline fallback, no GPU/internet required)

### Collections

| Collection | Contents |
|-----------|----------|
| `transactions` | All 1,200+ transactions with amount, category, date, description |
| `anomalies` | Flagged anomalies from the unsupervised ensemble |
| `summaries` | Monthly and category-level aggregate summaries |

### Usage

```bash
# Index all data (run after feature_engineering.py)
py -3 -m src.vectorstore.indexer

# Force re-index
py -3 -m src.vectorstore.indexer --force
```

The financial advisor agent automatically retrieves relevant context before every LLM call (RAG).

---

## MCP Server (Claude Desktop Integration)

`src/mcp/finance_mcp_server.py` exposes the pipeline data as a Model Context Protocol server, allowing Claude Desktop to query your finances directly.

### Tools exposed

| Tool | Description |
|------|-------------|
| `get_credit_profile` | Current credit score, risk label, key metrics |
| `get_income_and_spend` | Income vs spend summary with Livret A breakdown |
| `get_cashflow_forecast` | Next month spend forecast with confidence |
| `get_top_spending_categories` | Top N categories by spend |
| `get_anomalies` | Recent flagged transactions |
| `get_monthly_trend` | Month-by-month income/spend trend |
| `evaluate_affordability` | Can I afford a given monthly expense? |
| `search_transactions` | Semantic search across all transactions |
| `get_pipeline_status` | Last pipeline run status |
| `get_anomaly_investigation` | Deep-dive on a specific flagged transaction |

### Setup (Claude Desktop)

Copy `claude_desktop_config.json` to:
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```bash
# Or run the HTTP server manually (bound to localhost only)
py -3 -m src.mcp.finance_mcp_server --http
# → http://127.0.0.1:8052
```

---

## AI Financial Advisor Chat (LangChain RAG)

A conversational AI advisor that answers natural language questions about your finances in French or English, grounded in your real data.

### How it works

```
User question
     │
     ▼
Semantic vector search (Qdrant) → relevant transactions retrieved
     │
     ▼
Intent detection (keyword routing)
     │
     ▼
Context builder — loads relevant xlsx/SQLite data as structured text
     │
     ▼
LangChain LCEL chain: PromptTemplate | OllamaLLM | StrOutputParser
     │
     ▼
Mistral (local, via Ollama)  ←  no data leaves the machine
     │
     ▼
Answer grounded in real numbers
```

### Supported question types

| Intent | Example questions |
|--------|------------------|
| Credit score | "Pourquoi mon score de crédit a baissé ?" / "What is my current credit risk?" |
| Anomaly | "Quelles sont mes transactions anormales ?" / "Show me suspicious transactions" |
| Cash flow | "Quelles sont mes prévisions pour le mois prochain ?" / "What is my cashflow forecast?" |
| Spending | "Quels sont mes postes de dépenses les plus importants ?" / "How much do I spend on groceries?" |
| Affordability | "Est-ce que je peux me permettre un loyer de 800€ ?" / "Can I afford a 600€ rent?" |
| Income | "Quel est mon revenu moyen ?" / "Show me my salary history" |
| General | Open-ended — returns a compact 3-month financial summary |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, pipeline status, monthly income vs spend |
| 📈 Spending Analysis | Rolling averages, category breakdown, heatmaps |
| 💼 Revenus & Employeurs | Salary history, employer breakdown |
| 🏦 Livret A & Savings | Savings account evolution, combined net worth |
| 🏠 Capacité d'Emprunt | Loan affordability simulation (CDI vs current) |
| 🚨 Anomaly Detection | Flagged transactions, monthly anomaly rate, model comparison |
| 🤖 ML Models | Category classifier results, feature importance, confusion matrix |
| 🔤 NLP Classifier | TF-IDF text classifier comparison |
| 🏦 Creditworthiness | Credit score timeline, risk label distribution |
| 📉 Cash Flow Forecast | Actual vs predicted spend, next month forecast |
| 📋 Loan Decision Report | LLM-generated bank loan report (Ollama/Mistral) |
| 🔍 Transaction Explorer | Searchable, filterable full transaction table |
| 💬 Conseiller Financier IA | LangChain RAG chat — ask questions about your finances |
| 🔌 MCP Server | Vector store status, MCP tool list, live tool tester |

---

## Data Persistence

All pipeline scripts write to **both** `.xlsx` and `data/finance.db` (SQLite). Reads prefer SQLite via `db.py::read_table()` with `.xlsx` fallback.

```bash
# One-time migration (after first full run):
python migrate_xlsx_to_sqlite.py
```

### SQLite tables

| Table | Source script |
|-------|--------------|
| `transactions` | `parse_statements.py` |
| `features_full` | `feature_engineering.py` |
| `feature_matrix` | `feature_engineering.py` |
| `monthly_aggregates` | `feature_engineering.py` |
| `model_comparison` | `train_models.py` |
| `monthly_credit` | `creditworthiness.py` |
| `cashflow_metrics` | `cashflow_forecast.py` |
| `cashflow_forecast` | `cashflow_forecast.py` |
| `anomaly_results` | `anomaly_detection.py` |
| `synthetic_monthly` | `synthetic_augmentation.py` |

---

## Testing

```bash
pytest tests/ -v
```

| Layer | File | What it tests |
|-------|------|---------------|
| Unit | `test_utils.py` | `parse_amount()`, `assign_category()`, `validate_report_numbers()`, `build_constraints()` |
| Integration | `test_pipeline.py` | Schema contracts, cleaning steps, pandera validation |
| Regression | `test_models.py` | Joblib round-trip, snapshot prediction stability, metadata sidecars |

---

## Production Hardening

| Phase | Gap | Fix |
|-------|-----|-----|
| 1 | Hardcoded paths | `config.py` — single source of truth |
| 1 | No validation | `schemas.py` — pandera DataFrame schemas |
| 1 | No error recovery | `run_pipeline.py` — orchestrator + status JSON |
| 2 | No model persistence | `model_store.py` — joblib artifacts + metadata |
| 2 | Temporal leakage | Chronological splits in all classifiers |
| 2 | No ground truth | `data/labels/` — anomaly + category label files applied at ingestion |
| 2 | Circular labels | Supervised retraining on corrected labels (`retrain_with_labels.py`) |
| 2 | .xlsx database | SQLite via `db.py` — all scripts migrated |
| 2 | Negative R² | SelectKBest + Ridge in `cashflow_forecast.py` |
| 2 | Livret A invisible | Livret A net flow features in creditworthiness model |
| 3 | No tests | `tests/` — 3-layer pytest suite |
| 3 | No drift detection | `drift_check.py` — Evidently DataDriftPreset |
| 3 | Prompt versioning | `src/prompts/` — versioned .txt templates |
| 3 | LLM hallucination | `validate_report_numbers()` in `loan_report.py` |
| 3 | LLM contradicts ML | `build_constraints()` — hard prompt constraints |
| 3 | No logging | `logger.py` — RotatingFileHandler across all scripts |
| 3 | No semantic search | `src/vectorstore/` — Qdrant local + MiniLM embeddings |
| 3 | No LLM tool access | `src/mcp/` — FastMCP server for Claude Desktop |

---

## Key Design Decisions

- **All inference is local** — Ollama/Mistral, no external API calls. Personal financial data never leaves the machine.
- **Ground truth at ingestion** — Category corrections are applied in `feature_engineering.py` so every downstream script sees clean labels automatically.
- **Stable tx_id** — MD5 hash of `date|description[:60]|amount` provides durable transaction identity across re-runs without a database primary key.
- **3-tier embedding fallback** — nomic-embed-text → MiniLM → TF-IDF ensures semantic search works even without Ollama or GPU.
- **Livret A as buffer, not pure savings** — The creditworthiness model now tracks net Livret A flow per month, detecting months where savings are consumed to cover a spending deficit.
- **Dual persistence** — Scripts write to both `.xlsx` and `data/finance.db`; reads prefer SQLite with `.xlsx` fallback for backwards compatibility.
- **Ensemble anomaly detection** — Majority vote across Isolation Forest + One-Class SVM + LOF reduces false positives; supervised follow-up cuts flags from 44 → ~11.
- **SelectKBest + Ridge** in `cashflow_forecast.py` — `k=min(10, n_train//2)` prevents overfitting when feature count exceeds training months.
- **Chronological train/test splits** — No shuffle across all classifiers to prevent temporal leakage from monthly aggregate features.
- **MCP bound to localhost** — HTTP transport uses `127.0.0.1:8052`, never `0.0.0.0`.

---

## Configuration

All paths and constants are in `src/config.py`. Edit there to change:

- PDF input directories
- Account identifiers
- LLM model / temperature / token limit
- Data date bounds for validation

---

## Stack

| Category | Libraries |
|----------|-----------|
| Data | pandas, numpy, openpyxl, SQLite |
| PDF parsing | pdfplumber |
| ML | scikit-learn, XGBoost |
| LLM / AI | LangChain (LCEL), langchain-ollama, Ollama, Mistral |
| Embeddings | sentence-transformers (MiniLM), nomic-embed-text (Ollama) |
| Vector store | qdrant-client (local file mode) |
| MCP | fastmcp, mcp |
| Validation | pandera, pytest |
| Dashboard | Streamlit, Plotly |
| Versioning | DVC |
| Drift | Evidently |
| Logging | Python RotatingFileHandler |
