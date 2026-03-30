"""
config.py — Central configuration for all pipeline scripts.

All paths, account identifiers, and tunable constants live here.
Scripts import what they need instead of hardcoding values inline.
"""

from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Account identifiers ────────────────────────────────────────────────────────
ACCOUNT_CHEQUES = "23192700536"
ACCOUNT_LIVRET  = "24971411768"
ACCOUNT_OWNER   = "M Kanmegne Tabouguie Andre"

# ── Input: raw PDF directories ─────────────────────────────────────────────────
STATEMENTS_DIR = ROOT / "Documents/Documents Banque/Comptes" / f"Compte De Cheques - {ACCOUNT_CHEQUES}"
LIVRET_A_DIR   = ROOT / "documents/Documents Banque/Epargne Et Placements" / f"Livret A-part - {ACCOUNT_LIVRET}"

# ── Intermediate data files ────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"

TRANSACTIONS_XLSX        = DATA_DIR / "transactions.xlsx"
FEATURES_XLSX            = DATA_DIR / "features.xlsx"
LIVRET_A_XLSX            = DATA_DIR / "livret_a_transactions.xlsx"
MERGED_XLSX              = DATA_DIR / "merged_transactions.xlsx"
MODEL_RESULTS_XLSX       = DATA_DIR / "model_results.xlsx"
NLP_RESULTS_XLSX         = DATA_DIR / "nlp_results.xlsx"
CREDITWORTHINESS_XLSX    = DATA_DIR / "creditworthiness_results.xlsx"
CASHFLOW_RESULTS_XLSX    = DATA_DIR / "cashflow_results.xlsx"
ANOMALY_RESULTS_XLSX     = DATA_DIR / "anomaly_results.xlsx"
LOAN_REPORT_TXT          = DATA_DIR / "loan_report.txt"
PIPELINE_STATUS_JSON     = DATA_DIR / "pipeline_status.json"

# ── SQLite database ────────────────────────────────────────────────────────────
FINANCE_DB = DATA_DIR / "finance.db"

# ── Models directory ───────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"

# ── LLM settings ──────────────────────────────────────────────────────────────
# LLM: local Ollama only — personal financial data must not leave the machine
LLM_MODEL   = "mistral"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS  = 1800

# ── Data date bounds (for validation) ─────────────────────────────────────────
DATA_DATE_MIN = "2020-01-01"
DATA_DATE_MAX = "2028-01-01"
