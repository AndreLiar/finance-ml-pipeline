"""
synthetic_augmentation.py — Synthetic monthly record generation for cashflow forecasting.
==========================================================================================
Generates synthetic monthly records sampled from the empirical feature distributions
of the real monthly credit profile. Used ONLY for the cashflow forecaster — the model
most hurt by a small sample (typically ~30-42 months of real data).

Honest framing
--------------
This pipeline is trained on one person's financial history, not a general population
model. Synthetic data is generated purely by sampling from the empirical distributions
of the real data (bootstrap with perturbation). It reflects only the patterns present
in the original data — it does NOT add new information. Synthetic rows are flagged with
is_synthetic=True and EXCLUDED from all evaluation metrics.

Gap 1 fix: Synthetic augmentation increases the effective training set for the cashflow
forecaster from ~20-30 usable months to ~200+, improving lag/rolling-feature estimation.
Evaluation always uses real data only.

Usage:
    python synthetic_augmentation.py          # generates and saves data
    from synthetic_augmentation import generate_synthetic_months  # import in cashflow_forecast.py

Output:
    data/synthetic_monthly.xlsx   — 200 synthetic monthly rows, is_synthetic=True
    data/finance.db               — table 'synthetic_monthly'
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    CREDITWORTHINESS_XLSX   as INPUT_MONTHLY,
    FINANCE_DB,
    DATA_DIR,
)
from db     import read_table, write_table, table_exists
from logger import get_logger

log = get_logger(__name__)

# Output path
SYNTHETIC_XLSX = DATA_DIR / "synthetic_monthly.xlsx"

# Number of synthetic months to generate
N_SYNTHETIC = 200

# Columns to synthesise — continuous numeric columns from Monthly Credit Profile
# that are used as features in cashflow_forecast.py
SYNTH_COLUMNS = [
    'income', 'spend', 'overdraft',
    'dscr', 'savings_rate', 'overdraft_freq', 'expense_volatility',
    'income_stability', 'essential_ratio', 'discretionary_ratio',
    'cash_ratio', 'transfer_ratio', 'avg_tx_amount', 'max_tx_amount',
    'tx_count', 'avg_3m_income', 'avg_3m_spend', 'spend_trend', 'debt_payments',
]


def load_real_monthly() -> pd.DataFrame:
    """Load the real monthly credit profile from SQLite or .xlsx."""
    if table_exists("monthly_credit"):
        df = read_table("monthly_credit")
        log.info("Loaded %d real months from SQLite", len(df))
    elif INPUT_MONTHLY.exists():
        df = pd.read_excel(INPUT_MONTHLY, sheet_name='Monthly Credit Profile')
        log.info("Loaded %d real months from .xlsx", len(df))
    else:
        log.error("No monthly credit data found. Run creditworthiness.py first.")
        sys.exit(1)
    return df


def generate_synthetic_months(real_df: pd.DataFrame, n: int = N_SYNTHETIC,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic monthly records by:
      1. Bootstrap-sampling from real rows
      2. Adding small Gaussian noise to continuous columns
         (noise std = 5% of each column's std — preserves distribution shape
          while avoiding exact duplicates)

    Returns a DataFrame with is_synthetic=True on every row.
    Evaluation must NEVER use these rows.
    """
    rng = np.random.default_rng(seed)

    # Only use columns that actually exist in the real data
    available = [c for c in SYNTH_COLUMNS if c in real_df.columns]
    if not available:
        log.error("None of the expected numeric columns found in monthly data.")
        sys.exit(1)

    log.info("Synthesising from %d real months, columns: %s", len(real_df), available)

    real_subset = real_df[available].dropna()
    n_real = len(real_subset)

    if n_real < 3:
        log.warning("Only %d real months — synthetic data quality will be very low.", n_real)

    # Bootstrap: sample with replacement
    bootstrap_idx = rng.integers(0, n_real, size=n)
    synthetic = real_subset.iloc[bootstrap_idx].copy().reset_index(drop=True)

    # Add small Gaussian noise scaled to each column's std
    for col in available:
        col_std = real_subset[col].std()
        if col_std > 0:
            noise_std = 0.05 * col_std
            noise = rng.normal(0, noise_std, size=n)
            synthetic[col] = synthetic[col] + noise

    # Post-processing: clip negative values where semantically impossible
    non_negative_cols = [
        'income', 'spend', 'overdraft', 'expense_volatility',
        'avg_tx_amount', 'max_tx_amount', 'tx_count',
        'avg_3m_income', 'avg_3m_spend', 'debt_payments',
        'essential_ratio', 'discretionary_ratio', 'cash_ratio',
        'transfer_ratio', 'income_stability', 'overdraft_freq',
    ]
    for col in non_negative_cols:
        if col in synthetic.columns:
            synthetic[col] = synthetic[col].clip(lower=0)

    # Ratios must stay in [0, 1]
    ratio_cols = [
        'essential_ratio', 'discretionary_ratio', 'cash_ratio',
        'transfer_ratio', 'overdraft_freq', 'savings_rate',
    ]
    for col in ratio_cols:
        if col in synthetic.columns:
            synthetic[col] = synthetic[col].clip(0, 1)

    # Round tx_count to integer
    if 'tx_count' in synthetic.columns:
        synthetic['tx_count'] = synthetic['tx_count'].round().astype(int).clip(lower=1)

    # Generate synthetic year_month labels (continuing from last real month)
    last_real = real_df['year_month'].max() if 'year_month' in real_df.columns else '2024-01'
    try:
        start = pd.Period(last_real, freq='M') + 1
    except Exception:
        start = pd.Period('2024-02', freq='M')

    synthetic_periods = pd.period_range(start=start, periods=n, freq='M')
    synthetic.insert(0, 'year_month', synthetic_periods.astype(str))

    # Flag every row — this column is the contract: NEVER evaluate on synthetic rows
    synthetic['is_synthetic'] = True

    log.info("Generated %d synthetic monthly records (seed=%d)", len(synthetic), seed)
    return synthetic


def save_synthetic(synthetic_df: pd.DataFrame):
    """Write synthetic records to .xlsx and SQLite."""
    SYNTHETIC_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(SYNTHETIC_XLSX, engine='openpyxl') as writer:
        synthetic_df.to_excel(writer, sheet_name='Synthetic Monthly', index=False)
    log.info("Saved %d rows → %s", len(synthetic_df), SYNTHETIC_XLSX)

    write_table(synthetic_df, "synthetic_monthly")
    log.info("Saved → SQLite table 'synthetic_monthly'")


def main():
    log.info("=" * 60)
    log.info("SYNTHETIC AUGMENTATION")
    log.info("=" * 60)
    log.info("NOTE: Trained on one person's financial history.")
    log.info("      Synthetic data reflects only the empirical distributions")
    log.info("      of the real data and adds no new population-level signal.")
    log.info("      is_synthetic=True rows are EXCLUDED from evaluation.")
    log.info("=" * 60)

    real_df      = load_real_monthly()
    synthetic_df = generate_synthetic_months(real_df, n=N_SYNTHETIC)
    save_synthetic(synthetic_df)

    log.info("Done. %d synthetic months generated.", len(synthetic_df))
    log.info("Use in cashflow_forecast.py by importing generate_synthetic_months().")
    log.info("Always filter: df = df[~df['is_synthetic']] before evaluation.")


if __name__ == "__main__":
    main()
