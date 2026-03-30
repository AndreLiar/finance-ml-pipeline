"""
conftest.py — Shared pytest fixtures for the finance ML pipeline test suite.
"""

import datetime
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── Minimal synthetic transactions DataFrame ──────────────────────────────────

@pytest.fixture
def sample_transactions():
    """
    30 synthetic transactions covering both DEBIT and CREDIT types,
    multiple categories, and a plausible date range.
    Sufficient to exercise parse → feature pipeline logic without real PDFs.
    """
    rng = np.random.default_rng(42)
    n   = 30

    dates = pd.date_range("2024-01-05", periods=n, freq="3D")

    descriptions = [
        "VIREMENT SALAIRE LEOPOLD", "LIDL PARIS", "RATP NAVIGO", "KFC REPUBLIC",
        "COTIS CRISTAL CONFORT", "SFR TELECOM", "CARREFOUR MARKET", "UBER EATS",
        "SNCF BILLET", "VIREMENT REMBOURSEMENT", "MONOPRIX RUE DE RIVOLI",
        "RESTAURANT LE DOME", "TOTAL CARBURANT", "AMAZON MARKETPLACE",
        "NETFLIX ABONNEMENT", "PHARMACIE DU CENTRE", "VIREMENT LOYER",
        "BRICORAMA OUTILLAGE", "AIR FRANCE BILLET", "DECATHLON SPORT",
        "ORANGE MOBILE", "EDF ENERGIE", "FRAIS BANCAIRES TRIMESTRIEL",
        "RETRAIT DAB ESPECES", "VIREMENT AIDE FAMILLE", "LECLERC DRIVE",
        "ANTALYA KEBAB", "PARKING SAEMES", "MUTUELLE SANTE", "FNAC PRODUITS",
    ]

    types = (
        ['CREDIT'] * 3
        + ['DEBIT'] * 27
    )

    credits = [1800.0, 0.0, 0.0, 200.0] + [0.0] * 26
    debits  = [0.0, 0.0, 0.0, 0.0] + [
        25.0, 89.90, 12.60, 74.00, 16.00, 45.80, 9.99, 18.50,
        55.00, 130.00, 35.00, 22.40, 8.99, 120.00, 60.00, 9.90,
        50.00, 400.00, 15.00, 19.90, 250.00, 10.00, 70.00,
        35.00, 200.00, 180.00,
    ]

    # Pad/trim to exactly n rows
    credits = (credits + [0.0] * n)[:n]
    debits  = (debits  + [0.0] * n)[:n]
    types   = (types   + ['DEBIT'] * n)[:n]

    df = pd.DataFrame({
        'date_operation': dates,
        'description':    descriptions[:n],
        'type':           types,
        'debit':          debits,
        'credit':         credits,
        'amount':         [c - d for c, d in zip(credits, debits)],
    })

    df['date_operation'] = pd.to_datetime(df['date_operation'])
    return df


@pytest.fixture
def sample_monthly_profile():
    """
    6-month synthetic credit profile for creditworthiness / loan report tests.
    Values are plausible but do not correspond to real data.
    """
    months = pd.period_range("2024-01", periods=6, freq='M')
    rng = np.random.default_rng(7)

    records = []
    for m in months:
        income = rng.uniform(1600, 2200)
        spend  = rng.uniform(1200, 1900)
        debt   = rng.uniform(0, 200)
        records.append({
            'year_month':           str(m),
            'dscr':                 round(income / max(debt, 1), 2),
            'savings_rate':         round((income - spend) / income, 4),
            'overdraft_freq':       0.0,
            'expense_volatility':   round(float(rng.uniform(80, 400)), 1),
            'income_stability':     round(float(rng.uniform(0.05, 0.4)), 4),
            'essential_ratio':      round(float(rng.uniform(0.3, 0.6)), 4),
            'discretionary_ratio':  round(float(rng.uniform(0.1, 0.3)), 4),
            'cash_ratio':           round(float(rng.uniform(0.0, 0.2)), 4),
            'transfer_ratio':       round(float(rng.uniform(0.0, 0.15)), 4),
            'avg_tx_amount':        round(float(rng.uniform(30, 120)), 2),
            'max_tx_amount':        round(float(rng.uniform(200, 800)), 2),
            'tx_count':             int(rng.integers(15, 50)),
            'avg_3m_income':        round(income, 0),
            'avg_3m_spend':         round(spend, 0),
            'spend_trend':          round(float(rng.uniform(-80, 80)), 1),
            'debt_payments':        round(debt, 2),
            'credit_label':         'LOW_RISK',
            'credit_score':         int(rng.integers(55, 85)),
        })

    return pd.DataFrame(records)


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Temporary models directory for artifact persistence tests."""
    d = tmp_path / "models"
    d.mkdir()
    return d
