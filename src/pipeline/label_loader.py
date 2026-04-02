"""
src/pipeline/label_loader.py — Ground truth label management

Reads data/labels/anomaly_labels.csv and data/labels/category_corrections.csv,
applies them to the feature matrix, and exposes helpers for the supervised
anomaly classifier and the corrected category trainer.

Labels are keyed on a stable tx_id (MD5 of date|description[:60]|amount).
New transactions that don't match any label are treated as unlabeled.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, FEATURES_XLSX

log = logging.getLogger(__name__)

LABELS_DIR        = DATA_DIR / "labels"
ANOMALY_LABELS    = LABELS_DIR / "anomaly_labels.csv"
CATEGORY_LABELS   = LABELS_DIR / "category_corrections.csv"


# ── Stable transaction ID ──────────────────────────────────────────────────────

def make_tx_id(date, description: str, amount: float) -> str:
    """Stable 12-char hash ID for a transaction. Matches indexer logic."""
    key = f"{date}|{str(description)[:60]}|{float(amount):.2f}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def add_tx_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add tx_id column to a DataFrame that has date_operation, description, abs_amount."""
    df = df.copy()
    df["tx_id"] = df.apply(
        lambda r: make_tx_id(r["date_operation"], r.get("description", ""), r.get("abs_amount", 0)),
        axis=1,
    )
    return df


# ── Anomaly labels ─────────────────────────────────────────────────────────────

def load_anomaly_labels() -> pd.DataFrame:
    """
    Load anomaly ground truth labels.

    Returns:
        DataFrame with columns: tx_id, is_anomaly (0/1), your_note.
        Only rows where is_anomaly is 0 or 1 (skips blank/review rows).
    """
    if not ANOMALY_LABELS.exists():
        log.warning("anomaly_labels.csv not found at %s", ANOMALY_LABELS)
        return pd.DataFrame(columns=["tx_id", "is_anomaly", "your_note"])

    df = pd.read_csv(ANOMALY_LABELS)
    # Keep only rows with a definitive label
    df = df[df["is_anomaly"].isin([0, 1, "0", "1"])].copy()
    df["is_anomaly"] = df["is_anomaly"].astype(int)
    log.info(
        "Loaded %d anomaly labels (%d positive, %d negative)",
        len(df), (df.is_anomaly == 1).sum(), (df.is_anomaly == 0).sum(),
    )
    return df[["tx_id", "is_anomaly", "your_note"]]


# ── Category corrections ───────────────────────────────────────────────────────

def load_category_corrections() -> dict[str, str]:
    """
    Load category correction map: tx_id -> correct_category.
    Also returns a pattern-based correction map for unlabeled transactions.

    Returns:
        dict mapping tx_id -> correct_category string.
    """
    if not CATEGORY_LABELS.exists():
        log.warning("category_corrections.csv not found at %s", CATEGORY_LABELS)
        return {}

    df = pd.read_csv(CATEGORY_LABELS)
    # Keep only rows with a filled correct_category
    df = df[df["correct_category"].notna() & (df["correct_category"].str.strip() != "")].copy()
    mapping = dict(zip(df["tx_id"], df["correct_category"].str.strip()))
    log.info("Loaded %d category corrections", len(mapping))
    return mapping


def get_pattern_corrections() -> list[tuple[str, str]]:
    """
    Regex patterns for category corrections that apply to ALL transactions
    (not just the labeled ones). Used to fix systematic errors at scale.

    Returns list of (pattern, correct_category) tuples, applied in order.
    """
    return [
        # Savings transfers to Livret A — definitely SAVINGS not SALARY
        (r"(?i)(livret a|virement vers livret)",     "SAVINGS"),
        # Investment platforms
        (r"(?i)(trade republic|degiro|bourso)",       "INVESTMENT"),
        # Transit pass
        (r"(?i)(imagine r|navigo|ratp)",             "TRANSPORT"),
        # Government fine
        (r"(?i)(amende\.gouv|amende gov)",           "FINES"),
        # Buy-now-pay-later
        (r"(?i)(3x 4x oney|oney\b)",                 "FINANCE"),
        # Foreign transaction fees
        (r"(?i)frais paiement etranger",             "BANKING_FEES"),
        # Housing / rent
        (r"(?i)(immobilier|immobiliere 3f|loyer)",   "RENT"),
        # International money transfers
        (r"(?i)(taptap send|remitly|western union)",  "TRANSFER"),
        # AI subscriptions
        (r"(?i)(openai|anthropic|claude\.ai)",       "AI_SERVICES"),
        # Bus tickets
        (r"(?i)sageb bus",                           "TRANSPORT"),
    ]


# ── Apply corrections to a full feature DataFrame ─────────────────────────────

def apply_category_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both tx_id-level corrections and pattern-based corrections to df.
    Adds a 'correction_applied' bool column.

    Args:
        df: Must have columns: date_operation, description, abs_amount, category.
    """
    import re

    df = df.copy()
    if "tx_id" not in df.columns:
        df = add_tx_ids(df)

    df["correction_applied"] = False

    # 1. Pattern-based corrections (applied to all rows)
    patterns = get_pattern_corrections()
    for pattern, correct_cat in patterns:
        mask = df["description"].str.contains(pattern, regex=True, na=False)
        changed = mask & (df["category"] != correct_cat)
        if changed.any():
            log.info(
                "Pattern '%s' -> %s: corrected %d rows",
                pattern[:40], correct_cat, changed.sum(),
            )
            df.loc[mask, "category"] = correct_cat
            df.loc[mask, "correction_applied"] = True

    # 2. tx_id-level corrections (override pattern corrections)
    tx_map = load_category_corrections()
    if tx_map:
        matched = df["tx_id"].isin(tx_map)
        df.loc[matched, "category"] = df.loc[matched, "tx_id"].map(tx_map)
        df.loc[matched, "correction_applied"] = True
        log.info("tx_id corrections applied to %d rows", matched.sum())

    total_corrected = df["correction_applied"].sum()
    log.info("Total category corrections applied: %d / %d rows", total_corrected, len(df))
    return df


def apply_anomaly_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ground truth anomaly labels into df.
    Adds column 'label' (0/1) for labeled rows; NaN for unlabeled.

    Args:
        df: Must have date_operation, description, abs_amount columns.
    """
    if "tx_id" not in df.columns:
        df = add_tx_ids(df)

    labels = load_anomaly_labels()
    if labels.empty:
        df["label"] = float("nan")
        return df

    df = df.merge(
        labels[["tx_id", "is_anomaly"]].rename(columns={"is_anomaly": "label"}),
        on="tx_id",
        how="left",
    )
    labeled = df["label"].notna().sum()
    log.info("Anomaly labels merged: %d / %d rows labeled", labeled, len(df))
    return df


# ── Summary ────────────────────────────────────────────────────────────────────

def label_summary() -> dict:
    """Return a quick summary of label coverage."""
    anom = load_anomaly_labels()
    cats = load_category_corrections()
    patterns = get_pattern_corrections()

    # Estimate how many transactions the patterns will fix
    try:
        df = pd.read_excel(FEATURES_XLSX, sheet_name="Full Data")
        df = apply_category_corrections(df)
        pattern_fixed = int(df["correction_applied"].sum())
    except Exception:
        pattern_fixed = None

    return {
        "anomaly_labels": {
            "total":    len(anom),
            "positive": int((anom.is_anomaly == 1).sum()) if not anom.empty else 0,
            "negative": int((anom.is_anomaly == 0).sum()) if not anom.empty else 0,
        },
        "category_corrections": {
            "tx_level_corrections": len(cats),
            "pattern_rules":        len(patterns),
            "transactions_fixed":   pattern_fixed,
        },
    }


if __name__ == "__main__":
    import json, logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = label_summary()
    print(json.dumps(summary, indent=2))
