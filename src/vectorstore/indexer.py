"""
src/vectorstore/indexer.py — Index all financial data into the vector store

Reads:
  - data/transactions.xlsx        -> "transactions" collection (1226 docs)
  - data/features.xlsx            -> "summaries" collection (monthly + category)
  - data/anomaly_results.xlsx     -> "anomalies" collection (flagged transactions)

Run as:
  py -3 -m src.vectorstore.indexer

Or call build_index() programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    TRANSACTIONS_XLSX, FEATURES_XLSX, ANOMALY_RESULTS_XLSX, DATA_DIR,
)
from src.vectorstore.store import get_store, COLL_TRANSACTIONS, COLL_SUMMARIES, COLL_ANOMALIES
from src.vectorstore.retriever import store_status

log = logging.getLogger(__name__)


def _read_excel(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:
        log.warning("Could not read %s[%s]: %s", path.name, sheet, exc)
        return None


# ── Document builders ──────────────────────────────────────────────────────────

def _build_transaction_docs(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Convert each transaction row into a searchable text + metadata payload."""
    texts, payloads = [], []
    for _, row in df.iterrows():
        try:
            date = pd.to_datetime(row.get("date_operation", "")).strftime("%Y-%m-%d")
        except Exception:
            date = str(row.get("date_operation", "unknown"))

        amount   = float(row.get("debit", row.get("abs_amount", row.get("amount", 0))) or 0)
        credit   = float(row.get("credit", 0) or 0)
        category = str(row.get("category", "UNKNOWN"))
        desc     = str(row.get("description", ""))[:120]
        tx_type  = str(row.get("transaction_type", row.get("type", "")))

        # Natural language sentence for embedding
        if amount > 0:
            text = f"{date} debit EUR{amount:.0f} {category} {desc}"
        elif credit > 0:
            text = f"{date} credit EUR{credit:.0f} {category} {desc}"
        else:
            text = f"{date} {tx_type} {category} {desc}"

        payload = {
            "date": date, "category": category, "description": desc,
            "amount_eur": amount, "credit_eur": credit, "type": tx_type,
        }
        texts.append(text)
        payloads.append(payload)
    return texts, payloads


def _build_summary_docs(features_path: Path) -> tuple[list[str], list[dict]]:
    """Build monthly aggregate + category summary documents."""
    texts, payloads = [], []

    # Monthly aggregates
    monthly = _read_excel(features_path, "Monthly Aggregates")
    if monthly is not None and not monthly.empty:
        for _, row in monthly.iterrows():
            ym      = str(row.get("year_month", "?"))
            income  = float(row.get("monthly_income", 0) or 0)
            spend   = float(row.get("monthly_spend",  0) or 0)
            net     = float(row.get("monthly_net", income - spend) or 0)
            tx_count= int(row.get("transaction_count", row.get("tx_count", 0)) or 0)
            text = (
                f"Month {ym}: income EUR{income:.0f}, spend EUR{spend:.0f}, "
                f"net EUR{net:.0f}, {tx_count} transactions"
            )
            texts.append(text)
            payloads.append({"year_month": ym, "income": income, "spend": spend, "net": net, "doc_type": "monthly"})

    # Category summaries
    cat_summary = _read_excel(features_path, "Category Summary")
    if cat_summary is not None and not cat_summary.empty:
        spend_col = "total_spend" if "total_spend" in cat_summary.columns else "total_debit"
        count_col = "count"       if "count"       in cat_summary.columns else "transaction_count"
        for _, row in cat_summary.iterrows():
            cat    = str(row.get("category", "?"))
            total  = float(row.get(spend_col, 0) or 0)
            count  = int(row.get(count_col, 0) or 0)
            avg    = total / count if count > 0 else 0
            text = (
                f"Category {cat}: total spend EUR{total:.0f}, "
                f"{count} transactions, average EUR{avg:.0f}"
            )
            texts.append(text)
            payloads.append({"category": cat, "total_spend": total, "count": count, "avg": avg, "doc_type": "category"})

    return texts, payloads


def _build_anomaly_docs(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Build anomaly documents from flagged transactions."""
    texts, payloads = [], []
    for _, row in df.iterrows():
        try:
            date = pd.to_datetime(row.get("date_operation", "")).strftime("%Y-%m-%d")
        except Exception:
            date = str(row.get("date_operation", "unknown"))

        amount   = float(row.get("debit", row.get("abs_amount", 0)) or 0)
        category = str(row.get("category", "UNKNOWN"))
        desc     = str(row.get("description", ""))[:120]
        votes    = float(row.get("vote_count", row.get("anomaly_score", 0)) or 0)

        text = (
            f"ANOMALY {date} EUR{amount:.0f} {category} flagged by {votes:.0f}/3 models: {desc}"
        )
        payload = {
            "date": date, "category": category, "description": desc,
            "amount_eur": amount, "vote_count": votes, "doc_type": "anomaly",
        }
        texts.append(text)
        payloads.append(payload)
    return texts, payloads


# ── Main indexer ───────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> dict:
    """
    Index all financial data into the vector store.

    Args:
        force: Re-index even if collections are already populated.

    Returns:
        Status dict with counts per collection.
    """
    store = get_store()
    counts = {}

    # ── Transactions ───────────────────────────────────────────────────────────
    if force or not store.is_indexed(COLL_TRANSACTIONS):
        df = _read_excel(TRANSACTIONS_XLSX, "Transactions")
        if df is not None and not df.empty:
            texts, payloads = _build_transaction_docs(df)
            store.index(COLL_TRANSACTIONS, texts, payloads)
            counts[COLL_TRANSACTIONS] = len(texts)
            log.info("Indexed %d transactions", len(texts))
        else:
            log.warning("transactions.xlsx not found or empty — skipping")
            counts[COLL_TRANSACTIONS] = 0
    else:
        counts[COLL_TRANSACTIONS] = store.collection_size(COLL_TRANSACTIONS)
        log.info("Transactions already indexed (%d docs)", counts[COLL_TRANSACTIONS])

    # ── Summaries ──────────────────────────────────────────────────────────────
    if force or not store.is_indexed(COLL_SUMMARIES):
        texts, payloads = _build_summary_docs(FEATURES_XLSX)
        if texts:
            store.index(COLL_SUMMARIES, texts, payloads)
            counts[COLL_SUMMARIES] = len(texts)
            log.info("Indexed %d summary documents", len(texts))
        else:
            log.warning("No summary data found — skipping")
            counts[COLL_SUMMARIES] = 0
    else:
        counts[COLL_SUMMARIES] = store.collection_size(COLL_SUMMARIES)
        log.info("Summaries already indexed (%d docs)", counts[COLL_SUMMARIES])

    # ── Anomalies ──────────────────────────────────────────────────────────────
    if force or not store.is_indexed(COLL_ANOMALIES):
        df = _read_excel(ANOMALY_RESULTS_XLSX, "Flagged Anomalies")
        if df is not None and not df.empty:
            texts, payloads = _build_anomaly_docs(df)
            store.index(COLL_ANOMALIES, texts, payloads)
            counts[COLL_ANOMALIES] = len(texts)
            log.info("Indexed %d anomaly documents", len(texts))
        else:
            log.warning("anomaly_results.xlsx not found or empty — skipping")
            counts[COLL_ANOMALIES] = 0
    else:
        counts[COLL_ANOMALIES] = store.collection_size(COLL_ANOMALIES)
        log.info("Anomalies already indexed (%d docs)", counts[COLL_ANOMALIES])

    return counts


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build vector store index")
    parser.add_argument("--force", action="store_true", help="Re-index even if already populated")
    args = parser.parse_args()

    print("Building vector store index...")
    counts = build_index(force=args.force)
    print("\nIndex status:")
    for coll, count in counts.items():
        print(f"  {coll:<20}: {count} documents")

    print("\nFull status:")
    status = store_status()
    for coll, info in status.items():
        print(f"  {coll:<20}: indexed={info['indexed']} size={info['size']}")
