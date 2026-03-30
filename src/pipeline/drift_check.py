"""
drift_check.py — Monthly data drift detection using evidently.
===============================================================
Compares the most recent month's transaction distributions against a
6-month rolling baseline. Significant drift writes a warning to
data/pipeline_status.json and the dashboard shows a st.error() banner.

Usage (standalone):
    python drift_check.py

Called from run_pipeline.py after parse_statements.py completes.

Gap 8: evidently selected over alibi-detect (TF/PyTorch dep) and nannyml
(targets label-delay scenarios). evidently produces HTML reports from a
single function call, pip-installable, no server required.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PIPELINE_STATUS_JSON, FINANCE_DB, TRANSACTIONS_XLSX
from src.db import read_table, table_exists
from src.logger import get_logger

log = get_logger(__name__)

# Numeric columns we care about for drift detection
DRIFT_COLUMNS = ['amount', 'debit', 'credit']

# Proportion of columns showing drift that triggers a pipeline warning
DRIFT_THRESHOLD_RATIO = 0.5    # ≥50% of columns drifted → warning

# Output paths
_DATA_DIR    = Path(__file__).parent / "data"
_REPORT_DIR  = _DATA_DIR / "drift_reports"
_REPORT_HTML = None  # set at runtime with timestamp


def load_transactions() -> pd.DataFrame:
    """Load transactions from SQLite (preferred) or .xlsx fallback."""
    if table_exists("transactions"):
        df = read_table("transactions", parse_dates=["date_operation"])
        log.info("Drift check: loaded %d transactions from SQLite", len(df))
    elif TRANSACTIONS_XLSX.exists():
        df = pd.read_excel(TRANSACTIONS_XLSX, sheet_name="Transactions")
        df['date_operation'] = pd.to_datetime(df['date_operation'])
        log.info("Drift check: loaded %d transactions from .xlsx", len(df))
    else:
        log.error("No transaction data found. Run parse_statements.py first.")
        sys.exit(1)

    return df


def split_reference_current(df: pd.DataFrame):
    """
    Split transactions into:
      reference — 6 months prior to the most recent month
      current   — most recent calendar month

    Returns (reference_df, current_df, current_month_str)
    """
    df = df.copy()
    df['year_month'] = df['date_operation'].dt.to_period('M')

    sorted_months = sorted(df['year_month'].unique())
    if len(sorted_months) < 2:
        log.warning("Drift check requires at least 2 months of data. Skipping.")
        return None, None, None

    current_month = sorted_months[-1]
    # Use up to 6 months before current as reference
    reference_months = sorted_months[max(0, len(sorted_months) - 7): -1]

    current_df   = df[df['year_month'] == current_month].copy()
    reference_df = df[df['year_month'].isin(reference_months)].copy()

    log.info("Drift check: reference=%d months (%s→%s), current=%s (%d rows)",
             len(reference_months), reference_months[0], reference_months[-1],
             current_month, len(current_df))

    if len(current_df) < 5:
        log.warning("Current month has only %d transactions — drift check may be unreliable.",
                    len(current_df))

    return reference_df, current_df, str(current_month)


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                     current_month: str) -> dict:
    """
    Run evidently DataDriftPreset and return a summary dict:
      {column: {drifted: bool, p_value: float, test_name: str}}
    Also saves an HTML report to data/drift_reports/.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        log.error("'evidently' not installed. Run: pip install evidently")
        sys.exit(1)

    # Keep only the columns we track, dropping rows with nulls in those columns
    cols_available = [c for c in DRIFT_COLUMNS if c in reference_df.columns]
    if not cols_available:
        log.warning("None of %s found in DataFrame. Skipping drift check.", DRIFT_COLUMNS)
        return {}

    ref = reference_df[cols_available].dropna()
    cur = current_df[cols_available].dropna()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    # Save HTML report
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = _REPORT_DIR / f"drift_{current_month}_{timestamp}.html"
    report.save_html(str(html_path))
    log.info("Drift HTML report saved: %s", html_path)

    # Parse evidently result dict for per-column drift status
    result_dict  = report.as_dict()
    drift_summary: dict = {}

    try:
        metrics = result_dict.get("metrics", [])
        for metric in metrics:
            # DataDriftPreset exposes per-column results under 'result'
            result = metric.get("result", {})
            drift_by_column = result.get("drift_by_columns", {})
            for col, col_result in drift_by_column.items():
                drift_summary[col] = {
                    "drifted":   col_result.get("drift_detected", False),
                    "p_value":   col_result.get("p_value", None),
                    "threshold": col_result.get("threshold", None),
                    "test_name": col_result.get("stattest_name", "unknown"),
                }
    except Exception as e:
        log.warning("Could not parse evidently result dict: %s", e)

    return drift_summary


def update_pipeline_status(drift_summary: dict, current_month: str):
    """
    Append drift results to pipeline_status.json.
    If ≥50% of checked columns drifted, mark drift_warning=True.
    """
    if not drift_summary:
        log.info("No drift summary to write.")
        return

    n_drifted = sum(1 for v in drift_summary.values() if v.get("drifted"))
    n_total   = len(drift_summary)
    ratio     = n_drifted / n_total if n_total else 0
    has_warning = ratio >= DRIFT_THRESHOLD_RATIO

    drift_status = {
        "checked_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_month": current_month,
        "n_columns":     n_total,
        "n_drifted":     n_drifted,
        "drift_ratio":   round(ratio, 3),
        "drift_warning": has_warning,
        "columns":       drift_summary,
    }

    # Load existing status file if present, then merge in drift key
    existing = {}
    if PIPELINE_STATUS_JSON.exists():
        try:
            existing = json.loads(PIPELINE_STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass

    existing["drift"] = drift_status
    PIPELINE_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATUS_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    if has_warning:
        log.warning(
            "DRIFT WARNING: %d/%d columns drifted in %s (ratio=%.0f%%). "
            "Check data/drift_reports/ for the HTML report.",
            n_drifted, n_total, current_month, ratio * 100
        )
    else:
        log.info("Drift check OK: %d/%d columns drifted (%.0f%% < %.0f%% threshold).",
                 n_drifted, n_total, ratio * 100, DRIFT_THRESHOLD_RATIO * 100)

    log.info("Pipeline status updated: %s", PIPELINE_STATUS_JSON)


def run():
    """Entry point — callable from run_pipeline.py or standalone."""
    log.info("=" * 60)
    log.info("DRIFT CHECK — evidently")
    log.info("=" * 60)

    df = load_transactions()
    reference_df, current_df, current_month = split_reference_current(df)

    if reference_df is None:
        log.info("Drift check skipped — insufficient data.")
        return

    drift_summary = run_drift_report(reference_df, current_df, current_month)
    update_pipeline_status(drift_summary, current_month)

    log.info("Drift check complete.")


if __name__ == "__main__":
    run()
