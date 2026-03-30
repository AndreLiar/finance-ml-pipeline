"""
db.py — SQLite read/write helpers for the finance pipeline.

All pipeline scripts call read_table() / write_table() / append_table()
instead of pd.read_excel() / ExcelWriter. The database path comes from config.

Usage:
    from db import read_table, write_table, append_table, db_exists

    df = read_table("transactions")
    write_table(df, "transactions")
    append_table(df, "anomaly_results")
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager

from config import FINANCE_DB


@contextmanager
def _connect():
    """Context manager that opens and closes a SQLite connection."""
    con = sqlite3.connect(FINANCE_DB)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def db_exists() -> bool:
    """Return True if the database file exists."""
    return FINANCE_DB.exists()


def read_table(table: str, **kwargs) -> pd.DataFrame:
    """
    Read a full table from finance.db into a DataFrame.

    Args:
        table:   Table name (see migrate_xlsx_to_sqlite.py for naming)
        **kwargs: Passed to pd.read_sql (e.g. parse_dates=['date_operation'])
    """
    with _connect() as con:
        return pd.read_sql(f"SELECT * FROM {table}", con, **kwargs)


def write_table(df: pd.DataFrame, table: str, if_exists: str = "replace"):
    """
    Write a DataFrame to a table (default: replace existing).

    Args:
        df:        DataFrame to write
        table:     Target table name
        if_exists: 'replace' (default) | 'append' | 'fail'
    """
    with _connect() as con:
        df.to_sql(table, con, if_exists=if_exists, index=False)


def append_table(df: pd.DataFrame, table: str):
    """Append rows to an existing table (or create it if absent)."""
    write_table(df, table, if_exists="append")


def table_exists(table: str) -> bool:
    """Return True if the table exists in the database."""
    with _connect() as con:
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None


def list_tables() -> list[str]:
    """List all table names in the database."""
    with _connect() as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cur.fetchall()]
