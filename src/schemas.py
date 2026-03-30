"""
schemas.py — Pandera DataFrame schemas for pipeline validation.

Usage:
    from src.schemas import validate_transactions, validate_features

Each function raises pandera.errors.SchemaError with a human-readable
message if the DataFrame does not conform to the expected schema.
"""

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

from src.config import DATA_DATE_MIN, DATA_DATE_MAX

# ── 1. TRANSACTIONS SCHEMA ────────────────────────────────────────────────────
# Output of parse_statements.py — raw parsed transactions

TRANSACTIONS_SCHEMA = DataFrameSchema(
    columns={
        "date_operation": Column(
            pa.DateTime,
            checks=[
                Check(lambda s: s >= pd.Timestamp(DATA_DATE_MIN),
                      element_wise=True,
                      error=f"date_operation before {DATA_DATE_MIN}"),
                Check(lambda s: s <= pd.Timestamp(DATA_DATE_MAX),
                      element_wise=True,
                      error=f"date_operation after {DATA_DATE_MAX}"),
            ],
            nullable=False,
        ),
        "description": Column(
            str,
            checks=Check(lambda s: s.str.strip().str.len() > 0,
                         element_wise=False,
                         error="description contains empty strings"),
            nullable=False,
        ),
        "amount": Column(
            float,
            checks=[
                Check(lambda s: s.notna().all(), error="amount contains NaN"),
                Check(lambda s: s.apply(lambda x: x == x and x != float('inf') and x != float('-inf')),
                      element_wise=True,
                      error="amount contains inf"),
            ],
            nullable=False,
        ),
        "type": Column(
            str,
            checks=Check.isin(["DEBIT", "CREDIT"],
                              error="type must be DEBIT or CREDIT"),
            nullable=False,
        ),
    },
    checks=[
        # At least 10 rows — catches "parsed zero rows" silently
        Check(lambda df: len(df) >= 10,
              error="Parsed fewer than 10 transactions — check PDF format or directory"),
    ],
    coerce=True,
)

# ── 2. FEATURES SCHEMA ────────────────────────────────────────────────────────
# Output of feature_engineering.py — Feature Matrix sheet

FEATURE_MATRIX_SCHEMA = DataFrameSchema(
    columns={
        "date_operation": Column(pa.DateTime, nullable=False),
        "description":    Column(str,         nullable=False),
        "category":       Column(str,         nullable=False),
        "type":           Column(str,         checks=Check.isin(["DEBIT", "CREDIT"]), nullable=False),
        "abs_amount":     Column(float,       checks=Check.greater_than_or_equal_to(0), nullable=False),
        "log_amount":     Column(float,       nullable=False),
        "month":          Column(int,         checks=Check.in_range(1, 12), nullable=False),
        "day_of_week":    Column(int,         checks=Check.in_range(0, 6),  nullable=False),
    },
    coerce=True,
)


# ── 3. VALIDATION FUNCTIONS ───────────────────────────────────────────────────

def validate_transactions(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Validate a transactions DataFrame against TRANSACTIONS_SCHEMA.
    Raises pandera.errors.SchemaError with a clear message on failure.
    Returns the (coerced) DataFrame on success.
    """
    label = f" [{source}]" if source else ""
    try:
        validated = TRANSACTIONS_SCHEMA.validate(df, lazy=True)
        print(f"  [schema OK]{label} {len(validated)} transactions passed validation")
        return validated
    except pa.errors.SchemaErrors as err:
        print(f"  [schema FAIL]{label} Validation errors:")
        for _, row in err.failure_cases.iterrows():
            print(f"    Column '{row.get('column')}': {row.get('failure_case')} — {row.get('check')}")
        raise


def validate_features(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Validate a feature matrix DataFrame against FEATURE_MATRIX_SCHEMA.
    Raises pandera.errors.SchemaError with a clear message on failure.
    Returns the (coerced) DataFrame on success.
    """
    label = f" [{source}]" if source else ""
    try:
        validated = FEATURE_MATRIX_SCHEMA.validate(df, lazy=True)
        print(f"  [schema OK]{label} {len(validated)} feature rows passed validation")
        return validated
    except pa.errors.SchemaErrors as err:
        print(f"  [schema FAIL]{label} Validation errors:")
        for _, row in err.failure_cases.iterrows():
            print(f"    Column '{row.get('column')}': {row.get('failure_case')} — {row.get('check')}")
        raise
