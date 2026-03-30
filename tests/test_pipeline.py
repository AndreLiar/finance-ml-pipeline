"""
tests/test_pipeline.py — Integration tests for the parse → feature pipeline.

Layer 2: Uses synthetic fixture DataFrames (no real PDFs, no real DB).
Verifies that the schema produced by the parse stage satisfies the constraints
expected by the feature engineering stage.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


# ── Schema contract expected at parse_statements output ──────────────────────

REQUIRED_COLUMNS = {'date_operation', 'description', 'type', 'debit', 'credit', 'amount'}
VALID_TYPES      = {'DEBIT', 'CREDIT'}


def _assert_transactions_schema(df: pd.DataFrame):
    """Shared schema assertions used across multiple tests."""
    assert isinstance(df, pd.DataFrame), "Output must be a DataFrame"
    assert not df.empty, "DataFrame must not be empty"

    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"

    assert pd.api.types.is_datetime64_any_dtype(df['date_operation']), \
        "date_operation must be datetime dtype"

    assert df['date_operation'].notna().all(), "date_operation must have no NaT values"

    invalid_types = set(df['type'].unique()) - VALID_TYPES - {'BOTH', 'UNKNOWN'}
    assert not invalid_types, f"Unexpected type values: {invalid_types}"

    assert df['description'].notna().all(), "description must have no nulls"
    assert (df['description'].str.strip() != '').all(), "description must not be blank"

    assert df['debit'].ge(0).all(),  "debit must be non-negative"
    assert df['credit'].ge(0).all(), "credit must be non-negative"

    assert df['amount'].notna().all(), "amount must have no nulls"
    assert np.isfinite(df['amount']).all(), "amount must be finite"


class TestTransactionsSchema:
    """Verify the schema of the sample_transactions fixture itself."""

    def test_required_columns_present(self, sample_transactions):
        missing = REQUIRED_COLUMNS - set(sample_transactions.columns)
        assert not missing

    def test_date_operation_is_datetime(self, sample_transactions):
        assert pd.api.types.is_datetime64_any_dtype(sample_transactions['date_operation'])

    def test_no_null_dates(self, sample_transactions):
        assert sample_transactions['date_operation'].notna().all()

    def test_type_values_valid(self, sample_transactions):
        valid = {'DEBIT', 'CREDIT', 'BOTH', 'UNKNOWN'}
        assert set(sample_transactions['type'].unique()).issubset(valid)

    def test_description_non_empty(self, sample_transactions):
        assert (sample_transactions['description'].str.strip() != '').all()

    def test_debit_non_negative(self, sample_transactions):
        assert sample_transactions['debit'].ge(0).all()

    def test_credit_non_negative(self, sample_transactions):
        assert sample_transactions['credit'].ge(0).all()

    def test_amount_finite(self, sample_transactions):
        assert np.isfinite(sample_transactions['amount']).all()

    def test_row_count_above_minimum(self, sample_transactions):
        assert len(sample_transactions) >= 10


class TestFeatureEngineeringContract:
    """
    Verify that a cleaned transactions DataFrame produces the columns
    expected by downstream ML scripts.
    """

    def _clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirror the cleaning steps in feature_engineering.py."""
        df = df[~df['description'].str.contains('RECAPITULATIF', case=False, na=False)]
        df = df[~df['type'].isin(['BOTH', 'UNKNOWN'])]
        df['debit']  = df['debit'].fillna(0.0)
        df['credit'] = df['credit'].fillna(0.0)
        df['amount'] = df.apply(
            lambda r: r['credit'] if r['type'] == 'CREDIT' else -r['debit'], axis=1
        )
        df['description'] = df['description'].str.replace(r'\s*\|\s*', ' ', regex=True).str.strip()
        return df.reset_index(drop=True)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['year']        = df['date_operation'].dt.year
        df['month']       = df['date_operation'].dt.month
        df['day_of_week'] = df['date_operation'].dt.dayofweek
        df['quarter']     = df['date_operation'].dt.quarter
        return df

    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['abs_amount']     = df['amount'].abs()
        df['is_round_number'] = (df['abs_amount'] % 10 == 0).astype(int)
        df['log_amount']     = np.log1p(df['abs_amount'])
        return df

    def test_cleaning_removes_recapitulatif(self, sample_transactions):
        dirty = sample_transactions.copy()
        dirty.loc[0, 'description'] = 'RECAPITULATIF COMPTE'
        dirty.loc[0, 'type']        = 'DEBIT'
        cleaned = self._clean_transactions(dirty)
        assert 'RECAPITULATIF COMPTE' not in cleaned['description'].values

    def test_cleaning_removes_both_type(self, sample_transactions):
        dirty = sample_transactions.copy()
        dirty.loc[1, 'type'] = 'BOTH'
        cleaned = self._clean_transactions(dirty)
        assert 'BOTH' not in cleaned['type'].values

    def test_temporal_features_added(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_temporal_features(df)
        for col in ('year', 'month', 'day_of_week', 'quarter'):
            assert col in df.columns

    def test_year_values_plausible(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_temporal_features(df)
        assert df['year'].between(2020, 2030).all()

    def test_month_values_1_to_12(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_temporal_features(df)
        assert df['month'].between(1, 12).all()

    def test_transaction_features_added(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_transaction_features(df)
        for col in ('abs_amount', 'is_round_number', 'log_amount'):
            assert col in df.columns

    def test_abs_amount_non_negative(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_transaction_features(df)
        assert df['abs_amount'].ge(0).all()

    def test_log_amount_non_negative(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        df = self._add_transaction_features(df)
        assert df['log_amount'].ge(0).all()

    def test_credit_debit_split_correct(self, sample_transactions):
        df = self._clean_transactions(sample_transactions)
        credits = df[df['type'] == 'CREDIT']['amount']
        debits  = df[df['type'] == 'DEBIT']['amount']
        assert (credits >= 0).all(), "CREDIT rows must have non-negative amount"
        assert (debits  <= 0).all(), "DEBIT rows must have non-positive amount"


class TestSchemaValidation:
    """
    Test that pandera schema validation catches bad data correctly.
    """

    def test_valid_transactions_pass(self, sample_transactions):
        from src.schemas import validate_transactions
        result = validate_transactions(sample_transactions, source="test")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_empty_dataframe_raises(self):
        from src.schemas import validate_transactions
        empty = pd.DataFrame(columns=['date_operation', 'description', 'type',
                                      'debit', 'credit', 'amount'])
        with pytest.raises(Exception):
            validate_transactions(empty, source="test")

    def test_missing_column_raises(self, sample_transactions):
        from src.schemas import validate_transactions
        bad = sample_transactions.drop(columns=['description'])
        with pytest.raises(Exception):
            validate_transactions(bad, source="test")
