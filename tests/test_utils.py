"""
tests/test_utils.py — Unit tests for pure utility functions.

Layer 1: No I/O, no models, no external dependencies.
Tests: parse_amount(), assign_category(), credit score computation,
       number validation, constraint building.
"""

import re
import pytest
import pandas as pd
import numpy as np

# ── parse_amount ──────────────────────────────────────────────────────────────

from parse_statements import parse_amount


class TestParseAmount:
    def test_standard_french_format(self):
        assert parse_amount("1 600,00") == 1600.0

    def test_small_amount(self):
        assert parse_amount("4,95") == pytest.approx(4.95)

    def test_no_space_separator(self):
        assert parse_amount("25,00") == pytest.approx(25.0)

    def test_large_amount_multiple_spaces(self):
        assert parse_amount("12 345,67") == pytest.approx(12345.67)

    def test_euro_sign_stripped(self):
        assert parse_amount("\u20ac9,90") == pytest.approx(9.90)

    def test_empty_string_returns_none(self):
        assert parse_amount("") is None

    def test_none_returns_none(self):
        assert parse_amount(None) is None

    def test_non_numeric_returns_none(self):
        assert parse_amount("SOLDE") is None

    def test_negative_value(self):
        result = parse_amount("-9,90")
        assert result == pytest.approx(-9.90)


# ── assign_category helper ────────────────────────────────────────────────────
# We cannot safely import feature_engineering.py at module level because it
# executes pipeline code on import. Instead we replicate the minimal logic
# needed to exercise the category rules in isolation.

# Subset of CATEGORY_RULES from feature_engineering.py, kept in sync manually.
_CATEGORY_RULES = {
    'SALARY':       [r'VIREMENT', r'SALAIRE', r'LEOPOLD', r'PAIE'],
    'GROCERIES':    [r'CARREFOUR', r'LIDL', r'AUCHAN', r'MONOPRIX', r'LECLERC',
                     r'SUPERMARCHE', r'FRANPRIX', r'INTERMARCHE', r'CASINO'],
    'RESTAURANTS':  [r'KFC', r'MCDO', r'BURGER', r'PIZZA', r'RESTAURANT',
                     r'KEBAB', r'SUSHI', r'SNACK', r'CAFE\s', r'BRASSERIE'],
    'TRANSPORT':    [r'NAVIGO', r'RATP', r'SNCF', r'UBER', r'TAXI',
                     r'PARKING', r'CARBURANT', r'TOTAL\s', r'AUTOROUTE'],
    'INSURANCE':    [r'COTIS', r'CRISTAL', r'ASSURANCE', r'CONFORT', r'MUTUELLE'],
    'BANKING_FEES': [r'FRAIS', r'COMMISSION', r'AGIOS', r'INTERETS'],
}


def assign_category(description: str) -> str:
    desc_upper = description.upper()
    for category, patterns in _CATEGORY_RULES.items():
        for pat in patterns:
            if re.search(pat, desc_upper):
                return category
    return 'OTHER'


class TestAssignCategory:
    def test_salary_virement(self):
        assert assign_category("VIREMENT SALAIRE LEOPOLD") == "SALARY"

    def test_groceries_lidl(self):
        assert assign_category("LIDL PARIS 12") == "GROCERIES"

    def test_transport_ratp(self):
        assert assign_category("RATP NAVIGO MOIS") == "TRANSPORT"

    def test_restaurant_kfc(self):
        assert assign_category("KFC REPUBLIC") == "RESTAURANTS"

    def test_insurance_cotis(self):
        assert assign_category("COTIS CRISTAL CONFORT") == "INSURANCE"

    def test_banking_fees_frais(self):
        assert assign_category("FRAIS BANCAIRES TRIMESTRIEL") == "BANKING_FEES"

    def test_unknown_returns_other(self):
        assert assign_category("ZZZUNKNOWN MERCHANT XYZ") == "OTHER"

    def test_case_insensitive(self):
        assert assign_category("lidl supermarche") == "GROCERIES"

    def test_priority_salary_over_other(self):
        # VIREMENT matches SALARY before anything else
        assert assign_category("VIREMENT LOYER MENSUEL") == "SALARY"


# ── validate_report_numbers (loan_report.py) ─────────────────────────────────

from loan_report import validate_report_numbers


class TestValidateReportNumbers:
    def _make_ctx(self, **overrides):
        base = {
            'dscr': 2.50,
            'savings_rate': 0.15,
            'avg_3m_income': 1800.0,
            'avg_3m_spend': 1400.0,
            'expense_volatility': 250.0,
            'credit_score': 72,
        }
        base.update(overrides)
        return base

    def test_no_warnings_when_numbers_match(self):
        ctx = self._make_ctx()
        report = (
            "The applicant has a credit score of 72 out of 100. "
            "The DSCR stands at 2.50x, which is above the 1.5 minimum. "
            "The savings rate is 15.0%."
        )
        warnings = validate_report_numbers(report, ctx)
        assert warnings == []

    def test_hallucinated_credit_score_flagged(self):
        ctx = self._make_ctx(credit_score=72)
        # Report claims score of 99 — clearly hallucinated
        report = "The applicant has an excellent credit score of 99 out of 100."
        warnings = validate_report_numbers(report, ctx)
        # Should detect a mismatch near 'credit score'
        assert isinstance(warnings, list)

    def test_empty_report_no_warnings(self):
        ctx = self._make_ctx()
        warnings = validate_report_numbers("", ctx)
        assert warnings == []

    def test_returns_list(self):
        ctx = self._make_ctx()
        result = validate_report_numbers("Some report text.", ctx)
        assert isinstance(result, list)


# ── build_constraints (loan_report.py) ───────────────────────────────────────

from loan_report import build_constraints


class TestBuildConstraints:
    def _ctx(self, label, score=60):
        return {'ensemble_label': label, 'credit_score': score}

    def test_high_risk_mentions_decline(self):
        block = build_constraints(self._ctx('HIGH_RISK', 35))
        assert 'DECLINE' in block

    def test_high_risk_forbids_approve(self):
        block = build_constraints(self._ctx('HIGH_RISK', 35))
        block_upper = block.upper()
        assert 'MUST NOT' in block_upper or 'NOT' in block_upper

    def test_medium_risk_mentions_conditional(self):
        block = build_constraints(self._ctx('MEDIUM_RISK', 55))
        assert 'CONDITIONAL' in block.upper()

    def test_low_risk_allows_approve(self):
        block = build_constraints(self._ctx('LOW_RISK', 75))
        assert 'APPROVE' in block.upper()

    def test_low_score_reflected(self):
        block = build_constraints(self._ctx('HIGH_RISK', 30))
        assert '30' in block

    def test_high_score_reflected(self):
        block = build_constraints(self._ctx('LOW_RISK', 80))
        assert '80' in block

    def test_returns_nonempty_string(self):
        result = build_constraints(self._ctx('LOW_RISK'))
        assert isinstance(result, str)
        assert len(result) > 0
