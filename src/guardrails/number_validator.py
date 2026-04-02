"""
src/guardrails/number_validator.py — Reusable number hallucination detector.

Extracted from loan_report.py and generalised so all agents can use it.
"""

from __future__ import annotations
import re
from src.logger import get_logger

log = get_logger(__name__)


def check_numbers_in_text(
    text: str,
    expected: dict[str, float],
    tolerance: float = 0.15,
    min_financial_value: float = 10.0,
) -> list[str]:
    """
    Scan `text` for each metric in `expected` and flag values that deviate
    beyond `tolerance` from the expected figure.

    Args:
        text:                 LLM-generated text to scan.
        expected:             {label: expected_value} — e.g. {"income": 1061.0}.
        tolerance:            Fractional tolerance before flagging (default 15%).
        min_financial_value:  Skip numbers below this — they are ordinals/section
                              numbers, not financial figures (default 10).

    Returns:
        List of warning strings (empty = clean).
    """
    warnings: list[str] = []
    text_lower = text.lower()

    for label, expected_val in expected.items():
        if expected_val <= 0:
            continue  # can't compare negative/zero as unsigned prose

        # Search for the label keyword, then extract first qualifying number after it
        pattern = re.compile(re.escape(label.lower()), re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            window = text[m.end(): m.end() + 150]
            nums   = re.findall(r'(\d[\d,]*(?:\.\d+)?)', window)
            for raw in nums:
                found = float(raw.replace(",", ""))
                if found < min_financial_value:
                    continue  # skip small ints (ordinals, "3-month", section "1.")
                ratio = abs(found - expected_val) / expected_val
                if ratio > tolerance:
                    msg = (
                        f"[hallucination] '{label}': found {found:.2f} "
                        f"expected ~{expected_val:.2f} (deviation {ratio:.0%})"
                    )
                    warnings.append(msg)
                    log.warning(msg)
                break  # only first qualifying number per match
            break  # only first keyword occurrence

    return warnings


def assert_grounded(text: str, required_values: list[float], tolerance: float = 0.20) -> bool:
    """
    Return True if at least one of `required_values` appears verbatim (within
    tolerance) in `text`. Used to confirm an answer references real data.
    """
    nums_in_text = [
        float(n.replace(",", ""))
        for n in re.findall(r'\d[\d,]*(?:\.\d+)?', text)
        if float(n.replace(",", "")) >= 10
    ]
    for required in required_values:
        if required <= 0:
            continue
        for found in nums_in_text:
            if abs(found - required) / required <= tolerance:
                return True
    return False
