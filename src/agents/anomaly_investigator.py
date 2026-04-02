"""
src/agents/anomaly_investigator.py — Anomaly Investigator ReAct Agent

For each flagged transaction the agent:
  1. Fetches category statistics (avg, max, std)
  2. Checks the transaction's historical frequency
  3. Looks at the month context (was November generally anomalous?)
  4. Reasons about suspicion level
  5. Returns a structured AnomalyVerdict (Pydantic enforced)

Guardrails:
  - AnomalyVerdict.suspicion must be LOW/MEDIUM/HIGH (enum enforced)
  - AnomalyVerdict.action must be MONITOR/REVIEW/FLAG_FOR_AUDIT
  - AnomalyVerdict.reason must contain at least one number
  - Fallback to rule-based verdict if Ollama is down
  - Max 8 tool calls per transaction investigation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    ANOMALY_RESULTS_XLSX, FEATURES_XLSX,
)
from src.guardrails.structured_outputs import AnomalyVerdict, AnomalyInvestigationReport, RiskLevel
from src.guardrails.llm_fallback import ollama_is_available, anomaly_fallback
from src.logger import get_logger

log = get_logger(__name__)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _read(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:
        log.warning("Could not load %s[%s]: %s", path.name, sheet, exc)
        return None

# Cache full datasets at module level (loaded once per process)
_all_transactions: pd.DataFrame | None = None
_flagged:          pd.DataFrame | None = None
_by_category:      pd.DataFrame | None = None

def _load_data():
    global _all_transactions, _flagged, _by_category
    if _all_transactions is None:
        _df = _read(ANOMALY_RESULTS_XLSX, "All Transactions")
        _all_transactions = _df if _df is not None else pd.DataFrame()
        _df2 = _read(ANOMALY_RESULTS_XLSX, "Flagged Anomalies")
        _flagged = _df2 if _df2 is not None else pd.DataFrame()
        _df3 = _read(ANOMALY_RESULTS_XLSX, "By Category")
        _by_category = _df3 if _df3 is not None else pd.DataFrame()


# ── Investigation tools ────────────────────────────────────────────────────────

@tool
def get_category_stats(category: str) -> str:
    """
    Get spending statistics for a category: average, median, max, and std deviation.
    Use this to judge whether a transaction amount is unusual for its category.
    Args:
        category: The spending category name (e.g. TRANSFER, GROCERIES).
    """
    _load_data()
    if _all_transactions is None or _all_transactions.empty:
        return "Transaction data not available."
    df = _all_transactions[_all_transactions["category"].str.upper() == category.upper()]
    if df.empty:
        return f"No transactions found in category '{category}'."
    col = "debit" if "debit" in df.columns else "abs_amount"
    df  = df[df[col] > 0]
    return (
        f"Category '{category}' stats ({len(df)} transactions):\n"
        f"  Average : EUR{df[col].mean():.0f}\n"
        f"  Median  : EUR{df[col].median():.0f}\n"
        f"  Std dev : EUR{df[col].std():.0f}\n"
        f"  Max     : EUR{df[col].max():.0f}\n"
        f"  Min     : EUR{df[col].min():.0f}"
    )


@tool
def get_similar_amount_history(amount: float, tolerance_pct: float = 0.15) -> str:
    """
    Check if a transaction amount has appeared before in the full history
    (within a tolerance). Returns dates and descriptions of similar past transactions.
    Args:
        amount: Transaction amount in EUR.
        tolerance_pct: Fraction tolerance for 'similar' (default 0.15 = 15%).
    """
    _load_data()
    if _all_transactions is None or _all_transactions.empty:
        return "Transaction data not available."
    col = "debit" if "debit" in _all_transactions.columns else "abs_amount"
    lo, hi = amount * (1 - tolerance_pct), amount * (1 + tolerance_pct)
    matches = _all_transactions[
        (_all_transactions[col] >= lo) & (_all_transactions[col] <= hi)
    ]
    if matches.empty:
        return f"No historical transactions found near EUR{amount:.0f} (+-{tolerance_pct:.0%})."
    lines = [f"Found {len(matches)} similar transactions near EUR{amount:.0f}:"]
    for _, r in matches.head(5).iterrows():
        try:
            date = pd.to_datetime(r["date_operation"]).strftime("%Y-%m-%d")
        except Exception:
            date = str(r.get("date_operation", "?"))
        lines.append(f"  {date} | EUR{r.get(col,0):.0f} | {str(r.get('description',''))[:40]}")
    return "\n".join(lines)


@tool
def get_month_anomaly_context(year_month: str) -> str:
    """
    Get the anomaly rate and total transactions for a specific month.
    Use this to determine if the flagged transaction occurred in an
    already-suspicious month.
    Args:
        year_month: Month in format YYYY-MM (e.g. '2025-11').
    """
    df = _read(ANOMALY_RESULTS_XLSX, "Monthly Rate")
    if df is None or df.empty:
        return "Monthly anomaly data not available."
    row = df[df["year_month"] == year_month]
    if row.empty:
        return f"No data for month {year_month}."
    r = row.iloc[0]
    return (
        f"Month {year_month}:\n"
        f"  Total transactions : {int(r.get('total_tx', 0))}\n"
        f"  Anomalous          : {int(r.get('anomaly_tx', 0))}\n"
        f"  Anomaly rate       : {r.get('anomaly_rate', 0):.1f}%"
    )


@tool
def get_category_anomaly_rate(category: str) -> str:
    """
    Get how many anomalies were flagged in a given category and what
    fraction of all category transactions were anomalous.
    Args:
        category: The spending category name.
    """
    df = _read(ANOMALY_RESULTS_XLSX, "By Category")
    if df is None or df.empty:
        return "Category anomaly data not available."
    row = df[df["category"].str.upper() == category.upper()]
    if row.empty:
        return f"No anomaly data for category '{category}'."
    r = row.iloc[0]
    return (
        f"Category '{category}' anomaly stats:\n"
        f"  Total flagged  : {int(r.get('count', 0))}\n"
        f"  Total spend    : EUR{r.get('total', 0):.0f}\n"
        f"  Average amount : EUR{r.get('avg', 0):.0f}\n"
        f"  Max amount     : EUR{r.get('max', 0):.0f}"
    )


# ── Investigation tools list ───────────────────────────────────────────────────

_INVESTIGATION_TOOLS = [
    get_category_stats,
    get_similar_amount_history,
    get_month_anomaly_context,
    get_category_anomaly_rate,
]


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a financial fraud analyst. Your job is to investigate flagged transactions
and produce a structured verdict.

For each transaction you receive:
1. Call get_category_stats to understand normal spending in that category
2. Call get_similar_amount_history to check if this amount has appeared before
3. Call get_month_anomaly_context to see if this month was generally suspicious
4. Based on the tool results, decide: suspicion=LOW/MEDIUM/HIGH, action=MONITOR/REVIEW/FLAG_FOR_AUDIT
5. Write a reason that MUST include at least one specific number from the tools

RULES:
- NEVER invent statistics. Every number in your reason must come from a tool.
- suspicion MUST be exactly: LOW, MEDIUM, or HIGH
- action MUST be exactly: MONITOR, REVIEW, or FLAG_FOR_AUDIT
- reason MUST be one sentence containing at least one number

Respond with a JSON object in this exact format:
{
  "suspicion": "HIGH",
  "action": "FLAG_FOR_AUDIT",
  "reason": "Amount EUR487 is 3.4x above the TRANSFER category average of EUR143."
}
"""


# ── Agent factory ──────────────────────────────────────────────────────────────

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,      # very low — verdicts must be deterministic
            num_predict=300,      # short — only need the JSON verdict
        )
        _agent = create_react_agent(llm, _INVESTIGATION_TOOLS)
        log.info("Anomaly Investigator Agent initialised (model=%s)", LLM_MODEL)
    return _agent


# ── Verdict parser ─────────────────────────────────────────────────────────────

def _parse_verdict(text: str, row: pd.Series) -> dict:
    """Extract JSON verdict from agent output, with fallback parsing."""
    import re as _re

    # Try to find our expected JSON block {"suspicion":..., "action":..., "reason":...}
    match = _re.search(
        r'\{\s*"suspicion"\s*:\s*"(HIGH|MEDIUM|LOW)".*?"action"\s*:\s*"(\w+)".*?"reason"\s*:\s*"([^"]+)"',
        text, _re.DOTALL | _re.IGNORECASE
    )
    if match:
        return {
            "suspicion": match.group(1).upper(),
            "action":    match.group(2).upper(),
            "reason":    match.group(3),
        }

    # If text looks like a tool call list (not a final verdict), generate rule-based verdict
    if '"name"' in text and '"arguments"' in text:
        # Agent returned tool calls instead of final answer — apply rule-based verdict
        amount   = float(row.get("debit", row.get("abs_amount", 0)))
        category = str(row.get("category", "UNKNOWN"))
        votes    = float(row.get("vote_count", row.get("anomaly_score", 2)))
        suspicion = "HIGH" if votes >= 3 else "MEDIUM" if votes >= 2 else "LOW"
        action    = "FLAG_FOR_AUDIT" if suspicion == "HIGH" else "REVIEW"
        reason    = (
            f"EUR{amount:.0f} flagged by {votes:.0f}/3 ML models in category {category} "
            f"(rule-based verdict — agent returned tool calls without final answer)."
        )
        return {"suspicion": suspicion, "action": action, "reason": reason}

    # General fallback: extract keywords from free text
    text_upper = text.upper()
    suspicion = "HIGH"   if "HIGH"   in text_upper else \
                "MEDIUM" if "MEDIUM" in text_upper else "LOW"
    action    = "FLAG_FOR_AUDIT" if "FLAG"   in text_upper else \
                "REVIEW"         if "REVIEW" in text_upper else "MONITOR"
    amount    = float(row.get("debit", row.get("abs_amount", 0)))
    reason    = f"EUR{amount:.0f} — {text[:150].strip()}"
    return {"suspicion": suspicion, "action": action, "reason": reason}


# ── Public API ─────────────────────────────────────────────────────────────────

def investigate_transaction(row: pd.Series) -> AnomalyVerdict:
    """
    Investigate a single flagged transaction and return a structured verdict.
    """
    try:
        date = pd.to_datetime(row["date_operation"]).strftime("%Y-%m-%d")
    except Exception:
        date = str(row.get("date_operation", "unknown"))

    amount   = float(row.get("debit", row.get("abs_amount", 0)))
    category = str(row.get("category", "UNKNOWN"))
    desc     = str(row.get("description", ""))[:60]
    year_month = date[:7]  # YYYY-MM

    log.info("Investigating: %s | EUR%.0f | %s | %s", date, amount, category, desc[:30])

    # Guardrail: fallback if Ollama is down
    if not ollama_is_available():
        cat_df = _read(ANOMALY_RESULTS_XLSX, "By Category")
        cat_avg = 0.0
        if cat_df is not None:
            row_cat = cat_df[cat_df["category"].str.upper() == category.upper()]
            if not row_cat.empty:
                cat_avg = float(row_cat.iloc[0].get("avg", 0))
        fb = anomaly_fallback(desc, amount, cat_avg)
        return AnomalyVerdict(
            date=date, description=desc, amount_eur=amount, category=category,
            suspicion=RiskLevel(fb["suspicion"]),
            reason=fb["reason"],
            action=fb["action"],
        )

    question = (
        f"Investigate this flagged transaction:\n"
        f"  Date        : {date}\n"
        f"  Amount      : EUR{amount:.0f}\n"
        f"  Category    : {category}\n"
        f"  Description : {desc}\n"
        f"  Month       : {year_month}\n\n"
        f"Use the tools to gather context, then return your JSON verdict."
    )

    try:
        agent  = _get_agent()
        result = agent.invoke(
            {"messages": [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]},
            config={"recursion_limit": 10},  # guardrail: max ~4-5 tool calls
        )

        final = result["messages"][-1]
        text  = final.content if hasattr(final, "content") else str(final)
        parsed = _parse_verdict(text, row)

        # Guardrail: validate through Pydantic
        return AnomalyVerdict(
            date=date,
            description=desc,
            amount_eur=amount,
            category=category,
            suspicion=RiskLevel(parsed.get("suspicion", "MEDIUM")),
            reason=parsed.get("reason", f"EUR{amount:.0f} flagged by ensemble (3/3 models)."),
            action=parsed.get("action", "REVIEW"),
        )

    except Exception as exc:
        log.error("Investigation failed for %s EUR%.0f: %s", date, amount, exc)
        return AnomalyVerdict(
            date=date, description=desc, amount_eur=amount, category=category,
            suspicion=RiskLevel.MEDIUM,
            reason=f"EUR{amount:.0f} could not be analysed due to agent error: {str(exc)[:80]}",
            action="REVIEW",
        )


def run_investigation(top_n: int = 10) -> AnomalyInvestigationReport:
    """
    Investigate the top-N most suspicious flagged transactions and
    return a full AnomalyInvestigationReport.

    Args:
        top_n: Number of top transactions to investigate (default 10).
    """
    _load_data()

    if _flagged is None or _flagged.empty:
        return AnomalyInvestigationReport(
            total_flagged=0, investigated=0, verdicts=[],
            summary="No flagged anomalies found.",
            high_count=0, review_count=0,
        )

    total = len(_flagged)
    sort_col = "vote_count" if "vote_count" in _flagged.columns else "abs_amount"
    subset   = _flagged.sort_values(sort_col, ascending=False).head(top_n)

    log.info("Starting investigation: %d flagged, investigating top %d", total, len(subset))

    verdicts: list[AnomalyVerdict] = []
    for _, row in subset.iterrows():
        verdict = investigate_transaction(row)
        verdicts.append(verdict)

    high_count   = sum(1 for v in verdicts if v.suspicion == RiskLevel.HIGH)
    review_count = sum(1 for v in verdicts if v.action in {"REVIEW", "FLAG_FOR_AUDIT"})

    summary = (
        f"Investigated {len(verdicts)} of {total} flagged transactions. "
        f"{high_count} rated HIGH suspicion, {review_count} require immediate review. "
        f"{'Urgent action recommended.' if high_count > 0 else 'No critical anomalies detected.'}"
    )

    log.info("Investigation complete: HIGH=%d REVIEW=%d", high_count, review_count)

    return AnomalyInvestigationReport(
        total_flagged=total,
        investigated=len(verdicts),
        verdicts=verdicts,
        summary=summary,
        high_count=high_count,
        review_count=review_count,
    )
