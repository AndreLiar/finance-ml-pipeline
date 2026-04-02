"""
src/mcp/finance_mcp_server.py — Finance MCP Server (FastMCP)

Exposes all financial analysis tools via the Model Context Protocol so
Claude Desktop (or any MCP-compatible LLM) can call them directly.

Tools exposed:
  1. get_credit_profile         — credit score, DSCR, savings rate (6 months)
  2. get_income_and_spend       — avg monthly income & spend (3 months)
  3. get_cashflow_forecast      — ML ensemble forecast for next month
  4. get_top_spending_categories — top 10 categories by spend
  5. get_anomalies              — top flagged anomalous transactions
  6. get_monthly_trend          — month-by-month income/spend (6 months)
  7. evaluate_affordability     — is a monthly cost affordable?
  8. search_transactions        — semantic vector search across all 1226 transactions
  9. get_pipeline_status        — last pipeline run status (from pipeline_status.json)
 10. get_anomaly_investigation  — AI-investigated verdict for top-N anomalies

Run:
  py -3 -m src.mcp.finance_mcp_server          # stdio (Claude Desktop)
  py -3 -m src.mcp.finance_mcp_server --http   # HTTP on port 8052

Claude Desktop config (~/.config/claude/claude_desktop_config.json on Mac,
%APPDATA%\\Claude\\claude_desktop_config.json on Windows):
  {
    "mcpServers": {
      "finance": {
        "command": "py",
        "args": ["-3", "-m", "src.mcp.finance_mcp_server"],
        "cwd": "C:\\\\Users\\\\ktayl\\\\Desktop\\\\devprojectall\\\\FinanceProjects"
      }
    }
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    CREDITWORTHINESS_XLSX, CASHFLOW_RESULTS_XLSX,
    ANOMALY_RESULTS_XLSX, FEATURES_XLSX,
    PIPELINE_STATUS_JSON,
)

mcp = FastMCP(
    name="FinanceAdvisor",
    instructions=(
        "You have access to a personal finance analysis system covering 1226 bank transactions "
        "from a French Banque Populaire account. All data is local and private. "
        "Use these tools to answer questions about spending, income, credit risk, "
        "anomalies, cashflow forecasts, and affordability."
    ),
)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _read(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return None

def _fmt(val: float) -> str:
    try:
        return f"EUR{float(val):,.0f}"
    except Exception:
        return "EUR?"


# ── Tool 1: Credit profile ─────────────────────────────────────────────────────

@mcp.tool
def get_credit_profile() -> str:
    """
    Get the applicant's credit score, risk label, DSCR, and savings rate
    for the last 6 months. Use this for credit score or risk questions.
    """
    df = _read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Credit profile data not available."
    df = df.sort_values("year_month", ascending=False).head(6)
    lines = ["Credit profile (last 6 months):"]
    for _, r in df.iterrows():
        savings = r.get("savings_rate", 0)
        savings_pct = savings * 100 if abs(savings) <= 1.0 else savings
        lines.append(
            f"  {r['year_month']}: score={r.get('credit_score',0):.0f}/100 "
            f"label={r.get('credit_label','?')} "
            f"DSCR={r.get('dscr',0):.2f}x "
            f"savings={savings_pct:.1f}% "
            f"income={_fmt(r.get('avg_3m_income', r.get('income',0)))} "
            f"spend={_fmt(r.get('avg_3m_spend', r.get('spend',0)))}"
        )
    return "\n".join(lines)


# ── Tool 2: Income and spend ───────────────────────────────────────────────────

@mcp.tool
def get_income_and_spend() -> str:
    """
    Get average monthly income and spend for the last 3 months.
    Use this for affordability, budget, or income questions.
    """
    df = _read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Income data not available."
    df = df.sort_values("year_month", ascending=False).head(3)
    income_col = "avg_3m_income" if "avg_3m_income" in df.columns else "income"
    spend_col  = "avg_3m_spend"  if "avg_3m_spend"  in df.columns else "spend"
    avg_income = df[income_col].mean()
    avg_spend  = df[spend_col].mean()
    net        = avg_income - avg_spend
    return (
        f"Last 3-month averages:\n"
        f"  Income : {_fmt(avg_income)}/month\n"
        f"  Spend  : {_fmt(avg_spend)}/month\n"
        f"  Net    : {_fmt(net)}/month ({'surplus' if net >= 0 else 'DEFICIT'})"
    )


# ── Tool 3: Cashflow forecast ──────────────────────────────────────────────────

@mcp.tool
def get_cashflow_forecast() -> str:
    """
    Get the ML cashflow forecast for next month (ensemble of Ridge, RF, XGBoost,
    Gradient Boosting). Use this for budget planning or 'can I afford' questions.
    """
    df = _read(CASHFLOW_RESULTS_XLSX, "Next Month Forecast")
    if df is None or df.empty:
        return "Cashflow forecast not available."
    row = df.iloc[0]
    month = row.get("Month", "next month")
    lines = [f"Cashflow forecast for {month}:"]
    for col in ["Ridge", "Random Forest", "XGBoost", "Gradient Boosting", "Ensemble Average"]:
        if col in row:
            lines.append(f"  {col:<22}: {_fmt(row[col])}")
    ensemble = row.get("Ensemble Average")
    if ensemble is not None:
        lines.append(f"\n  Best estimate (ensemble): {_fmt(ensemble)}/month")
    return "\n".join(lines)


# ── Tool 4: Top spending categories ───────────────────────────────────────────

@mcp.tool
def get_top_spending_categories() -> str:
    """
    Get the top spending categories by total amount spent.
    Use this for 'where does my money go' or spending breakdown questions.
    """
    df = _read(FEATURES_XLSX, "Category Summary")
    if df is None or df.empty:
        return "Category data not available."
    spend_col = "total_spend" if "total_spend" in df.columns else "total_debit"
    count_col = "count"       if "count"       in df.columns else "transaction_count"
    df = df.sort_values(spend_col, ascending=False).head(10)
    lines = ["Top 10 spending categories:"]
    for _, r in df.iterrows():
        lines.append(
            f"  {str(r['category']):<20}: {_fmt(r.get(spend_col,0))} "
            f"({int(r.get(count_col,0))} transactions)"
        )
    return "\n".join(lines)


# ── Tool 5: Anomalies ─────────────────────────────────────────────────────────

@mcp.tool
def get_anomalies() -> str:
    """
    Get the top 10 flagged anomalous transactions sorted by suspicion score.
    Use this for fraud, unusual spending, or anomaly questions.
    """
    df = _read(ANOMALY_RESULTS_XLSX, "Flagged Anomalies")
    if df is None or df.empty:
        return "No anomalies detected."
    total = len(df)
    sort_col = "vote_count" if "vote_count" in df.columns else "abs_amount"
    df = df.sort_values(sort_col, ascending=False).head(10)
    lines = [f"Top anomalous transactions ({total} flagged in total):"]
    for _, r in df.iterrows():
        try:
            date = pd.to_datetime(r["date_operation"]).strftime("%Y-%m-%d")
        except Exception:
            date = str(r.get("date_operation", "?"))
        amt   = r.get("debit", r.get("abs_amount", 0))
        votes = r.get("vote_count", r.get("anomaly_score", 0))
        desc  = str(r.get("description", ""))[:40]
        cat   = r.get("category", "?")
        lines.append(f"  {date} | {_fmt(amt):>10} | {cat:<15} | {votes:.0f}/3 | {desc}")
    return "\n".join(lines)


# ── Tool 6: Monthly trend ─────────────────────────────────────────────────────

@mcp.tool
def get_monthly_trend() -> str:
    """
    Get income, spend, and net balance for each of the last 6 months.
    Use this for trend analysis, 'am I improving', or month-by-month questions.
    """
    df = _read(FEATURES_XLSX, "Monthly Aggregates")
    if df is None or df.empty:
        return "Monthly trend data not available."
    df = df.sort_values("year_month", ascending=False).head(6)
    lines = ["Monthly income vs spend (last 6 months):"]
    for _, r in df.iterrows():
        income = r.get("monthly_income", 0)
        spend  = r.get("monthly_spend",  0)
        net    = r.get("monthly_net", income - spend)
        lines.append(
            f"  {r['year_month']}: "
            f"income={_fmt(income)} spend={_fmt(spend)} net={_fmt(net)}"
        )
    return "\n".join(lines)


# ── Tool 7: Affordability ─────────────────────────────────────────────────────

@mcp.tool
def evaluate_affordability(monthly_cost: float) -> str:
    """
    Evaluate whether a recurring monthly cost (rent, loan repayment, subscription)
    is affordable given current income and spend patterns.

    Args:
        monthly_cost: The monthly cost in EUR to evaluate.
    """
    df = _read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Cannot evaluate affordability — data not available."
    df = df.sort_values("year_month", ascending=False).head(3)
    income_col = "avg_3m_income" if "avg_3m_income" in df.columns else "income"
    spend_col  = "avg_3m_spend"  if "avg_3m_spend"  in df.columns else "spend"
    avg_income = df[income_col].mean()
    avg_spend  = df[spend_col].mean()
    current_net = avg_income - avg_spend
    new_net     = current_net - monthly_cost
    ratio       = monthly_cost / avg_income if avg_income > 0 else float("inf")
    verdict = "AFFORDABLE" if new_net > 0 and ratio < 0.35 else \
              "TIGHT"      if new_net > 0 else "NOT AFFORDABLE"
    return (
        f"Affordability analysis for {_fmt(monthly_cost)}/month:\n"
        f"  Current avg income  : {_fmt(avg_income)}/month\n"
        f"  Current avg spend   : {_fmt(avg_spend)}/month\n"
        f"  Current net         : {_fmt(current_net)}/month\n"
        f"  Net after new cost  : {_fmt(new_net)}/month\n"
        f"  Cost as % of income : {ratio:.1%}\n"
        f"  Verdict             : {verdict}"
    )


# ── Tool 8: Semantic transaction search ───────────────────────────────────────

@mcp.tool
def search_transactions(query: str, top_k: int = 8) -> str:
    """
    Semantic vector search across all 1226 transactions to find specific
    payments, merchants, or spending patterns.

    Args:
        query: Natural language description — e.g. 'Netflix subscription',
               'large transfer in November', 'grocery shopping over 100 euros'.
        top_k: Number of results to return (default 8, max 20).
    """
    try:
        from src.vectorstore.retriever import semantic_search
        top_k = min(int(top_k), 20)
        results = semantic_search(query, collection="transactions", top_k=top_k)
        if not results:
            return "No matching transactions found in the vector store. Run the indexer first: py -3 -m src.vectorstore.indexer"
        lines = [f"Top {len(results)} transactions matching '{query}':"]
        for r in results:
            score = r.get("score", 0)
            date  = r.get("date", "?")
            cat   = r.get("category", "?")
            raw_amt = r.get("amount_eur", 0)
            try:
                amt_str = _fmt(float(raw_amt)) if raw_amt and str(raw_amt) not in ("nan", "None") else "N/A"
            except Exception:
                amt_str = "N/A"
            desc = r.get("description", "")[:60]
            lines.append(f"  [{score:.2f}] {date} | {amt_str} | {cat} | {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Vector search error: {exc}"


# ── Tool 9: Pipeline status ────────────────────────────────────────────────────

@mcp.tool
def get_pipeline_status() -> str:
    """
    Get the status of the last ML pipeline run — which stages succeeded,
    failed, were skipped, or auto-recovered after retry.
    """
    if not PIPELINE_STATUS_JSON.exists():
        return "No pipeline status found. Run: py -3 -m src.agents.pipeline_supervisor"
    try:
        data = json.loads(PIPELINE_STATUS_JSON.read_text(encoding="utf-8"))
        lines = [
            f"Pipeline run at: {data.get('run_at', '?')}",
            f"Overall status : {data.get('overall_status', '?')}",
            f"Summary        : {data.get('incident_summary', '?')}",
            "",
            "Stage results:",
        ]
        for s in data.get("stages", []):
            note = f" ({s['note']})" if s.get("note") else ""
            lines.append(
                f"  [{s.get('id','?'):>2}] {s.get('name','?'):<25} "
                f"{s.get('outcome','?'):<10} retries={s.get('retries',0)}{note}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Could not read pipeline status: {exc}"


# ── Tool 10: Anomaly investigation ────────────────────────────────────────────

@mcp.tool
def get_anomaly_investigation(top_n: int = 5) -> str:
    """
    Run AI-powered investigation on the top-N most suspicious anomalies.
    Each transaction is investigated by a ReAct agent that checks category stats,
    historical patterns, and month context, then returns a structured verdict.

    WARNING: This calls the Ollama LLM and may take 1-2 minutes for top_n=5.
    Ollama must be running (`ollama serve`).

    Args:
        top_n: Number of top anomalies to investigate (default 5, max 10).
    """
    try:
        from src.agents.anomaly_investigator import run_investigation
        top_n = min(int(top_n), 10)
        report = run_investigation(top_n=top_n)
        lines = [
            f"Anomaly Investigation Report",
            f"  Total flagged  : {report.total_flagged}",
            f"  Investigated   : {report.investigated}",
            f"  HIGH suspicion : {report.high_count}",
            f"  Need review    : {report.review_count}",
            f"  Summary        : {report.summary}",
            "",
            "Verdicts:",
        ]
        for v in report.verdicts:
            lines.append(
                f"  {v.date} | {_fmt(v.amount_eur)} | {v.category:<15} | "
                f"{v.suspicion.value:<6} | {v.action:<15} | {v.reason[:80]}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Investigation failed: {exc}"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finance MCP Server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP server on port 8052")
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()

    if args.http:
        print(f"Starting Finance MCP Server (HTTP) on port {args.port}...")
        mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        # Default: stdio mode for Claude Desktop
        mcp.run(transport="stdio")
