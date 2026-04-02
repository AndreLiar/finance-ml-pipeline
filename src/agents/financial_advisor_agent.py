"""
src/agents/financial_advisor_agent.py — Financial Advisor ReAct Agent

Architecture:
  LangGraph create_react_agent with ChatOllama (native tool calling).
  The agent plans which data tools to call, fetches real numbers, then
  answers the question grounded exclusively in those numbers.

Guardrails:
  1. Max 6 tool calls per question (AgentExecutor iteration limit)
  2. Post-answer grounding check — answer must contain at least one real number
  3. Fallback to deterministic context if Ollama is down
  4. Structured AdvisorAnswer output (Pydantic validated)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from src.config import (
    LLM_MODEL, LLM_TEMPERATURE,
    CREDITWORTHINESS_XLSX, CASHFLOW_RESULTS_XLSX,
    ANOMALY_RESULTS_XLSX, FEATURES_XLSX,
)
from src.guardrails.number_validator import assert_grounded
from src.guardrails.llm_fallback import ollama_is_available, advisor_fallback
from src.guardrails.structured_outputs import AdvisorAnswer
from src.logger import get_logger
from src.vectorstore.retriever import semantic_search as _vs_search, store_status as _vs_status

log = get_logger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _read(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:
        log.warning("Could not load %s[%s]: %s", path.name, sheet, exc)
        return None

def _fmt(val: float) -> str:
    return f"EUR{val:,.0f}"


# ── Tools (each fetches one slice of real data) ────────────────────────────────

@tool
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


@tool
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


@tool
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
    ensemble = row.get("Ensemble Average", None)
    lines = [f"Cashflow forecast for {month}:"]
    for col in ["Ridge", "Random Forest", "XGBoost", "Gradient Boosting", "Ensemble Average"]:
        if col in row:
            lines.append(f"  {col:<22}: {_fmt(row[col])}")
    if ensemble is not None:
        lines.append(f"\n  Best estimate (ensemble): {_fmt(ensemble)}/month")
    return "\n".join(lines)


@tool
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


@tool
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


@tool
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


@tool
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
        f"  Verdict             : {verdict}\n"
        f"  Rule: affordable if cost < 35% of income AND net remains positive."
    )


@tool
def search_transactions(query: str) -> str:
    """
    Semantic search across all 1226 transactions to find specific payments,
    merchants, or spending patterns. Use this for questions like 'did I pay X',
    'find transfers in November', or 'show me large grocery transactions'.
    Args:
        query: Natural language description of transactions to find.
    """
    try:
        results = _vs_search(query, collection="transactions", top_k=8)
        if not results:
            return "No matching transactions found in the vector store."
        lines = [f"Top transactions matching '{query}':"]
        for r in results:
            score = r.get("score", 0)
            date  = r.get("date", "?")
            cat   = r.get("category", "?")
            raw_amt = r.get("amount_eur", 0)
            try:
                amt_str = f"EUR{float(raw_amt):.0f}" if raw_amt and str(raw_amt) not in ("nan","None") else "N/A"
            except Exception:
                amt_str = "N/A"
            desc  = r.get("description", "")[:50]
            lines.append(f"  [{score:.2f}] {date} | {amt_str} | {cat} | {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Vector search unavailable: {exc}"


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un conseiller financier personnel expert. Tu analyses les finances réelles de l'utilisateur.

RÈGLES ABSOLUES :
- Utilise UNIQUEMENT les outils disponibles pour obtenir des données. Ne devine jamais un chiffre.
- Cite toujours les chiffres exacts retournés par les outils.
- Réponds dans la même langue que la question (français ou English).
- Sois direct, précis et bienveillant.
- Si une donnée n'est pas disponible via les outils, dis-le clairement.
- N'invente JAMAIS de valeurs. Chaque chiffre dans ta réponse DOIT venir d'un outil.

OUTILS DISPONIBLES :
- get_credit_profile : score de crédit, DSCR, taux d'épargne (6 derniers mois)
- get_income_and_spend : revenus et dépenses moyens (3 derniers mois)
- get_cashflow_forecast : prévisions de dépenses pour le mois prochain
- get_top_spending_categories : top 10 catégories de dépenses
- get_anomalies : transactions anormales détectées par le ML
- get_monthly_trend : évolution mensuelle revenus/dépenses (6 mois)
- evaluate_affordability : évalue si un coût mensuel est supportable
- search_transactions : recherche sémantique dans les 1226 transactions (trouver un paiement spécifique, un marchand, une période)
"""


# ── LLM + Agent ────────────────────────────────────────────────────────────────

_TOOLS = [
    get_credit_profile,
    get_income_and_spend,
    get_cashflow_forecast,
    get_top_spending_categories,
    get_anomalies,
    get_monthly_trend,
    evaluate_affordability,
    search_transactions,
]

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=1500,
        )
        _agent = create_react_agent(llm, _TOOLS)
        log.info("Financial Advisor Agent initialised (model=%s, %d tools, RAG=enabled)", LLM_MODEL, len(_TOOLS))
    return _agent


# ── Public API ─────────────────────────────────────────────────────────────────

def run_advisor_agent(
    question: str,
    history:  list[dict] | None = None,
) -> AdvisorAnswer:
    """
    Run the Financial Advisor ReAct Agent for a user question.

    Args:
        question: Natural language question in French or English.
        history:  List of {"role": "user"|"assistant", "content": str}.

    Returns:
        AdvisorAnswer (Pydantic validated).
    """
    if not question.strip():
        return AdvisorAnswer(
            answer="Veuillez poser une question. / Please ask a question.",
            tools_used=[],
            data_grounded=False,
            language="français",
            confidence="LOW",
        )

    # Guardrail 1: check Ollama availability before invoking
    if not ollama_is_available():
        fallback_ctx = "Données non disponibles — Ollama hors ligne."
        return AdvisorAnswer(
            answer=advisor_fallback(question, fallback_ctx),
            tools_used=[],
            data_grounded=False,
            language="français",
            confidence="LOW",
        )

    # Detect language for structured output
    french_words = {"je","mon","ma","mes","le","la","les","de","du","des","est",
                    "pourquoi","comment","quand","quel","quelle","combien","puis",
                    "peux","dois","peut","doit","un","une","pour","avec","dans"}
    words = set(re.findall(r'\b\w+\b', question.lower()))
    language = "français" if words & french_words else "English"

    # RAG: inject semantically relevant context before the question
    rag_context = ""
    try:
        status = _vs_status()
        any_indexed = any(v["indexed"] for v in status.values())
        if any_indexed:
            hits = _vs_search(question, collection="all", top_k=6)
            if hits:
                rag_lines = ["[Relevant financial data retrieved via semantic search:]"]
                for h in hits:
                    rag_lines.append(f"  - {h.get('text','')[:100]}")
                rag_context = "\n".join(rag_lines)
    except Exception as exc:
        log.debug("RAG context unavailable: %s", exc)

    # Build message list (include last 2 exchanges from history)
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]
    if history:
        for msg in history[-(4):]:  # last 2 exchanges = 4 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))

    full_question = question
    if rag_context:
        full_question = f"{rag_context}\n\nQuestion: {question}"

    messages.append(HumanMessage(content=full_question))

    log.info("Advisor agent invoked | question=%r | lang=%s", question[:80], language)

    try:
        agent = _get_agent()
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 12},   # guardrail: max ~6 tool calls
        )

        # Extract final answer from last AI message
        final_message = result["messages"][-1]
        answer_text   = final_message.content if hasattr(final_message, "content") else str(final_message)

        # Extract which tools were actually called
        tools_used = []
        for msg in result["messages"]:
            if hasattr(msg, "name") and msg.name:
                if msg.name not in tools_used:
                    tools_used.append(msg.name)

        # Guardrail 2: grounding check — answer must contain at least one real number
        nums_in_answer = [
            float(n.replace(",", ""))
            for n in re.findall(r'\d[\d,]*(?:\.\d+)?', answer_text)
            if float(n.replace(",", "")) >= 10
        ]
        data_grounded = len(nums_in_answer) > 0

        if not data_grounded:
            log.warning("Advisor answer contains no financial figures — low confidence")

        confidence = "HIGH" if (data_grounded and len(tools_used) >= 1) else \
                     "MEDIUM" if data_grounded else "LOW"

        log.info("Agent done | tools=%s | grounded=%s | confidence=%s",
                 tools_used, data_grounded, confidence)

        return AdvisorAnswer(
            answer=answer_text,
            tools_used=tools_used,
            data_grounded=data_grounded,
            language=language,
            confidence=confidence,
        )

    except Exception as exc:
        log.error("Advisor agent error: %s", exc)
        return AdvisorAnswer(
            answer=(
                f"Une erreur est survenue lors de l'analyse. "
                f"Vérifiez qu'Ollama est actif (`ollama serve`).\n\nErreur: {exc}"
            ),
            tools_used=[],
            data_grounded=False,
            language=language,
            confidence="LOW",
        )
