"""
src/pipeline/financial_advisor.py — Financial Advisor Chat (RAG-style, LangChain + Ollama/Mistral)

Architecture:
  1. Intent detection  — keyword routing to decide which data to load
  2. Context builder   — loads relevant xlsx data, formats as plain text (fits in prompt)
  3. LangChain chain   — ConversationChain with OllamaLLM + custom PromptTemplate
  4. ask_advisor()     — public API called by the Streamlit dashboard

All inference is local via Ollama/Mistral. No data leaves the machine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import (
    ROOT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    FEATURES_XLSX,
    CREDITWORTHINESS_XLSX,
    CASHFLOW_RESULTS_XLSX,
    ANOMALY_RESULTS_XLSX,
    TRANSACTIONS_XLSX,
)
from src.logger import get_logger

log = get_logger(__name__)

# ── Prompt template path ───────────────────────────────────────────────────────
PROMPT_TEMPLATE_PATH = ROOT / "src" / "prompts" / "financial_advisor_v1.txt"

# ── LangChain imports (LCEL — LangChain Expression Language) ──────────────────
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm: OllamaLLM | None = None


def get_llm() -> OllamaLLM:
    """Return a cached OllamaLLM instance."""
    global _llm
    if _llm is None:
        log.info("Initialising Ollama LLM: model=%s temperature=%.2f", LLM_MODEL, LLM_TEMPERATURE)
        _llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
        )
    return _llm


# ── Intent detection ───────────────────────────────────────────────────────────
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "credit_score": [
        "score", "crédit", "credit", "risque", "risk", "rating",
        "solvabilité", "solvability", "notation",
    ],
    "anomaly": [
        "anomalie", "anomaly", "anormal", "unusual", "suspect", "suspicious",
        "fraude", "fraud", "bizarre", "strange", "inhabituel",
    ],
    "cashflow": [
        "prévision", "forecast", "futur", "budget", "next month",
        "mois prochain", "projection", "prédiction", "predict",
    ],
    "spending": [
        "dépenses", "spending", "catégorie", "category", "groceries",
        "transport", "alimentation", "loyers", "abonnement", "subscription",
        "combien je dépense", "how much do i spend",
    ],
    "affordability": [
        "loyer", "rent", "afford", "peux-je", "can i", "puis-je",
        "capacité", "emprunt", "loan", "crédit immobilier", "mortgage",
        "€ par mois", "per month",
    ],
    "income": [
        "salaire", "salary", "revenu", "income", "paie", "pay",
        "employeur", "employer", "virement", "transfer",
    ],
    "anomaly_detail": [
        "transaction", "spécifique", "specific", "detail", "détail",
        "liste", "list", "quelles transactions", "which transactions",
    ],
}


def detect_intents(question: str) -> list[str]:
    """Return list of matched intent tags (order preserved, deduped)."""
    q_lower = question.lower()
    matched: list[str] = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.append(intent)
    return matched if matched else ["general"]


# ── Context builders ───────────────────────────────────────────────────────────

def _safe_read(path: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as exc:
        log.warning("Could not load %s[%s]: %s", path.name, sheet, exc)
        return None


def _fmt_eur(val: float) -> str:
    return f"{val:,.0f}€"


def _build_credit_context(n_months: int = 6) -> str:
    df = _safe_read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Données de score de crédit non disponibles."

    # Column is year_month in this xlsx
    sort_col = "year_month" if "year_month" in df.columns else "month"
    df = df.sort_values(sort_col, ascending=False).head(n_months)
    lines = ["Profil crédit mensuel (derniers {} mois) :".format(len(df))]
    lines.append(f"{'Mois':<10} {'Score':>6} {'Label':<15} {'DSCR':>6} {'Épargne':>8} {'Revenus':>10} {'Dépenses':>10}")
    lines.append("-" * 70)
    for _, r in df.iterrows():
        mois = r.get("year_month", r.get("month", "?"))
        savings = r.get("savings_rate", 0)
        # savings_rate may be a fraction (0-1) or percentage
        savings_pct = savings * 100 if abs(savings) <= 1.0 else savings
        lines.append(
            f"{str(mois):<10} "
            f"{r.get('credit_score', 0):>6.0f} "
            f"{str(r.get('credit_label','?')):<15} "
            f"{r.get('dscr', 0):>6.2f} "
            f"{savings_pct:>7.1f}% "
            f"{_fmt_eur(r.get('avg_3m_income', r.get('income', 0))):>10} "
            f"{_fmt_eur(r.get('avg_3m_spend', r.get('spend', 0))):>10}"
        )
    return "\n".join(lines)


def _build_anomaly_context(limit: int = 10) -> str:
    df = _safe_read(ANOMALY_RESULTS_XLSX, "Flagged Anomalies")
    if df is None or df.empty:
        return "Aucune transaction anormale détectée."

    total = len(df)
    # Sort by vote_count (consensus) then abs_amount
    sort_col = "vote_count" if "vote_count" in df.columns else "abs_amount"
    df = df.sort_values(sort_col, ascending=False).head(limit)
    lines = [f"Transactions flaggées comme anomalies ({total} au total, top {len(df)} affichées) :"]
    lines.append(f"{'Date':<12} {'Montant':>10} {'Description':<35} {'Votes':>6}")
    lines.append("-" * 70)
    for _, r in df.iterrows():
        date_val = r.get("date_operation")
        try:
            date = pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except Exception:
            date = str(date_val)
        amt = r.get("debit", r.get("abs_amount", 0))
        desc = str(r.get("description", ""))[:34]
        votes = r.get("vote_count", r.get("anomaly_score", 0))
        lines.append(f"{date:<12} {_fmt_eur(amt):>10} {desc:<35} {votes:>6.0f}/3")
    return "\n".join(lines)


def _build_cashflow_context() -> str:
    # Next Month Forecast cols: Month, Ridge, Random Forest, XGBoost, Gradient Boosting, Ensemble Average
    forecast = _safe_read(CASHFLOW_RESULTS_XLSX, "Next Month Forecast")
    # Actual vs Predicted cols: Month, Actual Spend (€), Baseline, Ridge, Random Forest, XGBoost, Gradient Boosting
    actuals  = _safe_read(CASHFLOW_RESULTS_XLSX, "Actual vs Predicted")
    lines = ["Prévisions de trésorerie :"]

    if forecast is not None and not forecast.empty:
        row = forecast.iloc[0]
        month = row.get("Month", "?")
        ensemble = row.get("Ensemble Average", None)
        if ensemble is not None:
            lines.append(f"  Prévision mois prochain ({month})  : {_fmt_eur(ensemble)} (moyenne ensemble)")
        else:
            # Fall back to first numeric column
            numeric = forecast.select_dtypes("number").iloc[0]
            lines.append(f"  Prévision mois prochain ({month}) : {_fmt_eur(numeric.iloc[0])}")

    if actuals is not None and not actuals.empty:
        actual_col = next((c for c in actuals.columns if "Actual" in c), None)
        ridge_col  = "Ridge" if "Ridge" in actuals.columns else None
        lines.append("\nHistorique réel vs prédit Ridge (4 derniers mois) :")
        lines.append(f"{'Mois':<10} {'Réel':>10} {'Prédit (Ridge)':>15} {'Écart':>8}")
        lines.append("-" * 48)
        for _, r in actuals.tail(4).iterrows():
            actual = r.get(actual_col, 0) if actual_col else 0
            pred   = r.get(ridge_col, 0) if ridge_col else 0
            delta  = actual - pred
            lines.append(f"{str(r.get('Month','?')):<10} {_fmt_eur(actual):>10} {_fmt_eur(pred):>15} {_fmt_eur(delta):>8}")

    return "\n".join(lines) if len(lines) > 1 else "Données de prévision non disponibles."


def _build_spending_context() -> str:
    # Category Summary cols: category, count, total_spend, avg_amount
    cat_sum = _safe_read(FEATURES_XLSX, "Category Summary")
    # Monthly Aggregates cols: year_month, monthly_income, monthly_spend, monthly_net, tx_count, ...
    monthly = _safe_read(FEATURES_XLSX, "Monthly Aggregates")
    lines = []

    if cat_sum is not None and not cat_sum.empty:
        spend_col = "total_spend" if "total_spend" in cat_sum.columns else "total_debit"
        count_col = "count" if "count" in cat_sum.columns else "transaction_count"
        cat_sum = cat_sum.sort_values(spend_col, ascending=False).head(12)
        lines.append("Top catégories de dépenses :")
        lines.append(f"{'Catégorie':<25} {'Total (€)':>12} {'Nb transactions':>16}")
        lines.append("-" * 55)
        for _, r in cat_sum.iterrows():
            lines.append(
                f"{str(r.get('category','?')):<25} "
                f"{_fmt_eur(r.get(spend_col, 0)):>12} "
                f"{int(r.get(count_col, 0)):>16}"
            )

    if monthly is not None and not monthly.empty:
        monthly = monthly.sort_values("year_month", ascending=False).head(3)
        lines.append("\nRésumé mensuel (3 derniers mois) :")
        for _, r in monthly.iterrows():
            lines.append(
                f"  {r.get('year_month','?')} — "
                f"Dépenses: {_fmt_eur(r.get('monthly_spend', r.get('total_debit', 0)))} | "
                f"Revenus: {_fmt_eur(r.get('monthly_income', r.get('total_credit', 0)))} | "
                f"Transactions: {int(r.get('tx_count', r.get('transaction_count', 0)))}"
            )

    return "\n".join(lines) if lines else "Données de dépenses non disponibles."


def _build_income_context() -> str:
    df = _safe_read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Données de revenus non disponibles."

    sort_col = "year_month" if "year_month" in df.columns else "month"
    df = df.sort_values(sort_col, ascending=False).head(6)
    income_col = "avg_3m_income" if "avg_3m_income" in df.columns else "income"
    avg_income = df[income_col].mean() if income_col in df.columns else 0

    lines = [f"Revenus (derniers 6 mois) — Moyenne: {_fmt_eur(avg_income)}/mois"]
    for _, r in df.iterrows():
        mois = r.get("year_month", r.get("month", "?"))
        lines.append(f"  {mois} : {_fmt_eur(r.get(income_col, 0))}")
    return "\n".join(lines)


def _build_affordability_context() -> str:
    """Combined context for affordability/loan capacity questions."""
    credit = _build_credit_context(n_months=3)
    cashflow = _build_cashflow_context()
    return credit + "\n\n" + cashflow


def _build_general_context() -> str:
    """Compact 3-month summary for open-ended questions."""
    df = _safe_read(CREDITWORTHINESS_XLSX, "Monthly Credit Profile")
    if df is None or df.empty:
        return "Données insuffisantes pour un résumé."

    sort_col   = "year_month" if "year_month" in df.columns else "month"
    income_col = "avg_3m_income" if "avg_3m_income" in df.columns else "income"
    spend_col  = "avg_3m_spend"  if "avg_3m_spend"  in df.columns else "spend"

    df = df.sort_values(sort_col, ascending=False).head(3)
    latest = df.iloc[0]
    avg_income = df[income_col].mean() if income_col in df.columns else 0
    avg_spend  = df[spend_col].mean()  if spend_col  in df.columns else 0
    label = latest.get("credit_label", "N/A")
    score = latest.get("credit_score", 0)

    return (
        f"Résumé financier récent :\n"
        f"  Score de crédit actuel : {score:.0f} ({label})\n"
        f"  Revenu moyen (3 mois)  : {_fmt_eur(avg_income)}/mois\n"
        f"  Dépenses moyennes      : {_fmt_eur(avg_spend)}/mois\n"
        f"  Solde disponible moyen : {_fmt_eur(avg_income - avg_spend)}/mois"
    )


_CONTEXT_BUILDERS = {
    "credit_score":   _build_credit_context,
    "anomaly":        _build_anomaly_context,
    "anomaly_detail": _build_anomaly_context,
    "cashflow":       _build_cashflow_context,
    "spending":       _build_spending_context,
    "income":         _build_income_context,
    "affordability":  _build_affordability_context,
    "general":        _build_general_context,
}


def build_context(intents: list[str]) -> str:
    """Retrieve and concatenate context blocks for all matched intents."""
    seen: set[str] = set()
    blocks: list[str] = []
    for intent in intents:
        builder = _CONTEXT_BUILDERS.get(intent, _build_general_context)
        # Avoid calling the same builder twice (e.g. anomaly + anomaly_detail)
        key = builder.__name__
        if key in seen:
            continue
        seen.add(key)
        block = builder()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks) if blocks else _build_general_context()


# ── Language detection ─────────────────────────────────────────────────────────
_FRENCH_WORDS = {
    "je", "mon", "ma", "mes", "le", "la", "les", "de", "du", "des",
    "est", "sont", "avec", "pour", "dans", "sur", "par", "en", "un", "une",
    "pourquoi", "comment", "quand", "quel", "quelle", "combien", "puis",
    "peux", "dois", "peut", "doit",
}


def detect_language(question: str) -> str:
    words = set(re.findall(r'\b\w+\b', question.lower()))
    return "français" if words & _FRENCH_WORDS else "English"


# ── LangChain chain builder (LCEL pipe syntax) ────────────────────────────────
def _build_chain():
    """Build an LCEL chain: PromptTemplate | OllamaLLM | StrOutputParser."""
    template_text = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")

    prompt = PromptTemplate(
        input_variables=["language", "context_block", "history_block", "question"],
        template=template_text,
    )
    llm = get_llm()
    return prompt | llm | StrOutputParser()


# Module-level chain (lazy init)
_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain


# ── Public API ─────────────────────────────────────────────────────────────────

def ask_advisor(question: str, history: list[dict] | None = None) -> str:
    """
    Main entry point for the Financial Advisor Chat.

    Args:
        question: The user's natural language question.
        history:  List of {"role": "user"|"assistant", "content": str} dicts.
                  Last 3 exchanges are included in the prompt.

    Returns:
        The LLM's answer as a string.
    """
    if not question.strip():
        return "Veuillez poser une question. / Please ask a question."

    # 1. Detect intent and language
    intents  = detect_intents(question)
    language = detect_language(question)
    log.info("question=%r intents=%s language=%s", question[:80], intents, language)

    # 2. Build context
    context_block = build_context(intents)

    # 3. Format conversation history (last 3 exchanges = 6 messages)
    history_block = _format_history(history or [], max_exchanges=3)

    # 4. Run LangChain LCEL chain — returns str directly via StrOutputParser
    chain = _get_chain()
    try:
        answer = chain.invoke({
            "language":      language,
            "context_block": context_block,
            "history_block": history_block,
            "question":      question,
        })
        answer = answer.strip()
        log.info("LLM response: %d chars", len(answer))
        return answer
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        return (
            f"Le service LLM n'est pas disponible. Vérifiez qu'Ollama est en cours d'exécution "
            f"(`ollama serve`) avec le modèle `{LLM_MODEL}` installé (`ollama pull {LLM_MODEL}`).\n\n"
            f"Erreur : {exc}"
        )


def _format_history(history: list[dict], max_exchanges: int = 3) -> str:
    """Format last N exchanges as plain text for the prompt."""
    if not history:
        return "Aucun historique."
    # Each exchange = 1 user + 1 assistant message
    recent = history[-(max_exchanges * 2):]
    lines = []
    for msg in recent:
        role = "Utilisateur" if msg["role"] == "user" else "Conseiller"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
