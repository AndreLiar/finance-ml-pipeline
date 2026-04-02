"""
src/guardrails/llm_fallback.py — Fallback responses when Ollama is unreachable.

All agents call these instead of crashing. Returns deterministic ML output
so the system degrades gracefully rather than returning nothing.
"""

from __future__ import annotations
import requests
from src.logger import get_logger

log = get_logger(__name__)

OLLAMA_URL = "http://localhost:11434"


def ollama_is_available(timeout: float = 2.0) -> bool:
    """Return True if Ollama is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def model_is_available(model: str, timeout: float = 2.0) -> bool:
    """Return True if the given model is pulled and available."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return False
        models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return model.split(":")[0] in models
    except Exception:
        return False


def advisor_fallback(question: str, context_summary: str) -> str:
    """
    Deterministic fallback for the Financial Advisor when Ollama is down.
    Returns a structured answer built from the data context alone.
    """
    log.warning("Ollama unavailable — returning deterministic advisor fallback")
    return (
        f"[Mode dégradé — LLM indisponible]\n\n"
        f"Votre question : {question}\n\n"
        f"Données disponibles :\n{context_summary}\n\n"
        f"Le service LLM (Ollama/Mistral) n'est pas joignable. "
        f"Lancez `ollama serve` puis réessayez. "
        f"Les données ci-dessus sont exactes et issues directement de votre pipeline ML."
    )


def anomaly_fallback(transaction_desc: str, amount: float, category_avg: float) -> dict:
    """
    Deterministic fallback for the Anomaly Investigator when Ollama is down.
    Uses simple ratio rule: >2x average = HIGH, >1.5x = MEDIUM, else LOW.
    """
    log.warning("Ollama unavailable — returning rule-based anomaly verdict for '%s'", transaction_desc)
    if category_avg > 0:
        ratio = amount / category_avg
        if ratio > 2.0:
            suspicion, action = "HIGH", "FLAG_FOR_AUDIT"
        elif ratio > 1.5:
            suspicion, action = "MEDIUM", "REVIEW"
        else:
            suspicion, action = "LOW", "MONITOR"
        reason = f"{ratio:.1f}x above category average of EUR{category_avg:.0f} (rule-based, LLM unavailable)"
    else:
        suspicion, action = "MEDIUM", "REVIEW"
        reason = "Category average unavailable — flagged for manual review (LLM unavailable)"

    return {"suspicion": suspicion, "action": action, "reason": reason}


def pipeline_fallback_note(stage: str, error: str) -> str:
    """Plain-language note for a pipeline stage that failed without LLM reasoning."""
    return (
        f"[Fallback] Stage '{stage}' failed: {error[:120]}. "
        f"LLM reasoning unavailable — manual review required."
    )
