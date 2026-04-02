"""
src/agents/pipeline_supervisor.py — AI Pipeline Supervisor Agent

Replaces the dumb subprocess loop in run_pipeline.py with an LLM-driven
orchestrator that reasons about failures and decides: retry / skip / halt.

Architecture:
  - Reads pipeline stage definitions from run_pipeline.py config
  - Runs each stage as a subprocess tool
  - On failure: calls the LLM to reason about the error and decide recovery strategy
  - Enforces guardrails: max 2 retries per stage, never retry data-integrity stages
  - Writes a PipelineReport (Pydantic validated) and pipeline_status.json

Guardrails:
  - Max 2 retries per stage (hard limit, not LLM-controlled)
  - parse_statements never retried (risk of partial data corruption)
  - LLM recovery reasoning is advisory — final decision enforced by code rules
  - Fallback to deterministic skip logic if Ollama is down
  - PipelineReport schema enforced via Pydantic
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config import PIPELINE_STATUS_JSON, ROOT
from src.guardrails.structured_outputs import (
    PipelineReport, StageResult, StageOutcome,
)
from src.guardrails.llm_fallback import ollama_is_available, pipeline_fallback_note
from src.logger import get_logger

log = get_logger(__name__)

# ── Stage definitions (mirrors run_pipeline.py) ────────────────────────────────

STAGES = [
    {"id": 1, "name": "parse_statements",    "script": "src/pipeline/parse_statements.py",    "optional": False, "no_retry": True},
    {"id": 2, "name": "drift_check",         "script": "src/pipeline/drift_check.py",         "optional": True,  "no_retry": False},
    {"id": 3, "name": "parse_livret_a",      "script": "src/pipeline/parse_livret_a.py",      "optional": True,  "no_retry": False},
    {"id": 4, "name": "feature_engineering", "script": "src/pipeline/feature_engineering.py", "optional": False, "no_retry": False},
    {"id": 5, "name": "train_models",        "script": "src/pipeline/train_models.py",        "optional": False, "no_retry": False},
    {"id": 6, "name": "nlp_classifier",      "script": "src/pipeline/nlp_classifier.py",      "optional": False, "no_retry": False},
    {"id": 7, "name": "creditworthiness",    "script": "src/pipeline/creditworthiness.py",    "optional": False, "no_retry": False},
    {"id": 8, "name": "cashflow_forecast",   "script": "src/pipeline/cashflow_forecast.py",   "optional": False, "no_retry": False},
    {"id": 9, "name": "anomaly_detection",   "script": "src/pipeline/anomaly_detection.py",   "optional": False, "no_retry": False},
    {"id":10, "name": "loan_report",         "script": "src/pipeline/loan_report.py",         "optional": True,  "no_retry": False},
]

MAX_RETRIES = 2   # hard guardrail — never more than 2 retries regardless of LLM advice


# ── LLM for recovery reasoning ─────────────────────────────────────────────────

_RECOVERY_SYSTEM = """You are a DevOps AI assistant monitoring an ML pipeline.
A stage has failed. Analyse the error and respond with exactly one of:

RETRY     — the error is transient (memory spike, file lock, network blip) and retrying may work
SKIP      — the stage is optional or the error is non-critical; pipeline can continue without it
HALT      — the error is critical (data corruption, missing required input); pipeline must stop

Respond with ONE word only: RETRY, SKIP, or HALT.
Then on the next line, one sentence explaining why.

Example:
RETRY
The error is a temporary file lock that should clear on the next attempt.
"""

_llm_cache: ChatOllama | None = None

def _get_llm() -> ChatOllama:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatOllama(model="mistral", temperature=0.1, num_predict=80)
    return _llm_cache


def _llm_recovery_decision(stage: dict, error_output: str, retries_so_far: int) -> tuple[str, str]:
    """
    Ask the LLM what to do with a failed stage.
    Returns (decision, reason) where decision is RETRY / SKIP / HALT.
    Guardrails are applied AFTER the LLM responds.
    """
    if not ollama_is_available():
        # Deterministic fallback: optional → SKIP, required → HALT
        decision = "SKIP" if stage["optional"] else "HALT"
        return decision, pipeline_fallback_note(stage["name"], error_output[:120])

    prompt = (
        f"Stage '{stage['name']}' (optional={stage['optional']}) failed "
        f"after {retries_so_far} retries.\n\n"
        f"Error output (last 400 chars):\n{error_output[-400:]}\n\n"
        f"What should the pipeline do?"
    )

    try:
        llm = _get_llm()
        result = llm.invoke([
            SystemMessage(content=_RECOVERY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        text  = result.content.strip()
        lines = text.split("\n", 1)
        raw_decision = lines[0].strip().upper()
        reason       = lines[1].strip() if len(lines) > 1 else "No reason provided."

        if raw_decision not in {"RETRY", "SKIP", "HALT"}:
            raw_decision = "SKIP" if stage["optional"] else "HALT"
            reason = "LLM returned invalid decision — applied default rule."

        return raw_decision, reason

    except Exception as exc:
        log.warning("LLM recovery decision failed: %s — using default", exc)
        decision = "SKIP" if stage["optional"] else "HALT"
        return decision, f"LLM error ({exc}) — applied default rule."


def _apply_guardrails(decision: str, stage: dict, retries_so_far: int) -> str:
    """
    Hard guardrails applied AFTER LLM decision — code always wins over LLM.
    """
    # Guardrail 1: never retry no_retry stages
    if decision == "RETRY" and stage.get("no_retry"):
        log.warning("[guardrail] LLM said RETRY for no_retry stage '%s' — overriding to HALT", stage["name"])
        return "HALT"

    # Guardrail 2: retry budget exhausted
    if decision == "RETRY" and retries_so_far >= MAX_RETRIES:
        log.warning("[guardrail] Retry budget exhausted for '%s' — overriding RETRY to %s",
                    stage["name"], "SKIP" if stage["optional"] else "HALT")
        return "SKIP" if stage["optional"] else "HALT"

    return decision


# ── Stage runner ───────────────────────────────────────────────────────────────

def _run_stage(stage: dict) -> tuple[bool, str]:
    """Run a stage as a subprocess. Returns (success, output)."""
    cmd = [sys.executable, "-W", "ignore", "-m",
           stage["script"].replace("/", ".").replace(".py", "")]
    log.info("[stage %d] Running: %s", stage["id"], stage["name"])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=600,
    )
    combined = proc.stdout + proc.stderr
    success  = proc.returncode == 0
    if success:
        log.info("[stage %d] %s — SUCCESS", stage["id"], stage["name"])
    else:
        log.error("[stage %d] %s — FAILED (rc=%d)", stage["id"], stage["name"], proc.returncode)
    return success, combined


# ── Main supervisor loop ───────────────────────────────────────────────────────

def run_supervised_pipeline(
    from_stage: int = 1,
    only_stage: str | None = None,
) -> PipelineReport:
    """
    Run the full pipeline with AI-assisted failure recovery.

    Args:
        from_stage:  Skip stages before this ID.
        only_stage:  Run only the stage with this name.

    Returns:
        PipelineReport (Pydantic validated).
    """
    run_at = datetime.datetime.now().isoformat()
    log.info("=" * 60)
    log.info("PIPELINE SUPERVISOR AGENT — %s", run_at)
    log.info("=" * 60)

    stage_results: list[StageResult] = []
    halt_requested = False
    upstream_failed = False

    stages_to_run = [
        s for s in STAGES
        if s["id"] >= from_stage
        and (only_stage is None or s["name"] == only_stage)
    ]

    for stage in stages_to_run:
        # Skip if upstream failure halted required stages
        if upstream_failed and not stage["optional"]:
            stage_results.append(StageResult(
                stage_id=stage["id"], stage_name=stage["name"],
                outcome=StageOutcome.SKIPPED,
                note="Skipped — upstream required stage failed.",
            ))
            continue

        retries   = 0
        outcome   = StageOutcome.FAILED
        last_note = None

        while True:
            success, output = _run_stage(stage)

            if success:
                outcome = StageOutcome.RETRIED if retries > 0 else StageOutcome.SUCCESS
                last_note = f"Succeeded after {retries} retry attempt(s)." if retries > 0 else None
                break

            # Stage failed — ask LLM what to do
            decision, reason = _llm_recovery_decision(stage, output, retries)
            decision = _apply_guardrails(decision, stage, retries)
            log.info("[stage %d] LLM decision: %s — %s", stage["id"], decision, reason)
            last_note = f"LLM: {decision} — {reason}"

            if decision == "RETRY":
                retries += 1
                log.info("[stage %d] Retrying (attempt %d/%d)...", stage["id"], retries, MAX_RETRIES)
                continue

            elif decision == "SKIP":
                outcome = StageOutcome.SKIPPED
                break

            else:  # HALT
                outcome = StageOutcome.FAILED
                if not stage["optional"]:
                    upstream_failed = True
                    halt_requested  = True
                break

        stage_results.append(StageResult(
            stage_id=stage["id"],
            stage_name=stage["name"],
            outcome=outcome,
            retries=retries,
            note=last_note,
        ))

        if halt_requested:
            log.error("Pipeline halted at stage %d (%s).", stage["id"], stage["name"])
            break

    # ── Build report ───────────────────────────────────────────────────────────
    succeeded      = sum(1 for s in stage_results if s.outcome in {StageOutcome.SUCCESS, StageOutcome.RETRIED})
    failed         = sum(1 for s in stage_results if s.outcome == StageOutcome.FAILED)
    skipped        = sum(1 for s in stage_results if s.outcome == StageOutcome.SKIPPED)
    auto_recovered = sum(1 for s in stage_results if s.outcome == StageOutcome.RETRIED)
    degraded       = sum(1 for s in stage_results if s.outcome == StageOutcome.DEGRADED)

    if failed > 0:
        overall = "FAILED"
    elif degraded > 0 or auto_recovered > 0:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    parts = []
    if auto_recovered:
        parts.append(f"{auto_recovered} stage(s) auto-recovered after retry")
    if failed:
        failed_names = [s.stage_name for s in stage_results if s.outcome == StageOutcome.FAILED]
        parts.append(f"{failed} stage(s) failed: {', '.join(failed_names)}")
    if skipped:
        parts.append(f"{skipped} stage(s) skipped")
    if not parts:
        parts.append("all stages completed successfully")

    incident_summary = (
        f"Pipeline {overall}. {succeeded}/{len(stage_results)} stages succeeded. "
        + " | ".join(parts) + "."
    )

    report = PipelineReport(
        run_at=run_at,
        total_stages=len(stage_results),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        auto_recovered=auto_recovered,
        degraded=degraded,
        stages=stage_results,
        incident_summary=incident_summary,
        overall_status=overall,
    )

    # ── Write pipeline_status.json ─────────────────────────────────────────────
    status_data = {
        "run_at":           run_at,
        "overall_status":   overall,
        "incident_summary": incident_summary,
        "auto_recovered":   auto_recovered,
        "stages": [
            {
                "id":       s.stage_id,
                "name":     s.stage_name,
                "outcome":  s.outcome.value,
                "retries":  s.retries,
                "note":     s.note,
            }
            for s in stage_results
        ],
    }
    PIPELINE_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATUS_JSON.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
    log.info("Pipeline status written -> %s", PIPELINE_STATUS_JSON)

    log.info("=" * 60)
    log.info("SUPERVISOR DONE: %s", incident_summary)
    log.info("=" * 60)

    return report


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Pipeline Supervisor")
    parser.add_argument("--from",  dest="from_stage", type=int, default=1,
                        help="Start from stage N (skip earlier stages)")
    parser.add_argument("--only",  dest="only_stage", type=str, default=None,
                        help="Run only stage NAME")
    args = parser.parse_args()

    report = run_supervised_pipeline(from_stage=args.from_stage, only_stage=args.only_stage)
    print(f"\n{'='*60}")
    print(f"STATUS  : {report.overall_status}")
    print(f"SUMMARY : {report.incident_summary}")
    print(f"{'='*60}")
    sys.exit(0 if report.overall_status != "FAILED" else 1)
