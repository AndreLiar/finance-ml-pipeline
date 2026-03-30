"""
run_pipeline.py — Pipeline Orchestrator
========================================
Runs every pipeline stage in order, captures per-stage outcomes,
and writes data/pipeline_status.json so the dashboard can show
a warning banner when any stage failed on its last run.

Usage:
    python run_pipeline.py            # run all stages
    python run_pipeline.py --from 3   # start from stage 3
    python run_pipeline.py --only creditworthiness

Exit code: 0 if all stages succeeded, 1 if any stage failed.
"""

import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

from config import PIPELINE_STATUS_JSON, ROOT

# ── Stage definitions (in dependency order) ───────────────────────────────────
STAGES = [
    {"id": 1,  "name": "parse_statements",    "script": "parse_statements.py",    "optional": False},
    {"id": 2,  "name": "drift_check",         "script": "drift_check.py",         "optional": True},
    {"id": 3,  "name": "parse_livret_a",      "script": "parse_livret_a.py",      "optional": True},
    {"id": 4,  "name": "feature_engineering", "script": "feature_engineering.py", "optional": False},
    {"id": 5,  "name": "train_models",        "script": "train_models.py",        "optional": False},
    {"id": 6,  "name": "nlp_classifier",      "script": "nlp_classifier.py",      "optional": False},
    {"id": 7,  "name": "creditworthiness",    "script": "creditworthiness.py",    "optional": False},
    {"id": 8,  "name": "cashflow_forecast",   "script": "cashflow_forecast.py",   "optional": False},
    {"id": 9,  "name": "anomaly_detection",   "script": "anomaly_detection.py",   "optional": False},
    {"id": 10, "name": "loan_report",         "script": "loan_report.py",         "optional": True},
]


def run_stage(stage: dict, python: str) -> dict:
    """Run a single pipeline stage. Returns a result dict."""
    script = ROOT / stage["script"]
    print(f"\n{'─' * 60}")
    print(f"[{stage['id']}/{len(STAGES)}] {stage['name']}")
    print(f"{'─' * 60}")

    start = time.time()
    result = subprocess.run(
        [python, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = round(time.time() - start, 1)

    success = result.returncode == 0
    status  = "success" if success else "failed"

    # Print stage output (stdout) to console
    if result.stdout:
        print(result.stdout)
    if result.stderr and not success:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)

    print(f"  [{status.upper()}] {stage['name']} — {elapsed}s")

    return {
        "id":       stage["id"],
        "name":     stage["name"],
        "status":   status,
        "elapsed":  elapsed,
        "optional": stage["optional"],
        "error":    result.stderr.strip()[-500:] if not success else None,
    }


def write_status(results: list, overall: str):
    """Write pipeline_status.json for the dashboard to read."""
    PIPELINE_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "run_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall": overall,
        "stages":  results,
    }
    PIPELINE_STATUS_JSON.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nStatus written to: {PIPELINE_STATUS_JSON}")


def main():
    parser = argparse.ArgumentParser(description="Run the finance ML pipeline.")
    parser.add_argument("--from",  type=int, dest="from_id", default=1,
                        help="Start from this stage ID (default: 1)")
    parser.add_argument("--only",  type=str, default=None,
                        help="Run only this stage by name")
    args = parser.parse_args()

    python = sys.executable

    # Filter stages
    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s["name"] == args.only]
        if not stages:
            print(f"ERROR: Unknown stage '{args.only}'. Valid names: {[s['name'] for s in STAGES]}")
            sys.exit(1)
    elif args.from_id > 1:
        stages = [s for s in STAGES if s["id"] >= args.from_id]

    print("=" * 60)
    print("FINANCE ML PIPELINE ORCHESTRATOR")
    print("=" * 60)
    print(f"Python : {python}")
    print(f"Stages : {[s['name'] for s in stages]}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results      = []
    any_required_failed = False

    for stage in stages:
        result = run_stage(stage, python)
        results.append(result)

        if result["status"] == "failed":
            if stage["optional"]:
                print(f"  (optional stage — continuing)")
            else:
                print(f"\nREQUIRED STAGE FAILED: {stage['name']}")
                print("Downstream stages depend on this output — stopping pipeline.")
                any_required_failed = True
                # Mark remaining stages as skipped
                remaining = [s for s in stages if s["id"] > stage["id"]]
                for s in remaining:
                    results.append({
                        "id": s["id"], "name": s["name"],
                        "status": "skipped", "elapsed": 0,
                        "optional": s["optional"], "error": "upstream stage failed",
                    })
                break

    overall = "failed" if any_required_failed else "success"

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for r in results:
        icon = {"success": "✓", "failed": "✗", "skipped": "—"}.get(r["status"], "?")
        opt  = " (optional)" if r["optional"] else ""
        print(f"  {icon} {r['name']:<25} {r['status']:<10} {r['elapsed']}s{opt}")

    print(f"\nOverall: {overall.upper()}")
    write_status(results, overall)

    sys.exit(0 if not any_required_failed else 1)


if __name__ == "__main__":
    main()
