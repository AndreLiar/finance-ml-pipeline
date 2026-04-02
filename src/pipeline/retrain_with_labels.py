"""
src/pipeline/retrain_with_labels.py — Retrain category classifier on corrected labels

Applies all ground truth category corrections (pattern + tx_id level) to the
full feature matrix, then retrains the category classifier and reports how much
accuracy improved over the baseline trained on uncorrected labels.

Run:
  py -3 -m src.pipeline.retrain_with_labels
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURES_XLSX, MODELS_DIR, DATA_DIR
from src.pipeline.label_loader import apply_category_corrections
from src.model_store import save_artifacts

log = logging.getLogger(__name__)

OUTPUT_XLSX = DATA_DIR / "corrected_model_results.xlsx"


def retrain_category_classifier() -> dict:
    """
    Load features, apply category corrections, retrain, compare to baseline.
    """
    # Load full feature matrix
    try:
        df = pd.read_excel(FEATURES_XLSX, sheet_name="Feature Matrix")
        full = pd.read_excel(FEATURES_XLSX, sheet_name="Full Data")
    except Exception as exc:
        log.error("Cannot load features.xlsx: %s", exc)
        return {"error": str(exc)}

    # Merge description + date + abs_amount into feature matrix for correction
    merge_cols = ["date_operation", "description", "abs_amount", "category"]
    available = [c for c in merge_cols if c in full.columns]
    if len(available) < 3:
        log.error("Full Data sheet missing required columns: %s", merge_cols)
        return {"error": "Missing columns in Full Data"}

    # Apply corrections to Full Data
    log.info("Applying category corrections to %d transactions...", len(full))
    full_corrected = apply_category_corrections(full)

    n_fixed = int(full_corrected["correction_applied"].sum())
    log.info("%d / %d transactions had their category corrected", n_fixed, len(full_corrected))

    # Map corrections back to feature matrix (join on index position — they're aligned)
    if len(df) == len(full_corrected):
        df["category_corrected"] = full_corrected["category"].values
    else:
        # Lengths differ (feature matrix may have dropped rows) — use original category
        log.warning(
            "Feature matrix (%d) and Full Data (%d) have different lengths — "
            "using original categories for non-matched rows",
            len(df), len(full_corrected)
        )
        df["category_corrected"] = df["category"] if "category" in df.columns else "UNKNOWN"

    # Build features
    text_features = [c for c in df.columns if c in [
        "abs_amount", "log_amount", "is_round_number", "is_weekend",
        "day_of_week", "month", "quarter", "week_of_year",
        "rolling_7d_spend", "rolling_30d_spend",
        "amount_vs_monthly_avg", "amount_vs_cat_avg", "amount_z_in_cat",
    ]]

    if not text_features:
        log.error("No numeric feature columns found in feature matrix")
        return {"error": "No numeric features"}

    X = df[text_features].fillna(0)

    # ── Baseline: original (uncorrected) labels ────────────────────────────────
    y_orig_raw = df.get("category", full["category"]).iloc[:len(df)]
    le_orig = LabelEncoder()
    y_orig = le_orig.fit_transform(y_orig_raw.fillna("UNKNOWN").str.upper())

    # ── Corrected labels ───────────────────────────────────────────────────────
    le_corr = LabelEncoder()
    y_corr = le_corr.fit_transform(df["category_corrected"].fillna("UNKNOWN").str.upper())

    # Chronological split (no shuffle — prevents temporal leakage)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_orig_train, y_orig_test = y_orig[:split], y_orig[split:]
    y_corr_train, y_corr_test = y_corr[:split], y_corr[split:]

    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # ── Train baseline model (original labels) ────────────────────────────────
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_baseline.fit(X_train, y_orig_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    acc_baseline = accuracy_score(y_orig_test, y_pred_baseline)
    log.info("Baseline accuracy (original labels): %.1f%%", acc_baseline * 100)

    # ── Train corrected model ─────────────────────────────────────────────────
    clf_corrected = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    clf_corrected.fit(X_train, y_corr_train)
    y_pred_corrected = clf_corrected.predict(X_test)
    acc_corrected = accuracy_score(y_corr_test, y_pred_corrected)
    log.info("Corrected accuracy (corrected labels): %.1f%%", acc_corrected * 100)

    # ── Detailed reports ───────────────────────────────────────────────────────
    # Present labels for test set only
    orig_present = sorted(set(y_orig_test) | set(y_pred_baseline))
    orig_names   = [le_orig.classes_[i] for i in orig_present if i < len(le_orig.classes_)]
    corr_present = sorted(set(y_corr_test) | set(y_pred_corrected))
    corr_names   = [le_corr.classes_[i] for i in corr_present if i < len(le_corr.classes_)]

    report_baseline  = classification_report(
        y_orig_test, y_pred_baseline,
        labels=orig_present, target_names=orig_names,
        output_dict=True, zero_division=0,
    )
    report_corrected = classification_report(
        y_corr_test, y_pred_corrected,
        labels=corr_present, target_names=corr_names,
        output_dict=True, zero_division=0,
    )

    log.info("Baseline report:\n%s",
             classification_report(y_orig_test, y_pred_baseline,
                                   labels=orig_present, target_names=orig_names, zero_division=0))
    log.info("Corrected report:\n%s",
             classification_report(y_corr_test, y_pred_corrected,
                                   labels=corr_present, target_names=corr_names, zero_division=0))

    # ── Category correction impact ─────────────────────────────────────────────
    correction_summary = (
        full_corrected[full_corrected["correction_applied"]]
        .groupby(["category"])  # category is now the corrected value
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # ── Save corrected model ───────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_artifacts("category_classifier_corrected", {
        "model": clf_corrected,
        "label_encoder": le_corr,
        "feature_cols": text_features,
    })
    log.info("Corrected model saved -> models/category_classifier_corrected")

    # ── Write results Excel ────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        # Comparison metrics
        metrics = pd.DataFrame({
            "metric": [
                "transactions_total", "transactions_corrected",
                "train_size", "test_size",
                "baseline_accuracy", "corrected_accuracy", "accuracy_delta",
                "baseline_macro_f1", "corrected_macro_f1",
            ],
            "value": [
                len(df), n_fixed,
                len(X_train), len(X_test),
                round(acc_baseline, 4), round(acc_corrected, 4),
                round(acc_corrected - acc_baseline, 4),
                round(report_baseline.get("macro avg", {}).get("f1-score", 0), 4),
                round(report_corrected.get("macro avg", {}).get("f1-score", 0), 4),
            ],
        })
        metrics.to_excel(writer, sheet_name="Comparison", index=False)
        correction_summary.to_excel(writer, sheet_name="Corrections Applied", index=False)
        full_corrected[["date_operation", "description", "abs_amount",
                        "category", "correction_applied"]].to_excel(
            writer, sheet_name="Full Corrected Data", index=False
        )

    return {
        "transactions_total":      len(df),
        "transactions_corrected":  n_fixed,
        "baseline_accuracy":       round(acc_baseline, 4),
        "corrected_accuracy":      round(acc_corrected, 4),
        "accuracy_delta":          round(acc_corrected - acc_baseline, 4),
        "baseline_macro_f1":       round(report_baseline.get("macro avg", {}).get("f1-score", 0), 4),
        "corrected_macro_f1":      round(report_corrected.get("macro avg", {}).get("f1-score", 0), 4),
        "output":                  str(OUTPUT_XLSX),
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("Retraining category classifier on corrected labels...")
    results = retrain_category_classifier()

    print("\n" + "=" * 60)
    print("CATEGORY CLASSIFIER — BASELINE vs CORRECTED")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<30}: {v}")

    if "accuracy_delta" in results:
        delta = results["accuracy_delta"]
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Accuracy change: {sign}{delta*100:.1f}pp from label corrections")
