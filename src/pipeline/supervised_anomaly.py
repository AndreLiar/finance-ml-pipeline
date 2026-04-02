"""
src/pipeline/supervised_anomaly.py — Supervised anomaly classifier

Replaces the unsupervised Isolation Forest / LOF / One-Class SVM ensemble
with a supervised gradient boosting classifier trained on ground truth labels.

Ground truth source: data/labels/anomaly_labels.csv (44 labeled transactions)
Training strategy:
  - 44 labeled examples from the anomaly flags (10 positive, 34 negative)
  - Augmented with 200 high-confidence negative examples from non-flagged
    transactions (low-amount, recurring, common categories)
  - Features: amount, category_encoded, z-score, rolling ratio,
    is_round_number, is_weekend, vote_count from unsupervised ensemble
  - Model: GradientBoostingClassifier (handles imbalance, no scaling needed)
  - Evaluation: cross-validated precision, recall, F1 vs. majority baseline
  - Output: anomaly_results.xlsx updated with supervised_score column
            models/supervised_anomaly.joblib persisted

Run:
  py -3 -m src.pipeline.supervised_anomaly
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.config import ANOMALY_RESULTS_XLSX, FEATURES_XLSX, MODELS_DIR, DATA_DIR
from src.pipeline.label_loader import apply_anomaly_labels, add_tx_ids
from src.model_store import save_artifacts, load_artifacts

log = logging.getLogger(__name__)

MODEL_KEY = "supervised_anomaly"
OUTPUT_XLSX = DATA_DIR / "supervised_anomaly_results.xlsx"


# ── Feature builder ────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from the anomaly results DataFrame.
    Uses only columns available in anomaly_results.xlsx.
    """
    cat_encoder = LabelEncoder()
    df = df.copy()
    df["category_enc"] = cat_encoder.fit_transform(
        df["category"].fillna("UNKNOWN").str.upper()
    )

    feature_cols = [
        "abs_amount",
        "category_enc",
        "amount_vs_cat_avg",
        "amount_z_in_cat",
        "is_round_number",
        "is_weekend",
        "rolling_7d_spend",
        "rolling_30d_spend",
        "amount_vs_monthly_avg",
    ]
    # vote_count only present in flagged anomalies sheet — add if available
    if "vote_count" in df.columns:
        feature_cols.append("vote_count")

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols].fillna(0), cat_encoder


# ── Negative sample augmentation ──────────────────────────────────────────────

def _get_negative_augmentation(n: int = 200) -> pd.DataFrame:
    """
    Pull high-confidence negative examples from the full transaction set
    (transactions that were NOT flagged by the unsupervised ensemble).
    These are used to supplement the 34 labeled negatives.
    """
    try:
        all_tx = pd.read_excel(ANOMALY_RESULTS_XLSX, sheet_name="All Transactions")
    except Exception:
        try:
            all_tx = pd.read_excel(FEATURES_XLSX, sheet_name="Full Data")
        except Exception:
            log.warning("Could not load full transaction set for augmentation")
            return pd.DataFrame()

    # High-confidence negatives: not flagged, small amount, common categories
    safe_cats = {"GROCERIES", "TRANSPORT", "SALARY", "UTILITIES", "TELECOM", "INSURANCE"}
    mask = (
        (all_tx.get("is_anomaly", pd.Series(0, index=all_tx.index)) == 0) &
        (all_tx["abs_amount"] < 150) &
        (all_tx["category"].str.upper().isin(safe_cats))
    )
    negatives = all_tx[mask].copy()
    if len(negatives) > n:
        negatives = negatives.sample(n=n, random_state=42)
    negatives["label"] = 0
    negatives["vote_count"] = 0
    log.info("Augmented with %d high-confidence negative examples", len(negatives))
    return negatives


# ── Main training function ─────────────────────────────────────────────────────

def train_supervised_anomaly() -> dict:
    """
    Train supervised anomaly classifier on ground truth labels.

    Returns:
        dict with evaluation metrics and model info.
    """
    # Load labeled anomaly data
    try:
        flagged = pd.read_excel(ANOMALY_RESULTS_XLSX, sheet_name="Flagged Anomalies")
    except Exception as exc:
        log.error("Cannot load anomaly results: %s", exc)
        return {"error": str(exc)}

    flagged = add_tx_ids(flagged)
    flagged = apply_anomaly_labels(flagged)

    # Keep only rows with definitive labels
    labeled = flagged[flagged["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)

    if len(labeled) < 10:
        log.error("Not enough labeled examples: %d (need at least 10)", len(labeled))
        return {"error": f"Only {len(labeled)} labeled examples"}

    log.info(
        "Labeled set: %d total, %d positive (anomaly), %d negative (normal)",
        len(labeled), (labeled.label == 1).sum(), (labeled.label == 0).sum(),
    )

    # Augment with high-confidence negatives from full dataset
    augmented = _get_negative_augmentation(n=200)
    if not augmented.empty:
        # Add missing columns that labeled has
        for col in labeled.columns:
            if col not in augmented.columns:
                augmented[col] = 0
        train_df = pd.concat([labeled, augmented[labeled.columns]], ignore_index=True)
    else:
        train_df = labeled.copy()

    log.info("Training set after augmentation: %d rows", len(train_df))

    # Build features
    X, cat_encoder = _build_features(train_df)
    y = train_df["label"].astype(int)

    # ── Baseline: always predict majority class ────────────────────────────────
    majority_class = int(y.mode()[0])
    baseline_acc   = float((y == majority_class).mean())
    log.info("Majority class baseline accuracy: %.1f%%", baseline_acc * 100)

    # ── Train GBM classifier ───────────────────────────────────────────────────
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validated evaluation (stratified to handle imbalance)
    n_splits = min(5, int((y == 1).sum()))  # can't have more folds than positives
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
        cv_precision = cross_val_score(clf, X, y, cv=cv, scoring="precision")
        cv_recall    = cross_val_score(clf, X, y, cv=cv, scoring="recall")
        log.info(
            "CV F1=%.3f (+/-%.3f) | Precision=%.3f | Recall=%.3f",
            cv_scores.mean(), cv_scores.std(),
            cv_precision.mean(), cv_recall.mean(),
        )
    else:
        cv_scores = np.array([0.0])
        cv_precision = np.array([0.0])
        cv_recall    = np.array([0.0])

    # Final fit on all labeled data
    clf.fit(X, y)

    # ── In-sample evaluation on original 44 labeled anomalies ─────────────────
    X_orig, _ = _build_features(labeled)
    # re-encode categories using fitted encoder
    labeled_copy = labeled.copy()
    labeled_copy["category_enc"] = cat_encoder.transform(
        labeled_copy["category"].fillna("UNKNOWN").str.upper().map(
            lambda c: c if c in cat_encoder.classes_ else cat_encoder.classes_[0]
        )
    )
    X_orig, _ = _build_features(labeled_copy)

    y_pred = clf.predict(X_orig)
    y_orig = labeled["label"].astype(int)
    report = classification_report(y_orig, y_pred, target_names=["normal", "anomaly"], output_dict=True)

    log.info("In-sample on labeled 44:\n%s",
             classification_report(y_orig, y_pred, target_names=["normal", "anomaly"]))

    # ── Feature importance ─────────────────────────────────────────────────────
    feat_names = X.columns.tolist()
    importances = dict(zip(feat_names, clf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info("Top features: %s", top_features)

    # ── Score ALL transactions ─────────────────────────────────────────────────
    try:
        all_tx = pd.read_excel(ANOMALY_RESULTS_XLSX, sheet_name="All Transactions")
    except Exception:
        all_tx = flagged.copy()

    all_tx["vote_count"] = 0
    X_all, _ = _build_features(all_tx)
    all_tx["supervised_score"]   = clf.predict_proba(X_all)[:, 1]
    all_tx["supervised_anomaly"] = clf.predict(X_all)

    # ── Save model ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_artifacts(MODEL_KEY, {"model": clf, "cat_encoder": cat_encoder})
    log.info("Model saved -> models/%s", MODEL_KEY)

    # ── Write results ─────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        all_tx.sort_values("supervised_score", ascending=False).to_excel(
            writer, sheet_name="All Scored", index=False
        )
        all_tx[all_tx["supervised_anomaly"] == 1].sort_values(
            "supervised_score", ascending=False
        ).to_excel(writer, sheet_name="Supervised Anomalies", index=False)

        # Metrics sheet
        metrics_df = pd.DataFrame({
            "metric": [
                "labeled_examples", "positive_labels", "negative_labels",
                "augmented_negatives", "training_total",
                "cv_f1_mean", "cv_f1_std", "cv_precision_mean", "cv_recall_mean",
                "baseline_accuracy",
                "insample_precision_anomaly", "insample_recall_anomaly", "insample_f1_anomaly",
                "supervised_flags_total",
            ],
            "value": [
                len(labeled), int((labeled.label == 1).sum()), int((labeled.label == 0).sum()),
                len(augmented) if not augmented.empty else 0, len(train_df),
                round(cv_scores.mean(), 4), round(cv_scores.std(), 4),
                round(cv_precision.mean(), 4), round(cv_recall.mean(), 4),
                round(baseline_acc, 4),
                round(report.get("anomaly", {}).get("precision", 0), 4),
                round(report.get("anomaly", {}).get("recall", 0), 4),
                round(report.get("anomaly", {}).get("f1-score", 0), 4),
                int((all_tx["supervised_anomaly"] == 1).sum()),
            ],
        })
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    n_supervised = int((all_tx["supervised_anomaly"] == 1).sum())
    log.info(
        "Supervised anomaly detection complete: %d transactions flagged (vs %d unsupervised)",
        n_supervised, len(flagged),
    )

    return {
        "labeled":            len(labeled),
        "positive_labels":    int((labeled.label == 1).sum()),
        "negative_labels":    int((labeled.label == 0).sum()),
        "training_total":     len(train_df),
        "cv_f1":              round(float(cv_scores.mean()), 4),
        "cv_precision":       round(float(cv_precision.mean()), 4),
        "cv_recall":          round(float(cv_recall.mean()), 4),
        "baseline_accuracy":  round(baseline_acc, 4),
        "insample_f1":        round(report.get("anomaly", {}).get("f1-score", 0), 4),
        "supervised_flags":   n_supervised,
        "top_features":       top_features,
        "output":             str(OUTPUT_XLSX),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = train_supervised_anomaly()
    print("\n" + "=" * 60)
    print("SUPERVISED ANOMALY CLASSIFIER RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<28}: {v}")
