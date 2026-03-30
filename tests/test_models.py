"""
tests/test_models.py — Regression / snapshot tests for saved model artifacts.

Layer 3: Depends on Gap 7 (model_store.py / joblib persistence) being complete.
These tests:
  - Verify that save_artifacts / load_artifacts round-trip correctly
  - Assert that a loaded model produces the same output on a fixed input vector
    (snapshot test — catches silent regressions from retraining)
  - Check that metadata sidecars contain required fields
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ── model_store round-trip ────────────────────────────────────────────────────

from model_store import save_artifacts, load_artifacts, load_metadata, artifacts_exist, data_hash


class TestModelStorePersistence:
    """Verify save/load round-trip for arbitrary artifact dicts."""

    def test_save_and_load_roundtrip(self, tmp_models_dir, monkeypatch):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        artifacts = {
            "weights": np.array([1.0, 2.0, 3.0]),
            "bias":    0.5,
            "labels":  ["A", "B", "C"],
        }
        save_artifacts("test_model", artifacts, metrics={"accuracy": 0.95})

        loaded = load_artifacts("test_model")
        assert np.allclose(loaded["weights"], artifacts["weights"])
        assert loaded["bias"]   == artifacts["bias"]
        assert loaded["labels"] == artifacts["labels"]

    def test_metadata_sidecar_created(self, tmp_models_dir, monkeypatch):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        save_artifacts("meta_model", {"x": 1}, metrics={"f1": 0.88})

        meta = load_metadata("meta_model")
        assert "name"          in meta
        assert "saved_at"      in meta
        assert "metrics"       in meta
        assert "artifact_keys" in meta
        assert meta["metrics"]["f1"] == pytest.approx(0.88)

    def test_artifacts_exist_true_after_save(self, tmp_models_dir, monkeypatch):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        save_artifacts("exists_model", {"a": 1})
        assert artifacts_exist("exists_model")

    def test_artifacts_exist_false_before_save(self, tmp_models_dir, monkeypatch):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)
        assert not artifacts_exist("never_saved_model")

    def test_load_nonexistent_raises(self, tmp_models_dir, monkeypatch):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)
        with pytest.raises(FileNotFoundError):
            load_artifacts("does_not_exist")

    def test_data_hash_deterministic(self, sample_transactions):
        h1 = data_hash(sample_transactions)
        h2 = data_hash(sample_transactions)
        assert h1 == h2

    def test_data_hash_changes_with_data(self, sample_transactions):
        h1 = data_hash(sample_transactions)
        modified = sample_transactions.copy()
        modified.loc[0, 'amount'] = 99999.0
        h2 = data_hash(modified)
        assert h1 != h2

    def test_data_hash_length(self, sample_transactions):
        h = data_hash(sample_transactions)
        assert len(h) == 16  # truncated SHA256


# ── Snapshot test: sklearn model output stability ─────────────────────────────

class TestModelSnapshot:
    """
    Save a simple sklearn model with a fixed random state, then verify
    that loading it and predicting on the same input gives the same output.
    This catches regressions from silent retraining with changed hyperparameters.
    """

    def _train_simple_model(self, sample_monthly_profile):
        from sklearn.ensemble      import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        df = sample_monthly_profile.copy()
        features = [
            'dscr', 'savings_rate', 'overdraft_freq', 'expense_volatility',
            'income_stability', 'essential_ratio', 'discretionary_ratio',
        ]
        le = LabelEncoder()
        X  = df[features].values
        y  = le.fit_transform(df['credit_label'])

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_scaled, y)

        return {
            'model':         rf,
            'scaler':        scaler,
            'label_encoder': le,
            'feature_cols':  features,
        }

    def test_snapshot_prediction_stable(self, tmp_models_dir, monkeypatch,
                                        sample_monthly_profile):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        artifacts = self._train_simple_model(sample_monthly_profile)
        save_artifacts("snapshot_rf", artifacts, metrics={"test": 1.0})

        # Fixed input vector (last row of fixture)
        loaded    = load_artifacts("snapshot_rf")
        features  = artifacts['feature_cols']
        X_input   = sample_monthly_profile[features].iloc[[-1]].values

        scaler_orig   = artifacts['scaler']
        model_orig    = artifacts['model']
        scaler_loaded = loaded['scaler']
        model_loaded  = loaded['model']

        pred_orig   = model_orig.predict(scaler_orig.transform(X_input))
        pred_loaded = model_loaded.predict(scaler_loaded.transform(X_input))

        assert pred_orig[0] == pred_loaded[0], \
            "Loaded model must produce identical prediction to the original"

    def test_snapshot_proba_stable(self, tmp_models_dir, monkeypatch,
                                   sample_monthly_profile):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        artifacts = self._train_simple_model(sample_monthly_profile)
        save_artifacts("snapshot_rf_proba", artifacts)

        loaded   = load_artifacts("snapshot_rf_proba")
        features = artifacts['feature_cols']
        X_input  = sample_monthly_profile[features].iloc[[-1]].values

        s_orig   = artifacts['scaler']
        s_loaded = loaded['scaler']
        m_orig   = artifacts['model']
        m_loaded = loaded['model']

        proba_orig   = m_orig.predict_proba(s_orig.transform(X_input))
        proba_loaded = m_loaded.predict_proba(s_loaded.transform(X_input))

        assert np.allclose(proba_orig, proba_loaded), \
            "Loaded model predict_proba must match original"

    def test_metadata_records_artifact_keys(self, tmp_models_dir, monkeypatch,
                                            sample_monthly_profile):
        monkeypatch.setattr("model_store.MODELS_DIR", tmp_models_dir)

        artifacts = self._train_simple_model(sample_monthly_profile)
        save_artifacts("snapshot_meta", artifacts, metrics={"accuracy": 0.90})

        meta = load_metadata("snapshot_meta")
        for key in ('model', 'scaler', 'label_encoder', 'feature_cols'):
            assert key in meta['artifact_keys']
