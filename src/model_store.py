"""
model_store.py — Joblib-based model persistence helpers.

Each training run saves:
  models/<name>.joblib          — the trained model object(s)
  models/<name>_meta.json       — training date, data hash, feature list, metrics

The dashboard calls load_artifacts() instead of retraining from scratch.

Usage:
    # In training scripts:
    from model_store import save_artifacts, artifacts_exist

    save_artifacts("category_classifier", {
        "model": rf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
    }, metrics={"accuracy": 0.92, "f1_weighted": 0.91})

    # In dashboard:
    from model_store import load_artifacts, artifacts_exist

    if artifacts_exist("category_classifier"):
        art = load_artifacts("category_classifier")
        rf, scaler, le = art["model"], art["scaler"], art["label_encoder"]
    else:
        st.warning("Run train_models.py first.")
"""

import json
import hashlib
import joblib
from datetime import datetime
from pathlib import Path

from src.config import MODELS_DIR


def _meta_path(name: str) -> Path:
    return MODELS_DIR / f"{name}_meta.json"

def _model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"


def save_artifacts(name: str, artifacts: dict, metrics: dict = None, data_hash: str = None):
    """
    Serialize model artifacts to models/<name>.joblib and write metadata sidecar.

    Args:
        name:      Unique identifier for this model set (e.g. "category_classifier")
        artifacts: Dict of objects to serialize — must be joblib-serializable
        metrics:   Optional dict of evaluation metrics to record in the sidecar
        data_hash: Optional SHA256 hash of the input data for traceability
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model artifacts
    joblib.dump(artifacts, _model_path(name))

    # Write metadata sidecar
    meta = {
        "name":        name,
        "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_hash":   data_hash or "not_provided",
        "metrics":     metrics or {},
        "artifact_keys": list(artifacts.keys()),
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  [model_store] Saved '{name}' → {_model_path(name).name}")
    if metrics:
        metrics_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in metrics.items())
        print(f"  [model_store] Metrics: {metrics_str}")


def load_artifacts(name: str) -> dict:
    """
    Load model artifacts from models/<name>.joblib.
    Raises FileNotFoundError if the artifact does not exist.
    """
    path = _model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved artifacts for '{name}'. "
            f"Run the corresponding training script first."
        )
    artifacts = joblib.load(path)
    meta = load_metadata(name)
    print(f"  [model_store] Loaded '{name}' (saved {meta.get('saved_at', 'unknown')})")
    return artifacts


def load_metadata(name: str) -> dict:
    """Load the metadata sidecar for a saved model."""
    path = _meta_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def artifacts_exist(name: str) -> bool:
    """Return True if saved artifacts exist for this name."""
    return _model_path(name).exists()


def data_hash(df) -> str:
    """Compute a SHA256 hash of a DataFrame for data traceability."""
    import pandas as pd
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def list_artifacts() -> list:
    """List all saved artifact names with their metadata."""
    if not MODELS_DIR.exists():
        return []
    results = []
    for meta_file in sorted(MODELS_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            results.append(meta)
        except Exception:
            pass
    return results
