"""
Phase 7 — Streaming inference.

A :class:`Forecaster` loads the four persisted horizon models for a chosen
algorithm and exposes :meth:`predict_row` / :meth:`predict_dataframe`.

Two independent tripwires enforce no-future-data at the API surface:
  1. The feature list from ``feature_columns.json`` is authoritative — any
     column outside it is refused.
  2. Any column in :data:`FORECAST_DROP_COLUMNS` — labels, timestamps,
     segment ids — is refused explicitly with a clear error message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from tejas.config import (
    MODELS_DIR,
    FORECAST_DROP_COLUMNS,
    FORECAST_LABEL_HORIZONS_MIN,
)
from tejas.forecasting.training import load_feature_columns_manifest


DEFAULT_ALGORITHM = "xgb"


# ── Forecaster ─────────────────────────────────────────────────────────────


@dataclass
class Forecaster:
    """Bundle of four horizon models + frozen feature contract."""

    algorithm: str
    feature_columns: list[str]
    horizon_models: dict = field(default_factory=dict)  # {horizon_min: artifact_dict}
    forbidden_columns: frozenset[str] = field(default_factory=frozenset)

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_row(self, row) -> dict[str, float]:
        """Return probabilities for all four horizons.

        Args:
            row: Mapping / pd.Series / dict of feature name → value.

        Returns:
            ``{"prob_5m": ..., "prob_15m": ..., "prob_30m": ..., "prob_60m": ...}``.
        """
        row = self._coerce_row(row)
        self._validate_columns(row.index)
        X = row.reindex(self.feature_columns).to_numpy(dtype=float).reshape(1, -1)
        out: dict[str, float] = {}
        for h in FORECAST_LABEL_HORIZONS_MIN:
            model = self.horizon_models[h]["model"]
            out[f"prob_{h}m"] = float(_predict_positive(model, X)[0])
        return out

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorised counterpart of :meth:`predict_row` for playback."""
        self._validate_columns(df.columns)
        X = df.reindex(columns=self.feature_columns).to_numpy(dtype=float)
        out = pd.DataFrame(index=df.index)
        for h in FORECAST_LABEL_HORIZONS_MIN:
            model = self.horizon_models[h]["model"]
            out[f"prob_{h}m"] = _predict_positive(model, X)
        return out

    def decision_thresholds(self) -> dict[int, float]:
        """Return the val-tuned decision threshold for each horizon."""
        return {
            h: float(self.horizon_models[h]["decision_threshold"])
            for h in FORECAST_LABEL_HORIZONS_MIN
        }

    # ── Validation ─────────────────────────────────────────────────────

    def _validate_columns(self, columns) -> None:
        cols = set(columns)

        illegal = cols & self.forbidden_columns
        if illegal:
            raise ValueError(
                "Forbidden columns passed to forecaster (label/timestamp "
                "leakage guard): " + ", ".join(sorted(illegal))
            )

        missing = set(self.feature_columns) - cols
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature column(s); first few: "
                + ", ".join(sorted(missing)[:5])
            )

    @staticmethod
    def _coerce_row(row) -> pd.Series:
        if isinstance(row, pd.Series):
            return row
        if isinstance(row, Mapping):
            return pd.Series(dict(row))
        raise TypeError(f"Unsupported row type: {type(row).__name__}")


# ── Loader ─────────────────────────────────────────────────────────────────


def load_forecaster(
    algorithm: str = DEFAULT_ALGORITHM,
    models_dir: Path | None = None,
) -> Forecaster:
    """Reload the four horizon models for one algorithm.

    Args:
        algorithm: One of ``logreg`` / ``xgb`` / ``lgbm``.
        models_dir: Override for :data:`MODELS_DIR`.

    Returns:
        A ready-to-use :class:`Forecaster`.
    """
    import joblib

    models_dir = Path(models_dir) if models_dir else Path(MODELS_DIR)
    feature_columns = load_feature_columns_manifest(models_dir=models_dir)

    horizon_models: dict[int, dict] = {}
    for h in FORECAST_LABEL_HORIZONS_MIN:
        path = models_dir / f"{algorithm}_horizon_{h}m.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact: {path}")
        artifact = joblib.load(path)
        horizon_models[h] = artifact

    return Forecaster(
        algorithm=algorithm,
        feature_columns=feature_columns,
        horizon_models=horizon_models,
        forbidden_columns=frozenset(FORECAST_DROP_COLUMNS),
    )


# ── Internal helpers ────────────────────────────────────────────────────────


def _predict_positive(model, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) for a fitted binary classifier."""
    if len(X) == 0:
        return np.zeros(0, dtype=float)
    probs = model.predict_proba(X)
    if probs.shape[1] == 1:
        try:
            class_label = int(model.classes_[0])
        except (AttributeError, TypeError):
            class_label = 1
        if class_label == 1:
            return probs[:, 0]
        return 1.0 - probs[:, 0]
    return probs[:, 1]
