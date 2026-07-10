"""
Phase 7 — Chronological blocked splits with purge.

Assigns each row of the training frame to ``train`` / ``val`` / ``test``
based on the day-of-month string in its ``datetime``.  A configurable purge
gap is applied to the tail of each split so a long-horizon label at the
last train row cannot depend on the first val row's truth (and similarly
for val → test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tejas.config import (
    FORECAST_TRAIN_DAYS,
    FORECAST_VAL_DAYS,
    FORECAST_TEST_DAYS,
    FORECAST_SPLIT_PURGE_S,
)


SPLIT_NAMES = ("train", "val", "test")


# ── Split assignment ────────────────────────────────────────────────────────


def assign_split(
    df: pd.DataFrame,
    train_days: list[str] | None = None,
    val_days: list[str] | None = None,
    test_days: list[str] | None = None,
) -> pd.Series:
    """Return a Series of ``train`` / ``val`` / ``test`` / ``""`` labels.

    Rows whose day-of-month string is not in any of the three lists get an
    empty label and should be dropped by callers.

    Args:
        df: DataFrame with a ``datetime`` column.
        train_days / val_days / test_days: Overrides for the config
            defaults, primarily for testing.

    Returns:
        Object dtype Series aligned with ``df.index``.
    """
    train = set(train_days) if train_days is not None else set(FORECAST_TRAIN_DAYS)
    val = set(val_days) if val_days is not None else set(FORECAST_VAL_DAYS)
    test = set(test_days) if test_days is not None else set(FORECAST_TEST_DAYS)

    if "datetime" not in df.columns:
        raise KeyError("assign_split requires a 'datetime' column")

    # ``.dt.day`` returns int; format to two-digit string to match config.
    day_str = df["datetime"].dt.day.map(lambda d: f"{int(d):02d}")

    labels = pd.Series("", index=df.index, dtype=object)
    labels[day_str.isin(train)] = "train"
    labels[day_str.isin(val)] = "val"
    labels[day_str.isin(test)] = "test"
    return labels


# ── Purge ──────────────────────────────────────────────────────────────────


def apply_purge(
    df: pd.DataFrame,
    split_labels: pd.Series,
    purge_s: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the last ``purge_s`` seconds of the train and val blocks.

    Concretely: the last ``purge_s`` seconds of ``train`` rows (measured on
    ``datetime``) are dropped so a 60-min-horizon label there cannot peek
    into val.  Same for the end of ``val`` w.r.t. test.  The ``test`` tail
    is left alone (no split follows it).

    Args:
        df: DataFrame with a ``datetime`` column.
        split_labels: Output of :func:`assign_split`, aligned with ``df``.
        purge_s: Purge gap in seconds; defaults to
            :data:`FORECAST_SPLIT_PURGE_S`.

    Returns:
        (filtered DataFrame, filtered split-labels Series).
    """
    if purge_s is None:
        purge_s = FORECAST_SPLIT_PURGE_S
    if df.empty:
        return df.copy(), split_labels.copy()

    keep = np.ones(len(df), dtype=bool)
    ts = df["datetime"].to_numpy()

    for split_name in ("train", "val"):
        mask = (split_labels == split_name).to_numpy()
        if not mask.any():
            continue
        block_end = ts[mask].max()
        cutoff = block_end - np.timedelta64(int(purge_s), "s")
        purge_mask = mask & (ts > cutoff)
        keep &= ~purge_mask

    return df.loc[keep].reset_index(drop=True), split_labels[keep].reset_index(drop=True)


# ── Materialise arrays for training ─────────────────────────────────────────


def build_split_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    split_labels: pd.Series,
) -> dict:
    """Return X / y arrays for each of the three splits.

    Args:
        df: DataFrame containing feature and label columns.
        feature_cols: Feature column list (from
            :func:`sampling.select_feature_columns`).
        label_col: Name of the label column (e.g. ``label_flare_within_5m``).
        split_labels: Row-wise split membership.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``.  Each array is a fresh copy.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column missing: {label_col}")
    if len(split_labels) != len(df):
        raise ValueError("split_labels length mismatch with df")

    out = {}
    for name in SPLIT_NAMES:
        mask = (split_labels == name).to_numpy()
        sub = df.loc[mask]
        out[f"X_{name}"] = sub[feature_cols].to_numpy(copy=True)
        out[f"y_{name}"] = sub[label_col].fillna(0).astype(np.int8).to_numpy(copy=True)
    return out
