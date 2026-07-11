"""TEJAS Phase 8 — Alert Engine.

Public API:
    * :class:`ConfirmationGate` — single-horizon streaming confirmation gate.
    * :class:`MultiHorizonGate` — bundle of four gates (one per horizon)
      for use in the dashboard playback loop.
    * :func:`simulate_gate_on_frame` — offline retrospective simulation
      helper used by benchmarking scripts.
"""

from tejas.alerts.confirmation import (
    ConfirmationGate,
    MultiHorizonGate,
    IDLE,
    PENDING,
    CONFIRMED,
    COOLDOWN,
)

__all__ = [
    "ConfirmationGate",
    "MultiHorizonGate",
    "IDLE",
    "PENDING",
    "CONFIRMED",
    "COOLDOWN",
]
